/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <map>

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/kernels/data/dataset.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.
class OptimizeDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit OptimizeDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx),
        graph_def_version_(ctx->graph_def_version()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    std::vector<string> optimizations;
    OP_REQUIRES_OK(
        ctx, ParseVectorArgument<string>(ctx, "optimizations", &optimizations));
    Dataset* dataset =
        new Dataset(ctx, input, optimizations, output_types_, output_shapes_);
    core::ScopedUnref unref(dataset);
    OP_REQUIRES_OK(ctx, dataset->Optimize(ctx, output));
  }

 private:
  class Dataset : public GraphDatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            const std::vector<string>& optimizations,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : GraphDatasetBase(ctx),
          input_(input),
          optimizations_(optimizations),
          output_types_(output_types),
          output_shapes_(output_shapes) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::Optimize")}));
    }

    Status Optimize(OpKernelContext* ctx, DatasetBase** output) {
      GraphDefBuilder b;
      DatasetGraphDefBuilder db(&b);
      Node* input_node = nullptr;
      TF_RETURN_IF_ERROR(db.AddParentDataset(ctx, input_, &input_node));
      string output_node = input_node->name();
      GraphDef graph_def;
      TF_RETURN_IF_ERROR(b.ToGraphDef(&graph_def));
      TF_RETURN_IF_ERROR(ApplyOptimizations(ctx, &graph_def, &output_node));

      Graph graph(OpRegistry::Global());
      TF_RETURN_IF_ERROR(ImportGraphDef({}, graph_def, &graph, nullptr));
      std::vector<Tensor> outputs;
      GraphRunner graph_runner(ctx->env());
      // Once rewrites that add/modify functions are introduced, we will need
      // persist the results in a function library runtime.
      TF_RETURN_IF_ERROR(graph_runner.Run(&graph, ctx->function_library(), {},
                                          {output_node}, &outputs));
      TF_RETURN_IF_ERROR(GetDatasetFromVariantTensor(outputs[0], output));
      (*output)->Ref();
      return Status::OK();
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override { return "OptimizeDatasetOp::Dataset"; }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status Initialize(IteratorContext* ctx) override {
        return errors::Unimplemented(strings::StrCat(prefix(), "::Initialize"));
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        return errors::Unimplemented(
            strings::StrCat(prefix(), "::GetNextInternal"));
      }
    };

    Status ApplyOptimizations(OpKernelContext* ctx, GraphDef* graph_def,
                              string* output_node) {
      // Add a fake sink node to allow rewriting the actual sink node.
      NodeDef* node = graph_def->mutable_node()->Add();
      node->set_name("FakeSink");
      node->set_op("IdentityDataset");
      node->add_input(*output_node);
      {
        grappler::GraphView graph(graph_def);
        NodeDef* sink = graph.GetNode(*output_node);
        (*node->mutable_attr())["output_shapes"] =
            sink->attr().at("output_shapes");
        (*node->mutable_attr())["output_types"] =
            sink->attr().at("output_types");
      }

      // Create metagraph.
      MetaGraphDef meta_graph_def;
      (*meta_graph_def.mutable_graph_def()) = *graph_def;

      // Grappler determines fetch ops from collection 'train_op'.
      CollectionDef collection_def;
      auto node_list = collection_def.mutable_node_list();
      node_list->add_value("FakeSink");
      (*meta_graph_def.mutable_collection_def())["train_op"] = collection_def;

      // Create Grappler item.
      tensorflow::RewriterConfig rewriter_config;
      for (const string& optimization : optimizations_) {
        rewriter_config.add_optimizers(optimization);
      }
      // If no optimizations were specified, supply a non-existent optimization
      // to prevent Grappler from applying the default set of optimizations as
      // some of them do not work out of the box at the moment (e.g. because we
      // have no cost model for dataset ops).
      if (optimizations_.empty()) {
        rewriter_config.add_optimizers("non-existent");
      }
      tensorflow::grappler::ItemConfig item_config;
      item_config.apply_optimizations = true;
      std::unique_ptr<tensorflow::grappler::GrapplerItem> grappler_item =
          tensorflow::grappler::GrapplerItemFromMetaGraphDef(
              "graph", meta_graph_def, item_config);
      std::unordered_map<string, tensorflow::DeviceProperties> device_map;
      tensorflow::grappler::VirtualCluster cluster(device_map);

      // Run optimizer.
      TF_RETURN_IF_ERROR(tensorflow::grappler::RunMetaOptimizer(
          *grappler_item, rewriter_config, ctx->device(), &cluster, graph_def));

      // Set `output_node` to the input of the fake sink node.
      {
        grappler::GraphView graph(graph_def);
        grappler::GraphView::InputPort input_port =
            graph.GetInputPort("FakeSink", 0);
        *output_node = graph.GetRegularFanin(input_port).node->name();
      }

      return Status::OK();
    }

    const DatasetBase* input_;
    const std::vector<string> optimizations_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
  };

  const int graph_def_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(Name("OptimizeDataset").Device(DEVICE_CPU),
                        OptimizeDatasetOp);

}  // namespace
}  // namespace tensorflow
