/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <deque>
#include <unordered_set>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/layout_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/frame.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace grappler {
namespace {

const char kPrefix[] = "LayoutOptimizer";
const char kDim[] = "LayoutOptimizerDim";
const char kPermNHWCToNCHW[] = "LayoutOptimizerPermConstNHWCToNCHW";
const char kPermNCHWToNHWC[] = "LayoutOptimizerPermConstNCHWToNHWC";
const char kGatherAxisConst[] = "LayoutOptimizerGatherAxisConst";
const char kTransposeNHWCToNCHW[] = "LayoutOptimizerTransposeNHWCToNCHW";
const char kTransposeNCHWToNHWC[] = "LayoutOptimizerTransposeNCHWToNHWC";
const char kPermVecNHWCToNCHW[] = "LayoutOptimizerPermVecNHWCToNCHW";
const char kReshapeNHWCToNCHW[] = "LayoutOptimizerReshapeNHWCToNCHW";
const char kReshapeConst[] = "LayoutOptimizerReshapeConst";
const char kReductionConst[] = "LayoutOptimizerReductionConst";

std::set<string> GetOpsFormatSupported() {
  std::set<string> ops_format_supported = {
      "AvgPool",
      "AvgPoolGrad",
      "Conv2D",
      "Conv2DBackpropFilter",
      "Conv2DBackpropInput",
      "BiasAdd",
      "BiasAddGrad",
      "DepthwiseConv2dNative",
      "DepthwiseConv2dNativeBackpropInput",
      "DepthwiseConv2dNativeBackpropFilter",
      "FusedBatchNorm",
      "FusedBatchNormGrad",
      "FusedConv2DBiasActivation",
      "MaxPool",
      "MaxPoolGrad",
      "SpaceToDepth",
      "DepthToSpace"};
  return ops_format_supported;
}

// TODO(yaozhang): enable SumProcessor with auto-tuning. Currently disabled
// because of the worse performance in some cases.
std::set<string> GetOpsFormatAgnostic() {
  std::set<string> ops_format_agnostic = {"Add",
                                          "AddN",
                                          "Concat",
                                          "ConcatV2",
                                          "Floor",
                                          "Identity",
                                          "Mul",
                                          "Neg",
                                          "Pad",
                                          "RealDiv",
                                          "Relu",
                                          "Relu6",
                                          "ReluGrad",
                                          "Sigmoid",
                                          "Slice",
                                          "Split",
                                          "SquaredDifference",
                                          "Squeeze",
                                          /*"Sum",*/ "Sub"};
  return ops_format_agnostic;
}

bool IsNodeByLayoutOptimizer(const string& node_name) {
  const string prefix_pattern = kPrefix;
  string prefix = node_name.substr(0, prefix_pattern.length());
  if (prefix.compare(prefix_pattern) == 0) {
    return true;
  }
  return false;
}

bool IsNodeNHWCToNCHW(const string& node_name) {
  const string transpose_node_prefix = kTransposeNHWCToNCHW;
  string prefix = node_name.substr(0, transpose_node_prefix.length());
  if (prefix.compare(transpose_node_prefix) == 0) {
    return true;
  }
  return false;
}

bool IsNodeNCHWToNHWC(const string& node_name) {
  const string transpose_node_prefix = kTransposeNCHWToNHWC;
  string prefix = node_name.substr(0, transpose_node_prefix.length());
  if (prefix.compare(transpose_node_prefix) == 0) {
    return true;
  }
  return false;
}

bool IsConcat(const NodeDef& node) {
  const auto op = node.op();
  return op == "Concat" || op == "ConcatV2";
}

bool IsConcatV1(const NodeDef& node) {
  const auto op = node.op();
  return op == "Concat";
}

bool IsMaxPoolGradV1(const NodeDef& node) {
  const auto& op = node.op();
  return op == "MaxPoolGrad";
}

class GraphProcessor {
 public:
  GraphProcessor(const VirtualPlacer& virtual_placer,
                 const std::unordered_set<string>& nodes_to_preserve,
                 GraphDef* graph, NodeMap* node_map)
      : virtual_placer_(virtual_placer),
        nodes_to_preserve_(nodes_to_preserve),
        graph_(graph),
        node_map_(node_map) {}

 protected:
  NodeDef* AddNodePermConst(const string& name, const string& device,
                            const std::vector<int>& permutation) {
    NodeDef* node = graph_->add_node();
    node_map_->AddNode(name, node);
    node->set_name(name);
    node->set_op("Const");
    AttrValue attr_data_type;
    attr_data_type.set_type(DT_INT32);
    node->mutable_attr()->insert({"dtype", attr_data_type});
    AttrValue attr_tensor;
    Tensor tensor(DT_INT32, TensorShape({4}));
    for (int i = 0; static_cast<size_t>(i) < permutation.size(); i++) {
      tensor.flat<int>()(i) = permutation[i];
    }
    tensor.AsProtoTensorContent(attr_tensor.mutable_tensor());
    node->mutable_attr()->insert({"value", attr_tensor});
    string device_name;
    if (device.empty()) {
      device_name = virtual_placer_.get_canonical_device_name(*node);
    } else {
      device_name = device;
    }
    node->set_device(device_name);
    return node;
  }

  NodeDef* AddNodeConstScalar(const string& name, const string& device,
                              DataType dtype, int value) {
    NodeDef* node = graph_->add_node();
    node_map_->AddNode(name, node);
    node->set_name(name);
    node->set_op("Const");
    AttrValue attr_data_type;
    attr_data_type.set_type(dtype);
    node->mutable_attr()->insert({"dtype", attr_data_type});
    AttrValue attr_tensor;
    Tensor tensor(dtype, TensorShape({}));
    tensor.scalar<int>()() = value;
    tensor.AsProtoTensorContent(attr_tensor.mutable_tensor());
    node->mutable_attr()->insert({"value", attr_tensor});
    string device_name;
    if (device.empty()) {
      device_name = virtual_placer_.get_canonical_device_name(*node);
    } else {
      device_name = device;
    }
    node->set_device(device_name);
    return node;
  }

  const VirtualPlacer& virtual_placer_;
  const std::unordered_set<string>& nodes_to_preserve_;
  GraphDef* graph_;
  NodeMap* node_map_;
};

struct OptimizeContext {
  OptimizeContext(GraphDef* graph, NodeDef* node, NodeMap* node_map,
                  const VirtualPlacer& virtual_placer,
                  const std::unordered_set<string>& nodes_to_preserve,
                  bool is_in_frame)
      : graph(graph),
        node(node),
        node_map(node_map),
        virtual_placer(virtual_placer),
        nodes_to_preserve(nodes_to_preserve),
        is_in_frame(is_in_frame) {}
  GraphDef* graph;
  NodeDef* node;
  NodeMap* node_map;
  const VirtualPlacer& virtual_placer;
  const std::unordered_set<string>& nodes_to_preserve;
  bool is_in_frame;
};

class NodeProcessor : public GraphProcessor {
 public:
  explicit NodeProcessor(const OptimizeContext& opt_cxt)
      : GraphProcessor(opt_cxt.virtual_placer, opt_cxt.nodes_to_preserve,
                       opt_cxt.graph, opt_cxt.node_map),
        node_(opt_cxt.node),
        is_in_frame_(opt_cxt.is_in_frame) {}
  virtual ~NodeProcessor() {}
  virtual Status ConvertNode() {
    if (ShouldProcess()) {
      UpdateAttrDataFormat();
      UpdateAttrKSize();
      UpdateAttrStrides();
      UpdateAttrShape();
      TF_RETURN_IF_ERROR(AddLayoutTransposeToInputs());
      TF_RETURN_IF_ERROR(AddLayoutTransposeToOutputs());
      TF_RETURN_IF_ERROR(CustomizedProcessing());
    }
    return Status::OK();
  }

 protected:
  bool IsDimsN(const NodeDef& node, int n) const {
    if (node.attr().find("_output_shapes") != node.attr().end()) {
      auto shape = node.attr().at("_output_shapes").list().shape(0);
      if (shape.dim_size() == n) {
        return true;
      }
    }
    return false;
  }

  bool IsDimsFour(const NodeDef& node) const { return IsDimsN(node, 4); }

  bool IsNHWC() const {
    if (node_->attr().find("data_format") != node_->attr().end()) {
      if (node_->attr().at("data_format").s().compare("NHWC") == 0) {
        return true;
      }
    }
    return false;
  }

  bool HasOutputs() const {
    auto outputs = node_map_->GetOutputs(node_->name());
    return !outputs.empty();
  }

  Status HasAttribute(const NodeDef& node, const string& attr) const {
    if (node.attr().find(attr) == node.attr().end()) {
      return Status(error::INVALID_ARGUMENT,
                    strings::StrCat("Missing attribute ", attr));
    }
    return Status::OK();
  }

  bool MustPreserve() const {
    return nodes_to_preserve_.find(node_->name()) != nodes_to_preserve_.end();
  }

  virtual bool ShouldProcess() const {
    return !MustPreserve() && IsNHWC() && IsDimsFour(*node_) && HasOutputs() &&
           IsOnGPU();
  }

  virtual bool IsOnGPU() const {
    string device_name;
    if (node_->device().empty()) {
      device_name = virtual_placer_.get_canonical_device_name(*node_);
    } else {
      device_name = node_->device();
    }
    string device;
    string not_used;
    if (DeviceNameUtils::SplitDeviceName(device_name, &not_used, &device) &&
        (StringPiece(str_util::Lowercase(device)))
            .contains(str_util::Lowercase(DEVICE_GPU))) {
      return true;
    }
    return false;
  }

  void UpdateAttrDataFormat() {
    if (node_->attr().find("data_format") != node_->attr().end()) {
      if (node_->attr().at("data_format").s().compare("NHWC") == 0) {
        string* data_format =
            node_->mutable_attr()->at("data_format").mutable_s();
        *data_format = "NCHW";
      }
    }
  }

  virtual void UpdateAttrShape() {
    if (node_->attr().find("_output_shapes") != node_->attr().end()) {
      auto shape = node_->mutable_attr()
                       ->at("_output_shapes")
                       .mutable_list()
                       ->mutable_shape(0);
      if (shape->dim_size() == 4) {
        int64 h = shape->dim(1).size();
        int64 w = shape->dim(2).size();
        int64 c = shape->dim(3).size();
        shape->mutable_dim(1)->set_size(c);
        shape->mutable_dim(2)->set_size(h);
        shape->mutable_dim(3)->set_size(w);
      }
    }
  }

  void UpdateAttrKSize() {
    if (node_->attr().find("ksize") != node_->attr().end()) {
      auto list = node_->mutable_attr()->at("ksize").mutable_list();
      UpdateTuple(list);
    }
  }

  void UpdateAttrStrides() {
    if (node_->attr().find("strides") != node_->attr().end()) {
      auto list = node_->mutable_attr()->at("strides").mutable_list();
      UpdateTuple(list);
    }
  }

  Status UpdateAttrValue(NodeDef* node) {
    TF_RETURN_IF_ERROR(HasAttribute(*node, "value"));
    Tensor tensor;
    auto success =
        tensor.FromProto(node->mutable_attr()->at({"value"}).tensor());
    if (!success) {
      LOG(ERROR) << "Failed to parse TensorProto.";
    }
    if (tensor.dims() == 1) {
      if (tensor.flat<int>().size() == 4) {
        int c = tensor.flat<int>()(3);
        tensor.flat<int>()(3) = tensor.flat<int>()(2);
        tensor.flat<int>()(2) = tensor.flat<int>()(1);
        tensor.flat<int>()(1) = c;
      } else if (tensor.flat<int>().size() == 3) {
        tensor.flat<int>()(0) = 0;
        tensor.flat<int>()(1) = 2;
        tensor.flat<int>()(2) = 3;
      } else {
        return Status(error::INVALID_ARGUMENT,
                      strings::StrCat("Unsupported tensor size: ",
                                      tensor.flat<int>().size()));
      }
    } else if (tensor.dims() == 2) {
      for (int i = 0; i < 2; i++) {
        int c = tensor.matrix<int>()(3, i);
        tensor.matrix<int>()(3, i) = tensor.matrix<int>()(2, i);
        tensor.matrix<int>()(2, i) = tensor.matrix<int>()(1, i);
        tensor.matrix<int>()(1, i) = c;
      }
    } else {
      return Status(
          error::INVALID_ARGUMENT,
          strings::StrCat("Unsupported dimension size: ", tensor.dims()));
    }
    tensor.AsProtoTensorContent(
        node->mutable_attr()->at({"value"}).mutable_tensor());
    return Status::OK();
  }

  Status UpdateAttrValueOfInput(int input_index) {
    auto input_node = node_map_->GetNode(node_->input(input_index));
    // We created a copy of the node, so that we don't modify the original node,
    // which might be used elsewhere. Note that this copy also copies the
    // control dependency input in the case this node is inside a loop,
    // to ensure added_node is in the same frame with node_.
    NodeDef* added_node = graph_->add_node();
    *added_node = *input_node;
    string base_name = strings::StrCat(node_->name(), "-", input_node->name());
    string node_name = AddPrefixToNodeName(base_name, "LayoutOptimizer", "-");
    added_node->set_name(node_name);
    *node_->mutable_input(input_index) = node_name;
    node_map_->AddNode(node_name, added_node);
    node_map_->AddOutput(node_name, node_->name());
    return UpdateAttrValue(added_node);
  }

  virtual std::vector<int> GetInputPos() const {
    std::vector<int> input_pos = {0};
    return input_pos;
  }

  virtual std::set<int> GetOutputPos() const {
    // For most nodes, no need to process control nodes or nodes that use an
    // output other than the first output: only the first output is of
    // 4D NCHW/NHWC format and thus relevant here.
    std::set<int> output_pos = {0};
    return output_pos;
  }

  NodeDef* AddNodeTranspose(const string& node_name, const string& input_name,
                            const string& const_name, DataType data_type,
                            const TensorShapeProto& input_shape,
                            bool NHWCToNCHW) {
    NodeDef* node = graph_->add_node();
    node_map_->AddNode(node_name, node);
    node->set_name(node_name);
    *node->add_input() = input_name;
    *node->add_input() = const_name;
    node->set_op("Transpose");
    node->set_device(node_->device());
    AttrValue attr_data_type;
    attr_data_type.set_type(data_type);
    node->mutable_attr()->insert({"T", attr_data_type});
    AttrValue attr_data_type_perm;
    attr_data_type_perm.set_type(DT_INT32);
    node->mutable_attr()->insert({"Tperm", attr_data_type_perm});
    if (!input_shape.unknown_rank()) {
      AttrValue attr_output_shape;
      auto output_shape = attr_output_shape.mutable_list()->add_shape();
      if (NHWCToNCHW) {
        output_shape->add_dim()->set_size(input_shape.dim(0).size());
        output_shape->add_dim()->set_size(input_shape.dim(3).size());
        output_shape->add_dim()->set_size(input_shape.dim(1).size());
        output_shape->add_dim()->set_size(input_shape.dim(2).size());
      } else {
        output_shape->add_dim()->set_size(input_shape.dim(0).size());
        output_shape->add_dim()->set_size(input_shape.dim(2).size());
        output_shape->add_dim()->set_size(input_shape.dim(3).size());
        output_shape->add_dim()->set_size(input_shape.dim(1).size());
      }
      node->mutable_attr()->insert({"_output_shapes", attr_output_shape});
    }
    return node;
  }

  virtual Status AddLayoutTransposeToInputs() {
    std::vector<int> input_pos = GetInputPos();
    for (const auto& pos : input_pos) {
      int output_pos;
      string input_node_name = ParseNodeName(node_->input(pos), &output_pos);
      string base_name =
          strings::StrCat(node_->name(), "-", input_node_name, "-", output_pos);
      string node_name =
          AddPrefixToNodeName(base_name, kTransposeNHWCToNCHW, "-");
      auto input_node = node_map_->GetNode(node_->input(pos));
      TF_RETURN_IF_ERROR(HasAttribute(*node_, "T"));
      TF_RETURN_IF_ERROR(HasAttribute(*input_node, "_output_shapes"));
      string const_name = GetOrAddNodePermNHWCToNCHW(pos);
      AddNodeTranspose(
          node_name, node_->input(pos), const_name,
          node_->attr().at("T").type(),
          input_node->attr().at("_output_shapes").list().shape(output_pos),
          true);
      node_map_->UpdateOutput(node_->input(pos), node_->name(), node_name);
      node_map_->AddOutput(node_name, node_->name());
      *node_->mutable_input(pos) = node_name;
    }
    return Status::OK();
  }

  virtual Status AddLayoutTransposeToOutputs() {
    auto outputs = node_map_->GetOutputs(node_->name());
    string const_name = GetOrAddNodePermNCHWToNHWC();
    for (const auto& output : outputs) {
      for (int i = 0; i < output->input_size(); i++) {
        auto& input = *output->mutable_input(i);
        int input_port;
        string input_name = ParseNodeName(input, &input_port);
        auto output_pos = GetOutputPos();
        if (input_name == node_->name() &&
            output_pos.find(input_port) != output_pos.end()) {
          string base_name =
              strings::StrCat(node_->name(), "-", output->name(), "-", i);
          string node_name =
              AddPrefixToNodeName(base_name, kTransposeNCHWToNHWC, "-");
          TF_RETURN_IF_ERROR(HasAttribute(*node_, "T"));
          TF_RETURN_IF_ERROR(HasAttribute(*node_, "_output_shapes"));
          AddNodeTranspose(
              node_name, input, const_name, node_->attr().at("T").type(),
              node_->attr().at("_output_shapes").list().shape(0), false);
          input = node_name;
          node_map_->AddOutput(node_->name(), node_name);
          node_map_->AddOutput(node_name, output->name());
        }
      }
      node_map_->RemoveOutput(node_->name(), output->name());
    }
    return Status::OK();
  }

  virtual Status CustomizedProcessing() { return Status::OK(); }

  NodeDef* AddNodePermNHWCToNCHW(const string& suffix,
                                 const string& depended_node,
                                 const string& device) {
    auto const_node = AddNodePermConst(
        strings::StrCat(kPermNHWCToNCHW, "-", suffix), device, {0, 3, 1, 2});
    // This is to ensure the transpose node and the const node are in the
    // same frame.
    *const_node->add_input() = AsControlDependency(depended_node);
    return const_node;
  }

  NodeDef* AddNodePermNCHWToNHWC(const string& suffix,
                                 const string& depended_node,
                                 const string& device) {
    auto const_node = AddNodePermConst(
        strings::StrCat(kPermNCHWToNHWC, "-", suffix), device, {0, 2, 3, 1});
    // This is to ensure the transpose node and the const node are in the same
    // frame.
    *const_node->add_input() = AsControlDependency(depended_node);
    return const_node;
  }

  NodeDef* node_;
  bool is_in_frame_;

 private:
  string GetOrAddNodePermNHWCToNCHW(int pos) {
    string const_name;
    if (is_in_frame_) {
      auto const_node = AddNodePermNHWCToNCHW(
          node_->input(pos), NodeName(node_->input(pos)), node_->device());
      const_name = const_node->name();
    } else {
      const_name = kPermNHWCToNCHW;
    }
    return const_name;
  }

  string GetOrAddNodePermNCHWToNHWC() {
    string const_name;
    if (is_in_frame_) {
      auto const_node =
          AddNodePermNCHWToNHWC(node_->name(), node_->name(), node_->device());
      const_name = const_node->name();
    } else {
      const_name = kPermNCHWToNHWC;
    }
    return const_name;
  }

  void UpdateTuple(AttrValue_ListValue* list) {
    int64 h = list->i(1);
    int64 w = list->i(2);
    int64 c = list->i(3);
    list->set_i(1, c);
    list->set_i(2, h);
    list->set_i(3, w);
  }
};

class AvgPoolGradProcessor : public NodeProcessor {
 public:
  explicit AvgPoolGradProcessor(const OptimizeContext& opt_cxt)
      : NodeProcessor(opt_cxt) {}

 protected:
  std::vector<int> GetInputPos() const override {
    std::vector<int> input_pos = {1};
    return input_pos;
  }
  Status CustomizedProcessing() override { return UpdateAttrValueOfInput(0); }
};

class BiasAddGradProcessor : public NodeProcessor {
 public:
  explicit BiasAddGradProcessor(const OptimizeContext& opt_cxt)
      : NodeProcessor(opt_cxt) {}

 protected:
  bool ShouldProcess() const override {
    if (MustPreserve()) {
      return false;
    }
    if (!IsOnGPU()) {
      return false;
    }
    auto input = node_map_->GetNode(node_->input(0));
    if (input) {
      if ((IsNHWC() && IsDimsFour(*input)) || IsNodeNCHWToNHWC(input->name())) {
        return true;
      }
    }
    return false;
  }

  Status AddLayoutTransposeToOutputs() override { return Status::OK(); }
};

class Conv2DProcessor : public NodeProcessor {
 public:
  Conv2DProcessor(const OptimizeContext& opt_cxt, bool no_gemm)
      : NodeProcessor(opt_cxt), no_gemm_(no_gemm) {}

 protected:
  bool ShouldProcess() const override {
    return !MustPreserve() && IsNHWC() && IsDimsFour(*node_) && HasOutputs() &&
           (!IsGemmUsed() || no_gemm_) && IsOnGPU();
  }

  TensorShapeProto GetShape(const string& input_name) const {
    string node_name;
    int output_pos;
    node_name = ParseNodeName(input_name, &output_pos);
    NodeDef* node = node_map_->GetNode(node_name);
    if (node->attr().find("_output_shapes") != node->attr().end()) {
      return node->attr().at("_output_shapes").list().shape(output_pos);
    }
    TensorShapeProto shape;
    return shape;
  }

  bool IsStrideOne() const {
    if (node_->attr().find("strides") != node_->attr().end()) {
      auto list = node_->attr().at("strides").list();
      return list.i(1) == 1 && list.i(2) == 1;
    }
    return false;
  }

  bool IsValidPadding() const {
    if (node_->attr().find("padding") != node_->attr().end()) {
      auto padding = node_->attr().at("padding").s();
      return padding == "VALID";
    }
    return false;
  }

  // The logic inside this function is based on the internal implementation of
  // Conv2D, Conv2DBackpropInput, and Conv2DBackpropFilter ops, and thus
  // needs to be updated accordingly if the internal implementation changes.
  bool IsGemmUsed(const TensorShapeProto& filter_shape,
                  const TensorShapeProto& input_shape) const {
    if (filter_shape.dim_size() == 4) {
      if (filter_shape.dim(0).size() == 1 && filter_shape.dim(1).size() == 1 &&
          IsStrideOne()) {
        return true;
      }
    }
    if (input_shape.dim_size() == 4 && filter_shape.dim_size() == 4) {
      if (input_shape.dim(1).size() == filter_shape.dim(0).size() &&
          input_shape.dim(2).size() == filter_shape.dim(1).size() &&
          IsValidPadding()) {
        return true;
      }
    }
    return false;
  }

  virtual bool IsGemmUsed() const {
    auto filter_shape = GetShape(node_->input(1));
    auto input_shape = GetShape(node_->input(0));
    return IsGemmUsed(filter_shape, input_shape);
  }

  bool no_gemm_;
};

class Conv2DBackpropFilterProcessor : public Conv2DProcessor {
 public:
  Conv2DBackpropFilterProcessor(const OptimizeContext& opt_cxt, bool no_gemm)
      : Conv2DProcessor(opt_cxt, no_gemm) {}

 protected:
  bool IsGemmUsed() const override {
    auto filter_shape = GetShape(node_->name());
    auto input_shape = GetShape(node_->input(0));
    return Conv2DProcessor::IsGemmUsed(filter_shape, input_shape);
  }

  std::vector<int> GetInputPos() const override {
    std::vector<int> input_pos = {0, 2};
    return input_pos;
  }

  Status AddLayoutTransposeToOutputs() override { return Status::OK(); }
  // No need to update output shape, as it is always of shape
  // [filter_height, filter_width, in_channels, out_channels], regardless of
  // whether NCHW or NHWC is used.
  void UpdateAttrShape() override {}
};

class Conv2DBackpropInputProcessor : public Conv2DProcessor {
 public:
  Conv2DBackpropInputProcessor(const OptimizeContext& opt_cxt, bool no_gemm)
      : Conv2DProcessor(opt_cxt, no_gemm) {}

 protected:
  bool IsGemmUsed() const override {
    auto filter_shape = GetShape(node_->input(1));
    auto input_shape = GetShape(node_->name());
    return Conv2DProcessor::IsGemmUsed(filter_shape, input_shape);
  }

  std::vector<int> GetInputPos() const override {
    std::vector<int> input_pos = {2};
    return input_pos;
  }

  Status CustomizedProcessing() override { return UpdateAttrValueOfInput(0); }
};

class FusedBatchNormGradProcessor : public NodeProcessor {
 public:
  explicit FusedBatchNormGradProcessor(const OptimizeContext& opt_cxt)
      : NodeProcessor(opt_cxt) {}

 protected:
  bool ShouldProcess() const override {
    return NodeProcessor::ShouldProcess() && IsTraining();
  }

  std::vector<int> GetInputPos() const override {
    std::vector<int> input_pos = {0, 1};
    return input_pos;
  }

 private:
  bool IsTraining() const {
    if (node_->attr().find("is_training") != node_->attr().end()) {
      if (node_->attr().at("is_training").b()) {
        return true;
      }
    }
    return false;
  }
};

class MaxPoolGradProcessor : public NodeProcessor {
 public:
  explicit MaxPoolGradProcessor(const OptimizeContext& opt_cxt)
      : NodeProcessor(opt_cxt) {}

 protected:
  std::vector<int> GetInputPos() const override {
    std::vector<int> input_pos = {0, 1, 2};
    return input_pos;
  }
};

class AgnosticNodeProcessor : public NodeProcessor {
 public:
  explicit AgnosticNodeProcessor(const OptimizeContext& opt_cxt)
      : NodeProcessor(opt_cxt) {}

 protected:
  bool ShouldProcess() const override {
    return !MustPreserve() && IsDimsFour(*node_) && HasOutputs() &&
           IsNodeAfterNCHWToNHWC() && IsOnGPU();
  }

  bool IsNodeAfterNCHWToNHWC() const {
    std::set<string> ops_format_agnostic = GetOpsFormatAgnostic();
    std::deque<NodeDef*> queue;
    auto first_node_pos = DataInputPos(*node_);
    for (const auto& pos : first_node_pos) {
      auto input_node = node_map_->GetNode(node_->input(pos));
      queue.push_back(input_node);
    }
    // The code will exit this while loop in one iteration in most cases, as the
    // graph is already topologically sorted.
    while (!queue.empty()) {
      NodeDef* current_node = queue.front();
      queue.pop_front();
      if (IsNodeNCHWToNHWC(current_node->name())) {
        return true;
      }
      // We only continue searching if the path is connected through
      // format-agnostic nodes.
      if (ops_format_agnostic.find(current_node->op()) !=
          ops_format_agnostic.end()) {
        auto current_node_pos = DataInputPos(*current_node);
        for (const auto& pos : current_node_pos) {
          auto input_node = node_map_->GetNode(current_node->input(pos));
          queue.push_back(input_node);
        }
      }
    }
    return false;
  }

 private:
  std::vector<int> DataInputPos(const NodeDef& node) const {
    std::vector<int> pos;
    if (IsSplit(node)) {
      return {1};
    }
    if (IsConcatV1(node)) {
      return {1};
    }
    if (IsAdd(node) || IsMul(node) || IsRealDiv(node) ||
        IsSquaredDifference(node) || IsSub(node)) {
      return {0, 1};
    }
    if (node.input_size() > 0 && !IsControlInput(node.input(0))) {
      return {0};
    }
    return {};
  }
};

class AddNProcessor : public AgnosticNodeProcessor {
 public:
  explicit AddNProcessor(const OptimizeContext& opt_cxt)
      : AgnosticNodeProcessor(opt_cxt) {}

 protected:
  std::vector<int> GetInputPos() const override {
    std::vector<int> input_pos;
    input_pos.reserve(node_->input_size());
    for (int i = 0; i < node_->input_size(); i++) {
      input_pos.push_back(i);
    }
    return input_pos;
  }
};

class BinaryOpProcessor : public AgnosticNodeProcessor {
 public:
  explicit BinaryOpProcessor(const OptimizeContext& opt_cxt)
      : AgnosticNodeProcessor(opt_cxt) {}

 protected:
  bool ShouldProcess() const override {
    return !MustPreserve() && IsDimsFour(*node_) && HasOutputs() &&
           IsNodeAfterNCHWToNHWC() &&
           (IsNDOperateWithMD(4, 0) || IsNDOperateWithMD(4, 1) ||
            IsNDOperateWithMD(4, 4) || IsNDOperateWithMD(0, 4) ||
            IsNDOperateWithMD(1, 4)) &&
           IsOnGPU();
  }

  std::vector<int> GetInputPos() const override {
    std::vector<int> input_pos;
    auto input0 = node_map_->GetNode(node_->input(0));
    auto input1 = node_map_->GetNode(node_->input(1));
    if (IsDimsFour(*input0)) {
      input_pos.push_back(0);
    }
    if (IsDimsFour(*input1)) {
      input_pos.push_back(1);
    }
    return input_pos;
  }

  bool IsDimsFour(const NodeDef& node) const {
    return NodeProcessor::IsDimsFour(node) || IsNodeNCHWToNHWC(node.name());
  }

  bool IsNDOperateWithMD(int n, int m) const {
    auto input0 = node_map_->GetNode(node_->input(0));
    auto input1 = node_map_->GetNode(node_->input(1));
    if (input0 && input1) {
      bool input0_is_n = (n == 4) ? IsDimsFour(*input0) : IsDimsN(*input0, n);
      bool input1_is_m = (m == 4) ? IsDimsFour(*input1) : IsDimsN(*input1, m);
      return input0_is_n && input1_is_m;
    }
    return false;
  }

  NodeDef* AddNodeShapeConst(const string& name, int num_channels) {
    NodeDef* node = graph_->add_node();
    node_map_->AddNode(name, node);
    node->set_name(name);
    node->set_op("Const");
    node->set_device(node_->device());
    AttrValue attr_data_type;
    attr_data_type.set_type(DT_INT32);
    node->mutable_attr()->insert({"dtype", attr_data_type});

    AttrValue attr_tensor;
    Tensor tensor(DT_INT32, TensorShape({4}));
    std::vector<int> shape = {1, num_channels, 1, 1};
    for (int i = 0; i < static_cast<int>(shape.size()); i++) {
      tensor.flat<int>()(i) = shape[i];
    }
    tensor.AsProtoTensorContent(attr_tensor.mutable_tensor());
    node->mutable_attr()->insert({"value", attr_tensor});
    return node;
  }

  NodeDef* AddNodeReshape(const string& node_name, const string& input_name,
                          const string& shape_const_node_name,
                          DataType data_type) {
    NodeDef* node = graph_->add_node();
    node_map_->AddNode(node_name, node);
    node->set_name(node_name);
    *node->add_input() = input_name;
    *node->add_input() = shape_const_node_name;
    node->set_op("Reshape");
    node->set_device(node_->device());

    AttrValue attr_type_indices;
    attr_type_indices.set_type(DT_INT32);
    node->mutable_attr()->insert({"Tshape", attr_type_indices});

    AttrValue attr_type_params;
    attr_type_params.set_type(data_type);
    node->mutable_attr()->insert({"T", attr_type_params});
    return node;
  }

  Status CustomizedProcessing() override {
    int vector_index = -1;
    if (IsNDOperateWithMD(4, 1)) {
      vector_index = 1;
    } else if (IsNDOperateWithMD(1, 4)) {
      vector_index = 0;
    }
    if (vector_index != -1) {
      string base_name =
          strings::StrCat(node_->name(), "-", node_->input(vector_index));
      string reshape_node_name =
          AddPrefixToNodeName(base_name, kReshapeNHWCToNCHW, "-");
      string shape_const_node_name =
          AddPrefixToNodeName(base_name, kReshapeConst, "-");
      auto input_node = node_map_->GetNode(node_->input(vector_index));
      TF_RETURN_IF_ERROR(HasAttribute(*input_node, "_output_shapes"));
      int vector_size =
          input_node->attr().at("_output_shapes").list().shape(0).dim(0).size();
      AddNodeShapeConst(shape_const_node_name, vector_size);
      TF_RETURN_IF_ERROR(HasAttribute(*node_, "T"));
      AddNodeReshape(reshape_node_name, node_->input(vector_index),
                     shape_const_node_name, node_->attr().at("T").type());
      node_map_->AddOutput(shape_const_node_name, reshape_node_name);
      node_map_->UpdateOutput(node_->input(vector_index), node_->name(),
                              reshape_node_name);
      node_map_->AddOutput(reshape_node_name, node_->name());
      *node_->mutable_input(vector_index) = reshape_node_name;
    }
    return Status::OK();
  }
};

class ConcatProcessor : public AgnosticNodeProcessor {
 public:
  explicit ConcatProcessor(const OptimizeContext& opt_cxt)
      : AgnosticNodeProcessor(opt_cxt) {
    // For Concat,  the concat axis is the first input; for ConcatV2,
    // the last input.
    axis_node_pos_ = (IsConcatV1(*node_)) ? 0 : (node_->input_size() - 1);
  }

 protected:
  std::vector<int> GetInputPos() const override {
    std::vector<int> input_pos;
    int start = (IsConcatV1(*node_)) ? 1 : 0;
    int end =
        (IsConcatV1(*node_)) ? node_->input_size() : (node_->input_size() - 1);
    for (int i = start; i < end; i++) {
      input_pos.push_back(i);
    }
    return input_pos;
  }

  Status CustomizedProcessing() override {
    auto dim_node = node_map_->GetNode(node_->input(axis_node_pos_));
    if (IsConstant(*dim_node)) {
      AddNodeDimConst();
    } else {
      AddNodeDataFormatDimMap();
    }
    return Status::OK();
  }

  int axis_node_pos_;

 private:
  void AddNodeDimConst() {
    auto dim_node = node_map_->GetNode(node_->input(axis_node_pos_));
    auto tensor = dim_node->attr().at({"value"}).tensor();
    int value = tensor.int_val(0);
    value = (value >= 0) ? value : value + 4;
    if (value == 1 || value == 2) {
      value = value + 1;
    } else if (value == 3) {
      value = 1;
    }
    // We created a copy of the node, so that we don't modify the original node,
    // which might be used elsewhere. Note that this copy also copies the
    // control dependency input in the case this node is inside a loop,
    // to ensure added_node is in the same frame with node_.
    NodeDef* added_node = graph_->add_node();
    *added_node = *dim_node;
    added_node->set_name(strings::StrCat(kDim, "-", node_->name()));
    node_map_->AddNode(added_node->name(), added_node);
    added_node->mutable_attr()->at({"value"}).mutable_tensor()->set_int_val(
        0, value);
    node_map_->RemoveOutput(node_->input(axis_node_pos_), node_->name());
    *node_->mutable_input(axis_node_pos_) = added_node->name();
    node_map_->AddOutput(added_node->name(), node_->name());
  }

  void AddNodeDataFormatDimMap() {
    NodeDef* added_node = graph_->add_node();
    added_node->set_name(strings::StrCat(kDim, "-", node_->name()));
    added_node->set_op("DataFormatDimMap");
    node_map_->AddNode(added_node->name(), added_node);
    added_node->set_device(node_->device());
    AttrValue attr_data_type;
    attr_data_type.set_type(DT_INT32);
    added_node->mutable_attr()->insert({"T", attr_data_type});
    AttrValue attr_format;
    attr_format.set_s("NHWC");
    added_node->mutable_attr()->insert({"src_format", attr_format});
    attr_format.set_s("NCHW");
    added_node->mutable_attr()->insert({"dst_format", attr_format});
    *added_node->add_input() = node_->input(axis_node_pos_);
    *node_->mutable_input(axis_node_pos_) = added_node->name();
    node_map_->UpdateOutput(added_node->input(0), node_->name(),
                            added_node->name());
    node_map_->AddOutput(added_node->name(), node_->name());
  }
};

class PadProcessor : public AgnosticNodeProcessor {
 public:
  explicit PadProcessor(const OptimizeContext& opt_cxt)
      : AgnosticNodeProcessor(opt_cxt) {}

 protected:
  bool ShouldProcess() const override {
    return !MustPreserve() && IsDimsFour(*node_) && HasOutputs() &&
           IsNodeAfterNCHWToNHWC() && PaddingSupported() && IsOnGPU();
  }
  Status CustomizedProcessing() override { return UpdateAttrValueOfInput(1); }

 private:
  bool PaddingSupported() const {
    auto pad_const = node_map_->GetNode(node_->input(1));
    bool is_const = IsConstant(*pad_const);
    bool is_4D = false;
    if (HasAttribute(*pad_const, "value").ok()) {
      Tensor tensor;
      if (tensor.FromProto(pad_const->mutable_attr()->at({"value"}).tensor())) {
        if (tensor.dims() == 2) {
          if (tensor.dim_size(0) == 4 && tensor.dim_size(1) == 2) {
            is_4D = true;
          }
        }
      }
    }
    return is_const && is_4D;
  }
};

class SplitProcessor : public ConcatProcessor {
 public:
  explicit SplitProcessor(const OptimizeContext& opt_cxt)
      : ConcatProcessor(opt_cxt) {
    axis_node_pos_ = 0;
  }

 protected:
  std::vector<int> GetInputPos() const override {
    std::vector<int> input_pos = {1};
    return input_pos;
  }

  std::set<int> GetOutputPos() const override {
    std::set<int> output_pos{0};
    if (HasAttribute(*node_, "num_split").ok()) {
      for (int i = 1; i < node_->attr().at("num_split").i(); i++) {
        output_pos.insert(i);
      }
    }
    return output_pos;
  }
};

class ReluGradProcessor : public AgnosticNodeProcessor {
 public:
  explicit ReluGradProcessor(const OptimizeContext& opt_cxt)
      : AgnosticNodeProcessor(opt_cxt) {}

 protected:
  std::vector<int> GetInputPos() const override {
    std::vector<int> input_pos = {0, 1};
    return input_pos;
  }
};

class SliceProcessor : public AgnosticNodeProcessor {
 public:
  explicit SliceProcessor(const OptimizeContext& opt_cxt)
      : AgnosticNodeProcessor(opt_cxt) {}

 protected:
  Status CustomizedProcessing() override {
    // Skip the first input, which is the data to be sliced.
    for (int i = 1; i < node_->input_size(); i++) {
      string base_name = strings::StrCat(node_->name(), "-input", i);
      string node_name =
          AddPrefixToNodeName(base_name, kPermVecNHWCToNCHW, "-");
      TF_RETURN_IF_ERROR(HasAttribute(*node_, "Index"));
      AddNodePermVec(node_name, node_->input(i), node_->device(),
                     node_->attr().at("Index").type(), true);
      node_map_->UpdateOutput(node_->input(i), node_->name(), node_name);
      node_map_->AddOutput(node_name, node_->name());
      *node_->mutable_input(i) = node_name;
    }
    return Status::OK();
  }

 private:
  NodeDef* AddNodeGatherAxisConst(const string& suffix,
                                  const string& depended_node,
                                  const string& device) {
    auto const_node = AddNodeConstScalar(
        strings::StrCat(kGatherAxisConst, "-", suffix), device, DT_INT32, 0);
    // This is to ensure the Slice node and the const node are
    // in the same frame.
    *const_node->add_input() = AsControlDependency(depended_node);
    return const_node;
  }

  string GetOrAddNodeGatherAxisConst() {
    string const_name;
    if (is_in_frame_) {
      auto const_node = AddNodeGatherAxisConst(
          node_->name(), NodeName(node_->input(0)), node_->device());
      const_name = const_node->name();
    } else {
      const_name = kGatherAxisConst;
    }
    return const_name;
  }

  string GetOrAddNodePermNHWCToNCHW() {
    string const_name;
    if (is_in_frame_) {
      auto const_node = AddNodePermNHWCToNCHW(
          node_->name(), NodeName(node_->input(0)), node_->device());
      const_name = const_node->name();
    } else {
      const_name = kPermNHWCToNCHW;
    }
    return const_name;
  }

  string GetOrAddNodePermNCHWToNHWC() {
    string const_name;
    if (is_in_frame_) {
      auto const_node = AddNodePermNCHWToNHWC(
          node_->name(), NodeName(node_->input(0)), node_->device());
      const_name = const_node->name();
    } else {
      const_name = kPermNCHWToNHWC;
    }
    return const_name;
  }

  void AddNodePermVec(const string& node_name, const string& input_name,
                      const string& device, DataType data_type,
                      bool NHWCToNCHW) {
    NodeDef* node = graph_->add_node();
    node_map_->AddNode(node_name, node);
    node->set_name(node_name);
    node->set_device(device);
    *node->add_input() = input_name;
    *node->add_input() = NHWCToNCHW ? GetOrAddNodePermNHWCToNCHW()
                                    : GetOrAddNodePermNCHWToNHWC();
    *node->add_input() = GetOrAddNodeGatherAxisConst();
    node->set_op("GatherV2");

    AttrValue attr_type_indices;
    attr_type_indices.set_type(DT_INT32);
    node->mutable_attr()->insert({"Tindices", attr_type_indices});

    AttrValue attr_type_axis;
    attr_type_axis.set_type(DT_INT32);
    node->mutable_attr()->insert({"Taxis", attr_type_axis});

    AttrValue attr_type_params;
    attr_type_params.set_type(data_type);
    node->mutable_attr()->insert({"Tparams", attr_type_params});
  }
};

// Specialized SliceProcessor, used if the second and third input are const
// nodes, which could be the case if a constant folding pass is applied
// before this optimization.
class SliceProcessorConst : public AgnosticNodeProcessor {
 public:
  explicit SliceProcessorConst(const OptimizeContext& opt_cxt)
      : AgnosticNodeProcessor(opt_cxt) {}

 protected:
  Status CustomizedProcessing() override {
    // Skip the first input, which is the data to be sliced.
    for (int i = 1; i < node_->input_size(); i++) {
      TF_RETURN_IF_ERROR(UpdateAttrValueOfInput(i));
    }
    return Status::OK();
  }
};

class SqueezeProcessor : public AgnosticNodeProcessor {
 public:
  explicit SqueezeProcessor(const OptimizeContext& opt_cxt)
      : AgnosticNodeProcessor(opt_cxt) {}

 protected:
  bool ShouldProcess() const override {
    return !MustPreserve() && IsDimsN(*node_, 2) && HasOutputs() &&
           IsNodeAfterNCHWToNHWC() && IsInputConvertible() && IsAlongDimHW() &&
           IsOnGPU();
  }

  Status AddLayoutTransposeToOutputs() override { return Status::OK(); }

  bool IsInputConvertible() const {
    auto input = node_map_->GetNode(node_->input(0));
    if (IsNodeNCHWToNHWC(input->name())) {
      input = node_map_->GetNode(input->input(0));
    }
    if (input->attr().find("_output_shapes") != input->attr().end()) {
      auto shape = input->attr().at("_output_shapes").list().shape(0);
      if (shape.dim_size() != 4) {
        return false;
      }
      if (shape.dim(1).size() == 1 && shape.dim(2).size() == 1) {
        return true;
      }
    }
    return false;
  }

  bool IsAlongDimHW() const {
    if (node_->attr().find("squeeze_dims") != node_->attr().end()) {
      auto list = node_->attr().at("squeeze_dims").list();
      if (list.i(0) == 1 && list.i(1) == 2) {
        return true;
      }
    }
    return false;
  }

  Status CustomizedProcessing() override {
    TF_RETURN_IF_ERROR(HasAttribute(*node_, "squeeze_dims"));
    auto list = node_->mutable_attr()->at("squeeze_dims").mutable_list();
    list->set_i(0, 2);
    list->set_i(1, 3);
    return Status::OK();
  }
};

class SumProcessor : public AgnosticNodeProcessor {
 public:
  explicit SumProcessor(const OptimizeContext& opt_cxt)
      : AgnosticNodeProcessor(opt_cxt) {}

 protected:
  bool ShouldProcess() const override {
    auto input0 = node_map_->GetNode(node_->input(0));
    return !MustPreserve() && HasOutputs() && IsNodeAfterNCHWToNHWC() &&
           (IsDimsFour(*input0) || IsNodeNCHWToNHWC(input0->name())) &&
           IsAlongDimNHW() && IsOnGPU();
  }

  Status AddLayoutTransposeToOutputs() override { return Status::OK(); }

  Status CustomizedProcessing() override { return UpdateAttrValueOfInput(1); }

 private:
  bool IsAlongDimNHW() const {
    NodeDef* reduction_indices = node_map_->GetNode(node_->input(1));
    if (!IsConstant(*reduction_indices)) {
      return false;
    }
    Tensor tensor;
    if (reduction_indices->attr().find({"value"}) ==
        reduction_indices->attr().end()) {
      return false;
    }
    auto success =
        tensor.FromProto(reduction_indices->attr().at({"value"}).tensor());
    if (!success) {
      LOG(ERROR) << "Failed to parse TensorProto.";
      return false;
    }
    if (tensor.flat<int>().size() != 3) {
      return false;
    }
    if (tensor.flat<int>()(0) == 0 && tensor.flat<int>()(1) == 1 &&
        tensor.flat<int>()(2) == 2) {
      return true;
    }
    return false;
  }
};

class DataLayoutOptimizer : GraphProcessor {
 public:
  explicit DataLayoutOptimizer(
      const VirtualPlacer& virtual_placer,
      const LayoutOptimizer::TuningConfig& config,
      const std::unordered_set<string>& nodes_to_preserve, GraphDef* graph,
      NodeMap* node_map)
      : GraphProcessor(virtual_placer, nodes_to_preserve, graph, node_map),
        config_(config) {}

  Status Optimize() {
    VLOG(1) << "Number of nodes for original graph: " << graph_->node_size();
    TF_RETURN_IF_ERROR(Expand());
    VLOG(1) << "Number of nodes after Expand: " << graph_->node_size();
    TF_RETURN_IF_ERROR(Collapse());
    VLOG(1) << "Number of nodes after Collapse: " << graph_->node_size();
    return Status::OK();
  }

 private:
  NodeDef* AddNodePermNHWCToNCHW() {
    return AddNodePermConst(kPermNHWCToNCHW, "", {0, 3, 1, 2});
  }

  NodeDef* AddNodePermNCHWToNHWC() {
    return AddNodePermConst(kPermNCHWToNHWC, "", {0, 2, 3, 1});
  }

  NodeDef* AddNodeGatherAxisConst() {
    return AddNodeConstScalar(kGatherAxisConst, "", DT_INT32, 0);
  }

  // Expand all nodes which is in NHWC, but supports NCHW or is layout agnostic.
  Status Expand() {
    int node_size_original = graph_->node_size();
    std::unordered_map<const NodeDef*, std::vector<int>> frames;
    int num_frames;
    TF_RETURN_IF_ERROR(IdentifyFrames(*graph_, &frames, &num_frames));

    // This is the first pass where we expand the nodes which support NCHW.
    std::set<string> ops_format_supported = GetOpsFormatSupported();
    for (int i = 0; i < node_size_original; i++) {
      if (IsNodeByLayoutOptimizer(graph_->node(i).name())) {
        return Status(error::INVALID_ARGUMENT,
                      "The graph is already optimized by layout optimizer.");
      }
      if (ops_format_supported.find(graph_->node(i).op()) !=
          ops_format_supported.end()) {
        auto node = graph_->mutable_node(i);
        bool is_in_frame = !frames[node].empty();
        OptimizeContext opt_cxt(graph_, node, node_map_, virtual_placer_,
                                nodes_to_preserve_, is_in_frame);
        std::unique_ptr<NodeProcessor> node_processor;
        if (IsAvgPoolGrad(*node)) {
          node_processor.reset(new AvgPoolGradProcessor(opt_cxt));
        } else if (IsBiasAddGrad(*node)) {
          node_processor.reset(new BiasAddGradProcessor(opt_cxt));
        } else if (IsConv2D(*node)) {
          node_processor.reset(new Conv2DProcessor(opt_cxt, config_.no_gemm));
        } else if (IsConv2DBackpropFilter(*node)) {
          node_processor.reset(
              new Conv2DBackpropFilterProcessor(opt_cxt, config_.no_gemm));
        } else if (IsConv2DBackpropInput(*node)) {
          node_processor.reset(
              new Conv2DBackpropInputProcessor(opt_cxt, config_.no_gemm));
        } else if (IsDepthwiseConv2dNative(*node)) {
          node_processor.reset(new Conv2DProcessor(opt_cxt, true));
        } else if (IsDepthwiseConv2dNativeBackpropFilter(*node)) {
          node_processor.reset(
              new Conv2DBackpropFilterProcessor(opt_cxt, true));
        } else if (IsDepthwiseConv2dNativeBackpropInput(*node)) {
          node_processor.reset(new Conv2DBackpropInputProcessor(opt_cxt, true));
        } else if (IsFusedBatchNormGradV1(*node)) {
          node_processor.reset(new FusedBatchNormGradProcessor(opt_cxt));
        } else if (IsMaxPoolGradV1(*node)) {
          node_processor.reset(new MaxPoolGradProcessor(opt_cxt));
        } else {
          node_processor.reset(new NodeProcessor(opt_cxt));
        }
        TF_RETURN_IF_ERROR(node_processor->ConvertNode());
      }
    }

    // This is the second pass where we expand layout-agnostic nodes. This pass
    // only needs to be performed if at least one node in the previous pass is
    // expanded.
    if (graph_->node_size() > node_size_original) {
      NodeDef* n = AddNodePermNHWCToNCHW();
      n = AddNodePermNCHWToNHWC();
      n = AddNodeGatherAxisConst();
      std::set<string> ops_format_agnostic = GetOpsFormatAgnostic();
      for (int i = 0; i < graph_->node_size(); i++) {
        if (ops_format_agnostic.find(graph_->node(i).op()) !=
            ops_format_agnostic.end()) {
          auto node = graph_->mutable_node(i);
          bool is_in_frame = !frames[node].empty();
          OptimizeContext opt_cxt(graph_, node, node_map_, virtual_placer_,
                                  nodes_to_preserve_, is_in_frame);
          std::unique_ptr<NodeProcessor> node_processor;
          if (IsAddN(*node)) {
            node_processor.reset(new AddNProcessor(opt_cxt));
          } else if (IsAdd(*node) || IsMul(*node) || IsRealDiv(*node) ||
                     IsSquaredDifference(*node) || IsSub(*node)) {
            node_processor.reset(new BinaryOpProcessor(opt_cxt));
          } else if (IsConcat(*node)) {
            node_processor.reset(new ConcatProcessor(opt_cxt));
          } else if (IsPad(*node)) {
            node_processor.reset(new PadProcessor(opt_cxt));
          } else if (IsReluGrad(*node)) {
            node_processor.reset(new ReluGradProcessor(opt_cxt));
          } else if (IsSlice(*node)) {
            auto input1 = node_map_->GetNode(NodeName(node->input(1)));
            auto input2 = node_map_->GetNode(NodeName(node->input(2)));
            if (IsConstant(*input1) && IsConstant(*input2)) {
              node_processor.reset(new SliceProcessorConst(opt_cxt));
            } else {
              node_processor.reset(new SliceProcessor(opt_cxt));
            }
          } else if (IsSplit(*node)) {
            node_processor.reset(new SplitProcessor(opt_cxt));
          } else if (IsSqueeze(*node)) {
            node_processor.reset(new SqueezeProcessor(opt_cxt));
          } else if (IsSum(*node)) {
            node_processor.reset(new SumProcessor(opt_cxt));
          } else {
            node_processor.reset(new AgnosticNodeProcessor(opt_cxt));
          }
          TF_RETURN_IF_ERROR(node_processor->ConvertNode());
        }
      }
    }
    return Status::OK();
  }

  // Remove all node pairs, where a NCHW-to-NHWC node is followed by
  // a NHWC-to-NCHW node.
  Status Collapse() {
    std::unordered_set<string> nodes_removable;
    for (int i = 0; i < graph_->node_size(); i++) {
      auto node = graph_->mutable_node(i);
      node->mutable_attr()->erase("_output_shapes");
      if (IsNodeNHWCToNCHW(node->name())) {
        if (IsNodeNCHWToNHWC(node->input(0))) {
          const string& trans_first = node->input(0);
          const string& trans_second = node->name();
          auto outputs = node_map_->GetOutputs(trans_second);
          CHECK(outputs.size() == 1)
              << "There is always only a single output for a Transpose node, "
              << "due to the way it is added by NodeProcessor.";
          NodeDef* output = *outputs.begin();
          string input = node_map_->GetNode(trans_first)->input(0);
          for (int i = 0; i < output->input_size(); i++) {
            if (output->input(i).compare(trans_second) == 0) {
              *output->mutable_input(i) = input;
              break;
            }
          }
          nodes_removable.insert(trans_first);
          nodes_removable.insert(trans_second);
        }
      }
    }
    graph_->mutable_node()->erase(
        std::remove_if(
            graph_->mutable_node()->begin(), graph_->mutable_node()->end(),
            [nodes_removable](const NodeDef& node) {
              return nodes_removable.find(node.name()) != nodes_removable.end();
            }),
        graph_->mutable_node()->end());
    return Status::OK();
  }

  const LayoutOptimizer::TuningConfig& config_;
};

int GetNumTranspose(const GraphDef& graph) {
  int number = 0;
  for (const auto& node : graph.node()) {
    if (IsTranspose(node)) {
      number++;
    }
  }
  VLOG(1) << "Number of Transpose nodes: " << number;
  return number;
}

int GetNumGPUs(const Cluster& cluster) {
  auto devices = cluster.GetDevices();
  int num_gpus = 0;
  for (const auto& device : devices) {
    if (device.second.type() == "GPU") {
      if (device.second.environment().find("architecture") !=
          device.second.environment().end()) {
        const string arch = device.second.environment().at("architecture");
        // TODO(yaozhang): Enable for Volta GPUs (compute capability version 7).
        if (arch < "7") {
          num_gpus++;
        }
      }
    }
  }
  return num_gpus;
}
}  // namespace

Status LayoutOptimizer::Tune(const GrapplerItem& item,
                             const GraphProperties& graph_properties,
                             const TuningConfig& config, GraphDef* output) {
  auto status = graph_properties.AnnotateOutputShapes(output);
  if (!status.ok()) {
    *output = item.graph;
    return status;
  }
  NodeMap node_map(output);
  DataLayoutOptimizer layout_optimizer(*virtual_placer_, config,
                                       nodes_to_preserve_, output, &node_map);
  status = layout_optimizer.Optimize();
  return status;
}

Status LayoutOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                 GraphDef* output) {
  if (GetNumGPUs(*cluster) < 1) {
    // LayoutOptimizer is currently only tuned for GPU.
    *output = item.graph;
    return Status::OK();
  }

  virtual_placer_.reset(new VirtualPlacer(cluster));
  nodes_to_preserve_ = item.NodesToPreserve();
  GraphProperties graph_properties(item);
  auto status = graph_properties.InferStatically(false);
  if (!status.ok()) {
    *output = item.graph;
    return status;
  }

  TuningConfig config;
  config.no_gemm = true;
  // TODO(yaozhang): Enable tuning with various TuningConfig choices wtih
  // the measurement-based estimator.
  status = Tune(item, graph_properties, config, output);
  if (!status.ok()) {
    *output = item.graph;
  }
  return status;
}

void LayoutOptimizer::Feedback(Cluster* cluster, const GrapplerItem& item,
                               const GraphDef& optimize_output, double result) {
  // Nothing to do for LayoutOptimizer.
}

}  // end namespace grappler
}  // end namespace tensorflow
