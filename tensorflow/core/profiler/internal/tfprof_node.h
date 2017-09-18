/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_NODE_H_
#define THIRD_PARTY_TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_NODE_H_

#include <map>
#include <set>
#include <string>
#include <vector>

#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor_description.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/profiler/internal/tfprof_options.h"
#include "tensorflow/core/profiler/tfprof_log.pb.h"

namespace tensorflow {
namespace tfprof {
std::vector<int64> ShapeProtoToVec(const TensorShapeProto& shape_pb);

TensorShapeProto VecToShapeProto(const std::vector<int64> shape_vec);

class TFGraphNode;

class ExecStep {
 public:
  ExecStep() {}

  void AddTimeStats(const string& dev, const NodeExecStats& step_stat);

  void AddMemoryStats(const string& dev, const NodeExecStats& step_stat);

  int64 run_count() const { return exec_.run_count(); }
  // The execution time of an op. If it runs on accelerator, then it's
  // accelerator_exec_micros(). Otherwise, it's CPU time.
  int64 exec_micros() const;
  // The accelerator execution time of an op. 0 if not run on accelerator.
  int64 accelerator_exec_micros() const;
  // The cpu execution time of an op.
  int64 cpu_exec_micros() const;

  const std::map<string, std::vector<std::pair<int64, int64>>>& op_execs()
      const {
    return op_execs_;
  }
  int64 all_start_micros() const { return exec_.all_start_micros(); }
  int64 latest_end_micros() const { return exec_.latest_end_micros(); }

  int64 requested_bytes() const { return exec_.requested_bytes(); }
  int64 peak_bytes() const { return exec_.peak_bytes(); }
  int64 residual_bytes() const { return exec_.residual_bytes(); }
  int64 output_bytes() const { return exec_.output_bytes(); }
  int64 accelerator_temp_bytes() const {
    return exec_.accelerator_temp_bytes();
  }
  int64 host_temp_bytes() const { return exec_.host_temp_bytes(); }
  int64 accelerator_persistent_bytes() const {
    return exec_.accelerator_persistent_bytes();
  }
  int64 host_persistent_bytes() const { return exec_.host_persistent_bytes(); }
  const std::map<int32, std::pair<int64, uint64>>& output_memory() const {
    return output_memory_;
  }
  int64 allocator_bytes_in_use() const {
    return exec_.allocator_bytes_in_use();
  }

  const ExecProfile& ToProto() {
    exec_.mutable_accelerator_execs()->clear();
    for (const auto& e : accelerator_execs_) {
      auto& exec_time = (*exec_.mutable_accelerator_execs())[e.first];
      for (const auto& p : e.second) {
        auto* t = exec_time.mutable_times()->Add();
        t->add_int64_values(p.first);
        t->add_int64_values(p.second);
      }
    }

    exec_.mutable_cpu_execs()->clear();
    for (const auto& e : cpu_execs_) {
      auto& exec_time = (*exec_.mutable_cpu_execs())[e.first];
      for (const auto& p : e.second) {
        auto* t = exec_time.mutable_times()->Add();
        t->add_int64_values(p.first);
        t->add_int64_values(p.second);
      }
    }

    exec_.mutable_devices()->Clear();
    exec_.mutable_devices()->Reserve(devices_.size());
    for (const string& d : devices_) {
      exec_.add_devices(d);
    }

    exec_.mutable_output_memory()->clear();
    for (const auto& mem : output_memory_) {
      auto& mem_pb = (*exec_.mutable_output_memory())[mem.first];
      mem_pb.set_bytes(mem.second.first);
      mem_pb.set_ptr(mem.second.second);
    }

    return exec_;
  }

  void FromProto(const ExecProfile& exec) {
    exec_.Clear();
    exec_.MergeFrom(exec);

    devices_.clear();
    devices_.insert(exec.devices().begin(), exec.devices().end());

    accelerator_execs_.clear();
    cpu_execs_.clear();
    op_execs_.clear();

    for (const auto& exec_time : exec_.accelerator_execs()) {
      auto& exec = accelerator_execs_[exec_time.first];
      auto& op_exec = op_execs_[exec_time.first];
      for (const auto& p : exec_time.second.times()) {
        exec.push_back(std::make_pair(p.int64_values(0), p.int64_values(1)));
        op_exec.push_back(std::make_pair(p.int64_values(0), p.int64_values(1)));
      }
    }
    for (const auto& exec_time : exec_.cpu_execs()) {
      auto& exec = cpu_execs_[exec_time.first];
      auto& op_exec = op_execs_[exec_time.first];
      for (const auto& p : exec_time.second.times()) {
        exec.push_back(std::make_pair(p.int64_values(0), p.int64_values(1)));
        op_exec.push_back(std::make_pair(p.int64_values(0), p.int64_values(1)));
      }
    }
    for (const auto& output_mem : exec_.output_memory()) {
      auto& mem = output_memory_[output_mem.first];
      mem.first = output_mem.second.bytes();
      mem.second = output_mem.second.ptr();
    }
  }

 private:
  ExecProfile exec_;
  // device -> vector of {op_start_micros, op_exec_micros} pairs.
  // accelerator_execs: gpu:id/stream:all -> {op_start_micros, op_exec_micros}
  // For accelerator, vector size can be larger than 1, multiple kernel fires
  // or in tf.while_loop.
  std::map<string, std::vector<std::pair<int64, int64>>> accelerator_execs_;
  // cpu_execs: cpu/gpu:id -> {op_start_micros, op_exec_micros}
  // For cpu, vector size can be larger than 1 if in tf.while_loop.
  std::map<string, std::vector<std::pair<int64, int64>>> cpu_execs_;
  // combines accelerator_execs_ and cpu_execs_.
  std::map<string, std::vector<std::pair<int64, int64>>> op_execs_;
  // All devices the op is associated with (e.g. gpu:0 (scheduling),
  // gpu:0:stream:xx (kernel exec), cpu:0 host)
  std::set<string> devices_;
  // output_idx -> {output_bytes, memory_ptr}
  std::map<int32, std::pair<int64, uint64>> output_memory_;
};

#define GRAPH_NODE_BYTES(type)             \
  do {                                     \
    if (execs_.empty()) {                  \
      return 0;                            \
    }                                      \
    if (step >= 0) {                       \
      auto exec = execs_.find(step);       \
      if (exec == execs_.end()) return 0;  \
      return exec->second.type##_bytes();  \
    }                                      \
                                           \
    int64 bytes = 0;                       \
    for (const auto& exec : execs_) {      \
      bytes += exec.second.type##_bytes(); \
    }                                      \
    return bytes / execs_.size();          \
  } while (0)

class TFGraphNode {
 public:
  TFGraphNode(const ProfileNode& node, const ProfileProto& profile) {
    FromProto(node, profile);
  }

  TFGraphNode(const NodeDef* node, int64 id) {
    node_.set_id(id);
    node_.set_name(node->name());
    node_.set_op(node->op());
    node_.set_float_ops(0);

    for (const auto& attr : node->attr()) {
      (*node_.mutable_attrs())[attr.first].MergeFrom(attr.second);
      if (attr.first == "shape" && attr.second.has_shape()) {
        if (!shape_.empty()) {
          fprintf(stderr, "Found duplicated shapes!\n");
          continue;
        }
        shape_ = ShapeProtoToVec(attr.second.shape());
      } else if (attr.first == "_output_shapes" && attr.second.has_list()) {
        if (!output_shapes_.empty()) {
          fprintf(stderr, "Found duplicated output shapes!\n");
          continue;
        }
        for (int i = 0; i < attr.second.list().shape_size(); ++i) {
          output_shapes_[i] = ShapeProtoToVec(attr.second.list().shape(i));
        }
      }
    }
    op_types_.insert(node->op());
  }

  void AddInput(TFGraphNode* input, int32 output_idx, int input_idx) {
    src_output_idx_[input->name()] = output_idx;

    inputs_[input_idx] = input->name();
    const auto& output_shape = input->output_shapes().find(output_idx);
    // Always create an empty vec even if the shape info might be missing.
    std::vector<int64>& shape_vec = input_shapes_[input_idx];
    if (output_shape != input->output_shapes().end()) {
      shape_vec.assign(output_shape->second.begin(),
                       output_shape->second.end());
    }
  }

  void AddOpType(const string& op_type) { op_types_.insert(op_type); }

  void AddStepStat(int64 step, const string& device,
                   const NodeExecStats& step_stat);

  void AddFloatOps(int64 float_ops) { node_.set_float_ops(float_ops); }

  // TODO(xpan): This could take a lot of memory.
  void AddCode(const CodeDef& code) { node_.mutable_trace()->MergeFrom(code); }

  const string& name() const { return node_.name(); }
  int64 id() const { return node_.id(); }
  const string& op() const { return node_.op(); }
  const ProfileNode& node() { return node_; }

  bool trackable(int64 step) const {
    auto exec = execs_.find(step);
    if (exec == execs_.end()) return false;

    if (exec->second.all_start_micros() == 0) return false;
    if (node_.canonical_device().empty() || node_.host_device().empty()) {
      return false;
    }
    return true;
  }

  const ProfileNode& ToProto(
      const std::map<string, std::unique_ptr<TFGraphNode>>& nodes_map) {
    node_.clear_shape();
    node_.mutable_shape()->Reserve(shape().size());
    for (int64 s : shape()) {
      node_.add_shape(s);
    }

    node_.clear_op_types();
    node_.mutable_op_types()->Reserve(op_types().size());
    for (const string& t : op_types()) {
      node_.add_op_types(t);
    }

    node_.clear_execs();
    for (auto& exec : execs_) {
      auto& exec_pb = (*node_.mutable_execs())[exec.first];
      exec_pb.MergeFrom(exec.second.ToProto());
    }

    node_.clear_inputs();
    for (const auto& inp : inputs_) {
      (*node_.mutable_inputs())[inp.first] = nodes_map.at(inp.second)->id();
    }

    node_.clear_input_shapes();
    for (const auto& s : input_shapes_) {
      auto& shape = (*node_.mutable_input_shapes())[s.first];
      for (int64 d : s.second) {
        shape.add_int64_values(d);
      }
    }

    node_.clear_output_shapes();
    for (const auto& s : output_shapes_) {
      auto& shape = (*node_.mutable_output_shapes())[s.first];
      for (int64 d : s.second) {
        shape.add_int64_values(d);
      }
    }

    node_.clear_src_output_index();
    for (const auto& s : src_output_idx_) {
      int64 id = nodes_map.at(s.first)->id();
      (*node_.mutable_src_output_index())[id] = s.second;
    }
    return node_;
  }

  void FromProto(const ProfileNode& node, const ProfileProto& profile) {
    node_.Clear();
    node_.MergeFrom(node);

    op_types_.clear();
    op_types_.insert(node_.op_types().begin(), node_.op_types().end());

    shape_.clear();
    for (int64 s : node_.shape()) {
      shape_.push_back(s);
    }

    execs_.clear();
    for (const auto& exec_pb : node.execs()) {
      auto& exec = execs_[exec_pb.first];
      exec.FromProto(exec_pb.second);
    }

    inputs_.clear();
    for (const auto& inp : node.inputs()) {
      inputs_[inp.first] = profile.nodes().at(inp.second).name();
    }

    input_shapes_.clear();
    for (const auto& s : node.input_shapes()) {
      auto& shape = input_shapes_[s.first];
      for (const int64 d : s.second.int64_values()) {
        shape.push_back(d);
      }
    }

    output_shapes_.clear();
    for (const auto& s : node.output_shapes()) {
      auto& shape = output_shapes_[s.first];
      for (const int64 d : s.second.int64_values()) {
        shape.push_back(d);
      }
    }

    src_output_idx_.clear();
    for (const auto& s : node.src_output_index()) {
      src_output_idx_[profile.nodes().at(s.first).name()] = s.second;
    }
  }

  const std::map<int32, string>& inputs() const { return inputs_; }
  const std::map<string, int32>& src_output_idx() const {
    return src_output_idx_;
  }

  // Number of times the graph node is executed. When step < 0, the
  // average number of times executed across all steps.
  int64 run_count(int64 step) const {
    if (execs_.empty()) {
      return 0;
    }
    if (step >= 0) {
      auto exec = execs_.find(step);
      if (exec == execs_.end()) {
        return 0;
      }
      return exec->second.run_count();
    }
    int64 total_run_count = 0;
    for (const auto& exec : execs_) {
      total_run_count += exec.second.run_count();
    }
    return total_run_count / execs_.size();
  }
  // This is overall computation time, including both cpu and accelerator.
  // Note, cpu and accelerator might or might not run in parallel.
  int64 exec_micros(int64 step) const {
    // Empty when no RunMetadata is provided.
    if (execs_.empty()) {
      return 0;
    }
    if (step >= 0) {
      auto exec = execs_.find(step);
      if (exec == execs_.end()) {
        return 0;
      }
      return exec->second.exec_micros();
    }

    int64 total_micros = 0;
    for (const auto& exec : execs_) {
      total_micros += exec.second.exec_micros();
    }
    return total_micros / execs_.size();
  }

  // This is accelerator computation time of a step, or average of
  // multiple step, when step < 0.
  int64 accelerator_exec_micros(int64 step) const {
    // Empty when no RunMetadata is provided.
    if (execs_.empty()) {
      return 0;
    }
    if (step >= 0) {
      auto exec = execs_.find(step);
      if (exec == execs_.end()) {
        return 0;
      }
      return exec->second.accelerator_exec_micros();
    }

    int64 total_micros = 0;
    for (const auto& exec : execs_) {
      total_micros += exec.second.accelerator_exec_micros();
    }
    return total_micros / execs_.size();
  }

  // This is cpu computation time of a step, or average of
  // multiple step, when step < 0.
  int64 cpu_exec_micros(int64 step) const {
    // Empty when no RunMetadata is provided.
    if (execs_.empty()) {
      return 0;
    }
    if (step >= 0) {
      auto exec = execs_.find(step);
      if (exec == execs_.end()) {
        return 0;
      }
      return exec->second.cpu_exec_micros();
    }

    int64 total_micros = 0;
    for (const auto& exec : execs_) {
      total_micros += exec.second.cpu_exec_micros();
    }
    return total_micros / execs_.size();
  }

  int64 requested_bytes(int64 step) const { GRAPH_NODE_BYTES(requested); }
  int64 peak_bytes(int64 step) const { GRAPH_NODE_BYTES(peak); }
  int64 residual_bytes(int64 step) const { GRAPH_NODE_BYTES(residual); }
  int64 output_bytes(int64 step) const { GRAPH_NODE_BYTES(output); }

  int64 all_start_micros(int64 step) const {
    auto exec = execs_.find(step);
    if (exec == execs_.end()) {
      return 0;
    }
    return exec->second.all_start_micros();
  }

  int64 latest_end_micros(int64 step) const {
    auto exec = execs_.find(step);
    if (exec == execs_.end()) {
      return 0;
    }
    return exec->second.latest_end_micros();
  }

  const std::map<string, std::vector<std::pair<int64, int64>>>& op_execs(
      int64 step) const {
    auto exec = execs_.find(step);
    if (exec == execs_.end()) {
      return empty_op_execs_;
    }
    return exec->second.op_execs();
  }

  const std::map<int64, ExecStep>& all_op_execs() const { return execs_; }

  int64 accelerator_temp_bytes(int64 step) const {
    auto exec = execs_.find(step);
    if (exec == execs_.end()) {
      return 0;
    }
    return exec->second.accelerator_temp_bytes();
  }
  int64 host_temp_bytes(int64 step) const {
    auto exec = execs_.find(step);
    if (exec == execs_.end()) {
      return 0;
    }
    return exec->second.host_temp_bytes();
  }
  int64 accelerator_persistent_bytes(int64 step) const {
    auto exec = execs_.find(step);
    if (exec == execs_.end()) {
      return 0;
    }
    return exec->second.accelerator_persistent_bytes();
  }
  int64 host_persistent_bytes(int64 step) const {
    auto exec = execs_.find(step);
    if (exec == execs_.end()) {
      return 0;
    }
    return exec->second.host_persistent_bytes();
  }
  const std::map<int32, std::pair<int64, uint64>>& output_memory(
      int64 step) const {
    auto exec = execs_.find(step);
    if (exec == execs_.end()) {
      return empty_output_memory_;
    }
    return exec->second.output_memory();
  }
  int64 allocator_bytes_in_use(int64 step) const {
    auto exec = execs_.find(step);
    if (exec == execs_.end()) {
      return 0;
    }
    return exec->second.allocator_bytes_in_use();
  }

  int64 parameters() const {
    if (!shape().empty()) {
      int64 params = 1;
      bool complete_shape = true;
      for (int64 d : shape()) {
        // Sometimes parameters could be <0 when a dim is unknown.
        if (d < 0) {
          complete_shape = false;
          break;
        }
        params *= d;
      }
      if (complete_shape) {
        return params;
      } else {
        fprintf(stderr, "Incomplete shape.\n");
      }
    }
    return 0;
  }

  int64 float_ops(int64 step) const {
    // If not run, return static analysis.
    if (execs_.empty()) {
      return node_.float_ops();
    }
    // Otherwise, return dynamic float_ops.
    return node_.float_ops() * run_count(step);
  }
  const CodeDef& code() { return node_.trace(); }
  string canonical_device() const { return node_.canonical_device(); }
  string host_device() const { return node_.host_device(); }
  const std::set<string>& op_types() const { return op_types_; }

  const AttrValue* op_attrs(const string& name) const {
    const auto it = node_.attrs().find(name);
    if (it == node_.attrs().end()) {
      return nullptr;
    }
    return &it->second;
  }

  const std::vector<int64>& shape() const { return shape_; }

  const std::map<int, std::vector<int64>>& output_shapes() const {
    return output_shapes_;
  }
  const std::map<int, std::vector<int64>>& input_shapes() const {
    return input_shapes_;
  }

 private:
  std::map<int, string> inputs_;
  std::map<string, int32> src_output_idx_;

  ProfileNode node_;

  std::vector<int64> shape_;
  // Won't missing input_idx. But some shapes might be empty (unknown).
  std::map<int, std::vector<int64>> input_shapes_;
  // Could miss output_idx if no _output_shapes attr. some shapes can also
  // be empty.
  std::map<int, std::vector<int64>> output_shapes_;

  std::set<string> op_types_;

  std::map<int64, ExecStep> execs_;

  std::map<int32, std::pair<int64, uint64>> empty_output_memory_;
  std::map<string, std::vector<std::pair<int64, int64>>> empty_op_execs_;
};

class TFMultiGraphNode {
 public:
  TFMultiGraphNode(const string& name)
      : name_(name),
        step_(-1),
        run_count_(0),
        exec_micros_(0),
        accelerator_exec_micros_(0),
        cpu_exec_micros_(0),
        requested_bytes_(0),
        peak_bytes_(0),
        residual_bytes_(0),
        output_bytes_(0),
        float_ops_(0),
        parameters_(0) {}

  bool SnapshotNodes(int64 step, const std::vector<string>& type_regexes) {
    run_count_ = 0;
    exec_micros_ = 0;
    accelerator_exec_micros_ = 0;
    cpu_exec_micros_ = 0;

    requested_bytes_ = 0;
    peak_bytes_ = 0;
    residual_bytes_ = 0;
    output_bytes_ = 0;

    float_ops_ = 0;
    parameters_ = 0;
    op_types_.clear();
    shapes_.clear();
    devices_.clear();
    snapshot_nodes_.clear();

    step_ = step;
    std::vector<const TFGraphNode*> nodes = pick_nodes(type_regexes);

    if (nodes.empty()) {
      return (type_regexes.size() == 1 && type_regexes[0] == ".*");
    }

    for (const TFGraphNode* node : nodes) {
      op_types_.insert(node->op_types().begin(), node->op_types().end());

      run_count_ += node->run_count(step);
      exec_micros_ += node->exec_micros(step);
      accelerator_exec_micros_ += node->accelerator_exec_micros(step);
      cpu_exec_micros_ += node->cpu_exec_micros(step);

      requested_bytes_ += node->requested_bytes(step);
      peak_bytes_ += node->peak_bytes(step);
      residual_bytes_ += node->residual_bytes(step);
      output_bytes_ += node->output_bytes(step);

      float_ops_ += node->float_ops(step);
      parameters_ += node->parameters();
      if (node->shape().size() > 0) {
        shapes_.push_back(node->shape());
      }
      devices_.insert(node->canonical_device());
      snapshot_nodes_[node->name()] = node;
    }
    return true;
  }

  int64 step() const { return step_; }

  void AddGraphNode(const TFGraphNode* node) {
    if (nodes_.find(node->name()) != nodes_.end()) {
      return;
    }
    nodes_[node->name()] = node;
  }

  const std::map<string, const TFGraphNode*>& graph_nodes() const {
    return snapshot_nodes_;
  }

  const string& name() const { return name_; }

  int64 run_count() const { return run_count_; }
  int64 exec_micros() const { return exec_micros_; }
  int64 accelerator_exec_micros() const { return accelerator_exec_micros_; }
  int64 cpu_exec_micros() const { return cpu_exec_micros_; }

  int64 requested_bytes() const { return requested_bytes_; }
  int64 peak_bytes() const { return peak_bytes_; }
  int64 residual_bytes() const { return residual_bytes_; }
  int64 output_bytes() const { return output_bytes_; }

  int64 float_ops() const { return float_ops_; }

  int64 parameters() const { return parameters_; }

  const std::set<string>& devices() const { return devices_; }

  const std::set<string>& op_types() const { return op_types_; }

  const std::vector<std::vector<int64>>& shapes() const { return shapes_; }

 private:
  std::vector<const TFGraphNode*> pick_nodes(
      const std::vector<string>& type_regexes) {
    if (type_regexes.empty()) {
      return {};
    }
    std::vector<const TFGraphNode*> ret;
    if (type_regexes.size() == 1 && type_regexes[0] == ".*") {
      for (const auto& n : nodes_) {
        ret.push_back(n.second);
      }
      return ret;
    }

    for (const string& regex : type_regexes) {
      for (const auto& n : nodes_) {
        for (const string& type : n.second->op_types()) {
          if (RE2::FullMatch(type, regex)) {
            ret.push_back(n.second);
            break;
          }
        }
      }
    }
    return ret;
  }

  const string name_;
  int64 step_;
  // Snapshot based on type_regexes
  std::set<string> op_types_;
  int64 run_count_;
  int64 exec_micros_;
  int64 accelerator_exec_micros_;
  int64 cpu_exec_micros_;

  int64 requested_bytes_;
  int64 peak_bytes_;
  int64 residual_bytes_;
  int64 output_bytes_;
  int64 float_ops_;
  int64 parameters_;
  std::set<string> devices_;
  std::vector<std::vector<int64>> shapes_;
  std::map<string, const TFGraphNode*> snapshot_nodes_;

  // Overall data held by the TFMultiGraphNode.
  std::map<string, const TFGraphNode*> nodes_;
};

bool IsPlacedOnAccelerator(const string& device);
}  // namespace tfprof
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_NODE_H_
