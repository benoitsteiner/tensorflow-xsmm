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

#include "tensorflow/core/grappler/optimizers/loop_optimizer.h"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <deque>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/frame.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"

using tensorflow::strings::StrCat;

namespace tensorflow {
namespace grappler {
namespace {

std::vector<int> GetStackPushNodesToConvert(const SimpleGraphView& graph_view,
                                            int stack_node_idx) {
  VLOG(1) << "Stack node: " << graph_view.graph()->node(stack_node_idx).name();
  const std::unordered_set<string> op_types_to_traverse(
      {"Stack", "StackV2", "Enter", "RefEnter", "Switch", "RefSwitch",
       "Identity", "RefIdentity"});
  std::vector<int> nodes_to_convert;
  std::set<int> fanout;
  graph_view.DepthFirstSearch(op_types_to_traverse, stack_node_idx, &fanout);
  for (int fanout_idx : fanout) {
    const NodeDef& fanout_node = graph_view.graph()->node(fanout_idx);
    VLOG(1) << "Fanout " << fanout_idx << " : " << fanout_node.name();
    if (IsStackPushOp(fanout_node)) {
      nodes_to_convert.push_back(fanout_idx);
    } else if (IsStackOp(fanout_node) || IsStackCloseOp(fanout_node) ||
               op_types_to_traverse.find(fanout_node.op()) !=
                   op_types_to_traverse.end()) {
      continue;
    } else if (!IsStackPopOp(fanout_node) ||
               !graph_view.outputs(fanout_idx).empty()) {
      // The node is either a stack pop with consumers or something unexpected
      // so we leave the graph alone.
      nodes_to_convert.clear();
      break;
    }
  }
  return nodes_to_convert;
}

Status RemoveStackOps(const GraphDef& graph, GraphDef* optimized_graph) {
  *optimized_graph = graph;
  NodeMap node_map(optimized_graph);
  SimpleGraphView graph_view;
  TF_RETURN_IF_ERROR(graph_view.Initialize(graph));
  for (int node_idx = 0; node_idx < graph.node_size(); ++node_idx) {
    if (IsStackOp(graph.node(node_idx))) {
      for (int push_node_idx :
           GetStackPushNodesToConvert(graph_view, node_idx)) {
        // We found push nodes without corresponding pops. Convert them to
        // Identity passing the data through and add a control dependency from
        // the op supplying the stack handle.
        NodeDef* push_node = optimized_graph->mutable_node(push_node_idx);
        VLOG(1) << "Converting " << push_node_idx << " : "
                << push_node->DebugString();
        if (push_node->attr().count("swap_memory") != 0) {
          push_node->mutable_attr()->erase("swap_memory");
        }
        push_node->set_op("Identity");
        push_node->mutable_input()->SwapElements(0, 1);
        const string ctrl_dep = ConstantFolding::AddControlDependency(
            push_node->input(1), optimized_graph, &node_map);
        push_node->set_input(1, ctrl_dep);
        VLOG(1) << "After converting: " << push_node->DebugString();
      }
    }
  }
  return Status::OK();
}

}  // namespace

Status LoopOptimizer::LINMHandleInvariantEnter(NodeDef* node,
                                               const int num_outputs) {
  auto consumers = node_map_->GetOutputs(node->name());
  std::vector<string> enter_control_inputs;
  string enter_input;
  for (auto& input : node->input()) {
    if (IsControlInput(input)) {
      enter_control_inputs.push_back(input);
    } else {
      enter_input = input;
    }
  }
  for (auto* consumer : consumers) {
    if (invariant_nodes_.count(consumer)) {
      for (int i = 0; i < consumer->input_size(); ++i) {
        if (NodeName(consumer->input(i)) == node->name()) {
          consumer->set_input(i, enter_input);
          node_map_->AddOutput(NodeName(enter_input), consumer->name());
          node_map_->RemoveOutput(node->name(), consumer->name());
        }
      }
      for (auto& control_input : enter_control_inputs) {
        consumer->add_input(control_input);
        node_map_->AddOutput(NodeName(control_input), consumer->name());
      }
    }
  }
  return Status::OK();
}

Status LoopOptimizer::LINMHandleConst(NodeDef* node,
    const int num_outputs, const int frame_id) {
  NodeDef* const_node;
  if (num_outputs == 0) {
    // all successor nodes are invariant
    // Remove the control inputs from this frame to the const node,
    // when moving it out of the frame (in parent frame)
    const_node = node;
    node_map_->RemoveInputs(node->name());
    node->clear_input();
  } else {
    // some successor nodes are variant
    // Have to keep the const node in the frame,
    // so create a new one outside the frame (in parent frame)
    const_node = optimized_graph_->add_node();
    const_node->set_name(AddPrefixToNodeName(node->name(), kLoopOptimizer));
    const_node->set_op("Const");
    const_node->set_device(node->device());
    *const_node->mutable_attr() = node->attr();
    node_map_->AddNode(const_node->name(), const_node);
    auto consumers = node_map_->GetOutputs(node->name());
    for (auto* consumer : consumers) {
      if (invariant_nodes_.count(consumer)) {
        for (int i = 0; i < consumer->input_size(); ++i) {
          if (NodeName(consumer->input(i)) == node->name()) {
            if (IsControlInput(consumer->input(i))) {
              *consumer->mutable_input(i) = AsControlDependency(*const_node);
            } else {
              *consumer->mutable_input(i) = const_node->name();
            }
            node_map_->AddOutput(const_node->name(), consumer->name());
            node_map_->RemoveOutput(node->name(), consumer->name());
          }
        }
      }
    }
  }
  // add a control input from the parent frame
  auto parent_it = frame_parent_.find(frame_id);
  if (parent_it != frame_parent_.end()) {
    int parent_id = parent_it->second;
    auto loop_cond_it = loop_cond_.find(parent_id);
    if (loop_cond_it == loop_cond_.end()) {
      return errors::InvalidArgument(
          "Frame ", frame_id, " doesn't have a LoopCond node");
    }
    auto& loop_cond_name = loop_cond_it->second->name();
    NodeDef* switch_node = nullptr;
    for (auto* node : node_map_->GetOutputs(loop_cond_name)) {
      if (node->op() == "Switch") {
        switch_node = node;
        break;
      }
    }
    if (!switch_node) {
      return errors::InvalidArgument(
          "LoopCond node of Frame ", frame_id,
          " doesn't connect to any Switch node");
    }
    string switch_output = StrCat(switch_node->name(), ":1");
    const string ctrl_dep = ConstantFolding::AddControlDependency(
        switch_output, optimized_graph_, node_map_.get());
    const_node->add_input(ctrl_dep);
    node_map_->AddOutput(NodeName(ctrl_dep), const_node->name());
  }
  return Status::OK();
}

Status LoopOptimizer::LINMHandleInvariantNode(NodeDef* node,
    const int num_outputs, const int frame_id) {
  // have to remove control inputs to the invariant node from the same frame
  // when moving this node out of this frame
  for (int i = 0; i < node->input_size(); ++i) {
    if (IsControlInput(node->input(i))) {
      node->mutable_input()->SwapElements(i, node->input_size() - 1);
      node->mutable_input()->RemoveLast();
    }
  }
  if (num_outputs == 0) {
    return Status::OK();
  }

  DataTypeVector input_types;
  DataTypeVector output_types;
  OpRegistryInterface* op_registry = OpRegistry::Global();
  const OpRegistrationData* op_reg_data = nullptr;
  TF_RETURN_IF_ERROR(
      op_registry->LookUp(node->op(), &op_reg_data));
  TF_RETURN_IF_ERROR(
      InOutTypesForNode(*node, op_reg_data->op_def,
                        &input_types, &output_types));

  auto consumers = node_map_->GetOutputs(node->name());
  string fname = invariant_enters_[frame_id][0]->attr().at("frame_name").s();
  int piterations = invariant_enters_[frame_id][0]
                    ->attr().at("parallel_iterations").i();
  for (auto* consumer : consumers) {
    if (!invariant_nodes_.count(consumer)) {
      for (int i = 0; i < consumer->input_size(); ++i) {
        int port;
        string node_name = ParseNodeName(consumer->input(i), &port);
        if (node_name != node->name()) {
          continue;
        }
        if (port < 0) {
          return errors::InvalidArgument(
              "Invariant node should not have control outputs "
              "to variant node");
        }
        DataType output_type = output_types[port];
        NodeDef* new_enter = optimized_graph_->add_node();
        new_enter->set_op("Enter");
        new_enter->set_device(node->device());
        new_enter->set_name(AddPrefixToNodeName(
            StrCat(fname, "_enter_", new_enter_id_++), kLoopOptimizer));
        AttrValue data_type;
        data_type.set_type(output_type);
        new_enter->mutable_attr()->insert({"T", data_type});
        AttrValue frame_name;
        frame_name.set_s(fname);
        new_enter->mutable_attr()->insert({"frame_name", frame_name});
        AttrValue is_const;
        is_const.set_b(true);
        new_enter->mutable_attr()->insert({"is_constant", is_const});
        AttrValue parallel_iterations;
        parallel_iterations.set_i(piterations);
        new_enter->mutable_attr()->insert(
            {"parallel_iterations", parallel_iterations});
        new_enter->add_input(consumer->input(i));
        *consumer->mutable_input(i) = new_enter->name();
        node_map_->AddNode(new_enter->name(), new_enter);
        node_map_->AddOutput(node->name(), new_enter->name());
        node_map_->AddOutput(new_enter->name(), consumer->name());
      }
    }
  }
  return Status::OK();
}

Status LoopOptimizer::MoveInvariantNodes(const int frame_id) {
  for (auto iter = invariant_nodes_.begin();
       iter != invariant_nodes_.end(); ++iter) {
    auto* invariant_node = iter->first;
    const int num_outputs = iter->second;
    if (IsEnter(*invariant_node)) {
      TF_RETURN_IF_ERROR(
          LINMHandleInvariantEnter(invariant_node, num_outputs));
    } else if (IsConstant(*invariant_node)) {
      TF_RETURN_IF_ERROR(
          LINMHandleConst(invariant_node, num_outputs, frame_id));
    } else {
      TF_RETURN_IF_ERROR(
          LINMHandleInvariantNode(invariant_node, num_outputs, frame_id));
    }
  }
  return Status::OK();
}

Status LoopOptimizer::RevertInvariantNodes() {
  std::deque<const NodeDef*> reverted_nodes;
  for (auto iter=invariant_nodes_.begin(); iter != invariant_nodes_.end();) {
    bool erased = false;
    const auto* node = iter->first;
    if (!IsConstant(*node) && !IsEnter(*node) && iter->second > 0) {
      auto& consumers = node_map_->GetOutputs(node->name());
      for (auto* consumer : consumers) {
        if (!invariant_nodes_.count(consumer)) {
          for (const auto& input : consumer->input()) {
            if (IsControlInput(input) && NodeName(input) == node->name()) {
              reverted_nodes.push_back(node);
              invariant_nodes_.erase(iter++);
              erased = true;
              break;
            }
          }
          if (erased) break;
        }
      }
    }
    if (!erased) ++iter;
  }
  while (!reverted_nodes.empty()) {
    const auto* node = reverted_nodes.front();
    reverted_nodes.pop_front();
    std::set<NodeDef*> producers;
    for (const auto& input : node->input()) {
      auto* producer = node_map_->GetNode(input);
      auto iter = invariant_nodes_.find(producer);
      if (iter != invariant_nodes_.end()) {
        if (IsControlInput(input) &&
            !IsConstant(*producer) && !IsEnter(*producer)) {
          reverted_nodes.push_back(producer);
          invariant_nodes_.erase(iter);
        } else {
          producers.insert(producer);
        }
      }
    }
    for (auto* producer : producers) {
      auto iter = invariant_nodes_.find(producer);
      if (iter != invariant_nodes_.end()) {
        ++iter->second;
      }
    }
    for (auto* consumer : node_map_->GetOutputs(node->name())) {
      auto iter = invariant_nodes_.find(consumer);
      if (iter != invariant_nodes_.end()) {
        reverted_nodes.push_back(consumer);
        invariant_nodes_.erase(iter);
      }
    }
  }
  return Status::OK();
}

Status LoopOptimizer::FindInvariantNodes(NodeDef* node) {
  auto consumers = node_map_->GetOutputs(node->name());
  invariant_nodes_.insert(std::make_pair(node, consumers.size()));
  for (auto* consumer : consumers) {
    if (invariant_nodes_.count(consumer) ||
        ModifiesFrameInfo(*consumer)) {
      continue;
    }
    bool is_invariant = true;
    for (const auto& input : consumer->input()) {
      if (!IsControlInput(input)) {
        const auto& name = NodeName(input);
        auto* producer = node_map_->GetNode(name);
        if (!invariant_nodes_.count(producer)) {
          if (IsConstant(*producer)) {
            invariant_nodes_.insert(
                std::make_pair(producer, node_map_->GetOutputs(name).size()));
          } else {
            is_invariant = false;
            break;
          }
        }
      }
    }
    if (is_invariant) {
      std::set<NodeDef*> producers;
      for (const auto& input : consumer->input()) {
        auto* producer = node_map_->GetNode(input);
        producers.insert(producer);
      }
      for (auto* producer : producers) {
        auto iter = invariant_nodes_.find(producer);
        if (iter != invariant_nodes_.end()) {
          --iter->second;
        }
      }
      TF_RETURN_IF_ERROR(FindInvariantNodes(consumer));
    }
  }
  return Status::OK();
}

Status LoopOptimizer::LoopInvariantNodeMotion() {
  std::deque<int> worklist;
  for (auto iter = frame_map_.begin(); iter != frame_map_.end(); ++iter) {
    auto* node = iter->first;
    auto& frame_ids = iter->second;
    if (frame_ids.size() >= 3) {
      for (unsigned int i = 1; i < frame_ids.size() - 1; ++i) {
        frame_parent_[frame_ids[i]] = frame_ids[i - 1];
        frame_children_[frame_ids[i]].insert(frame_ids[i + 1]);
      }
    }
    if (frame_ids.size() >= 2) {
      frame_children_[frame_ids[0]].insert(frame_ids[1]);
      frame_parent_[frame_ids.back()] = frame_ids[frame_ids.size() - 2];
    }
    if (frame_ids.size() >= 1) {
      frame_children_.insert(std::make_pair(frame_ids.back(), empty_set_));
      if (node->op() == "LoopCond") {
        if (loop_cond_.count(frame_ids.back())) {
          return errors::InvalidArgument(
              "Loop ", frame_ids.back(),
              " has more than one LoopCond node: ", node->name(), " and ",
              loop_cond_[frame_ids.back()]->name());
        }
        loop_cond_[frame_ids.back()] = node;
      }
      if (IsEnter(*node) && node->attr().at("is_constant").b()) {
        invariant_enters_[frame_ids.back()].push_back(
            const_cast<NodeDef*>(node));
      }
    }
  }

  for (auto it = frame_children_.begin(); it != frame_children_.end(); ++it) {
    if (it->second.size() == 0) {
      worklist.push_back(it->first);
    }
  }

  while (!worklist.empty()) {
    int frame_id = worklist.front();
    new_enter_id_ = 0;
    worklist.pop_front();
    auto parent_it = frame_parent_.find(frame_id);
    if (parent_it != frame_parent_.end()) {
      int parent_id = parent_it->second;
      frame_children_[parent_id].erase(frame_id);
      if (frame_children_[parent_id].size() == 0) {
        worklist.push_back(parent_id);
      }
    }

    if (invariant_enters_[frame_id].empty()) {
      continue;
    }
    invariant_nodes_.clear();
    for (auto* enter : invariant_enters_[frame_id]) {
      TF_RETURN_IF_ERROR(FindInvariantNodes(enter));
    }

    // revert invariant nodes that have control outputs to variant nodes
    TF_RETURN_IF_ERROR(RevertInvariantNodes());

    TF_RETURN_IF_ERROR(MoveInvariantNodes(frame_id));
  }
  return Status::OK();
}

Status LoopOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                               GraphDef* optimized_graph) {
  TF_RETURN_IF_ERROR(RemoveStackOps(item.graph, optimized_graph));
  optimized_graph_ = optimized_graph;

  // Set up helper data structures.
  node_map_.reset(new NodeMap(optimized_graph_));
  int num_frames;
  TF_RETURN_IF_ERROR(IdentifyFramesWithNodeMap(*optimized_graph_, *node_map_,
                                               &frame_map_, &num_frames));

  TF_RETURN_IF_ERROR(LoopInvariantNodeMotion());
  return Status::OK();
}

void LoopOptimizer::Feedback(Cluster* /*cluster*/, const GrapplerItem& /*item*/,
                             const GraphDef& /*optimized_graph*/,
                             double /*result*/) {
  // Nothing to do for LoopOptimizer.
}

}  // end namespace grappler
}  // end namespace tensorflow
