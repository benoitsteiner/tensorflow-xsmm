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
#include "tensorflow/core/grappler/utils/functions.h"

#include <unordered_map>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/scanner.h"

namespace tensorflow {
namespace grappler {

namespace {

Status OutputNameRange(const FunctionLibraryDefinition& flib,
                       const NodeDef& node,
                       tensorflow::NameRangeMap* outputs_range_map) {
  const OpRegistrationData* registration;
  TF_RETURN_IF_ERROR(flib.LookUp(node.op(), &registration));
  TF_RETURN_IF_ERROR(tensorflow::NameRangesForNode(node, registration->op_def,
                                                   nullptr, outputs_range_map));
  return Status::OK();
}

Status RegisterFunctionBodyOutputs(const FunctionLibraryDefinition& flib,
                                   const NodeDef& node,
                                   GrapplerFunctionConnectivity* connectivity) {
  tensorflow::NameRangeMap outputs_range_map;
  TF_RETURN_IF_ERROR(OutputNameRange(flib, node, &outputs_range_map));
  connectivity->RegisterFunctionBodyOutputs(node.name(), outputs_range_map);
  return Status::OK();
}

// Replace the placeholder attribute values with the values specified in
// instantiation attributes.
Status ResolveFunctionBodyNodeAttrPlaceholders(
    const AttrValueMap& func_instantiation_attr, NodeDef* node) {
  for (auto& attr : *node->mutable_attr()) {
    const string& placeholder = attr.second.placeholder();
    if (placeholder.empty()) continue;

    auto it = func_instantiation_attr.find(placeholder);
    if (it != func_instantiation_attr.end()) {
      attr.second = it->second;
    } else {
      return errors::InvalidArgument("Can't resolve placeholder: ",
                                     placeholder);
    }
  }
  return Status::OK();
}

}  // namespace

void GrapplerFunctionConnectivity::RegisterInputArgExpansion(
    const InputArgExpansion& input_arg_expansion) {
  const auto& input_name = input_arg_expansion.input_name;
  const auto& placeholders = input_arg_expansion.placeholders;
  input_arg_expansions_.emplace(input_name, input_arg_expansion);
  for (int i = 0; i < placeholders.size(); ++i) {
    const string& placeholder = input_arg_expansion.placeholders[i];
    input_arg_placeholders_.emplace(
        placeholder, InputArgPlaceholder{input_name, /*position=*/i});
  }
}

void GrapplerFunctionConnectivity::RegisterFunctionBodyOutputs(
    const string& node_name, const tensorflow::NameRangeMap& outputs) {
  function_body_outputs_[node_name] = outputs;
}

Status GrapplerFunctionConnectivity::ExpandFunctionDefInput(
    const string& func_def_input, std::vector<string>* graph_def_inputs) const {
  using ::tensorflow::strings::Scanner;

  if (IsControlInput(func_def_input)) {
    graph_def_inputs->push_back(func_def_input);
    return Status::OK();
  }

  // Parse input format: "node_name[:node_output][:position]"
  string node_name;
  string node_output;
  int position = -1;

  StringPiece capture;
  StringPiece remaining;

  // Parse "node_name"
  if (Scanner(func_def_input)
          .One(strings::Scanner::LETTER_DIGIT_DOT_UNDERSCORE)
          .Any(strings::Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE)
          .GetResult(&remaining, &capture)) {
    node_name = string(capture.data(), capture.size());
  }

  // Parse "node_output" if it exists
  if (Scanner(remaining)
          .OneLiteral(":")
          .RestartCapture()
          .One(strings::Scanner::LOWERLETTER)
          .Any(strings::Scanner::LETTER_DIGIT_UNDERSCORE)
          .GetResult(&remaining, &capture)) {
    node_output = string(capture.data(), capture.size());
  }

  // Parse "position" if it exists
  if (Scanner(remaining)
          .OneLiteral(":")
          .RestartCapture()
          .Many(strings::Scanner::DIGIT)
          .GetResult(nullptr, &capture)) {
    CHECK(strings::safe_strto32(capture, &position));
  }

  // If "node_output" is not empty, it must be an output of a function body node
  bool is_function_body_output = !node_output.empty();

  // Function input argument: "node_name[:position]"
  if (!is_function_body_output) {
    auto input_arg = input_arg_expansions_.find(node_name);
    if (input_arg != input_arg_expansions_.end()) {
      const InputArgExpansion& input_arg_expansion = input_arg->second;
      const auto& placeholders = input_arg_expansion.placeholders;

      if (position == -1) {
        // If position is not defined use all placeholders
        graph_def_inputs->reserve(placeholders.size());
        for (const string& placeholder : placeholders) {
          graph_def_inputs->push_back(placeholder);
        }
      } else {
        if (position > input_arg_expansion.placeholders.size() - 1) {
          return errors::InvalidArgument("Invalid input ", node_name,
                                         "position: ", position,
                                         " (out of range)");
        }
        graph_def_inputs->push_back(input_arg_expansion.placeholders[position]);
      }

      return Status::OK();
    }
  }

  // Function body output: "node_name:node_output[:position]"
  if (is_function_body_output) {
    auto function_body_outputs = function_body_outputs_.find(node_name);
    if (function_body_outputs != function_body_outputs_.end()) {
      const tensorflow::NameRangeMap& outputs = function_body_outputs->second;
      auto output = outputs.find(node_output);
      if (output != outputs.end()) {
        const auto& output_range = output->second;

        if (position == -1) {
          // If position is not defined expand node output range
          for (int i = output_range.first; i < output_range.second; ++i) {
            i == 0 ? graph_def_inputs->push_back(node_name)
                   : graph_def_inputs->push_back(
                         strings::StrCat(node_name, ":", i));
          }
        } else {
          if (position > (output_range.second - output_range.first)) {
            return errors::InvalidArgument(
                "Invalid node ", node_name, " output ", node_output,
                " position: ", position, " (out of range)");
          }
          int pos = output_range.first + position;
          pos == 0 ? graph_def_inputs->push_back(node_name)
                   : graph_def_inputs->push_back(
                         strings::StrCat(node_name, ":", pos));
        }

        return Status::OK();
      }
    }
  }

  return errors::InvalidArgument("Failed to expand a function def input: ",
                                 func_def_input);
}

Status GrapplerFunctionConnectivity::ExpandNodeInputs(
    NodeDef* function_body_node) const {
  std::vector<string> expanded_inputs;

  for (const string& function_def_input : function_body_node->input()) {
    TF_RETURN_IF_ERROR(
        ExpandFunctionDefInput(function_def_input, &expanded_inputs));
  }

  function_body_node->clear_input();
  for (const string& expanded_input : expanded_inputs)
    function_body_node->add_input(expanded_input);
  return Status::OK();
}

Status GrapplerFunctionConnectivity::AsFunctionDefInput(
    const string& graph_def_input, string* func_def_input) const {
  using gtl::FindOrNull;

  if (IsControlInput(graph_def_input)) {
    *func_def_input = graph_def_input;
    return Status::OK();
  }

  int position;
  string node_name = ParseNodeName(graph_def_input, &position);
  CHECK_GE(position, 0);

  // Check if it's an input arg placeholder
  if (position == 0) {
    const InputArgPlaceholder* placeholder =
        FindOrNull(input_arg_placeholders_, node_name);
    if (placeholder != nullptr) {
      *func_def_input =
          strings::StrCat(placeholder->input_name, ":", placeholder->position);
      return Status::OK();
    }
  }

  // It must be output from one of the function body nodes
  const tensorflow::NameRangeMap* outputs_range_map =
      FindOrNull(function_body_outputs_, node_name);
  if (outputs_range_map != nullptr) {
    for (const auto& el : *outputs_range_map) {
      const auto& output_name = el.first;
      const auto& output_range = el.second;
      if (position >= output_range.first && position < output_range.second) {
        int pos = position - output_range.first;
        *func_def_input =
            strings::StrCat(node_name, ":", output_name, ":", pos);
        return Status::OK();
      }
    }
  }

  return errors::InvalidArgument("Unknown graph def input: ", graph_def_input);
}

Status GrapplerFunctionConnectivity::AsFunctionDefNode(
    NodeDef* function_body_node) const {
  string func_def_input;

  for (int i = 0; i < function_body_node->input_size(); ++i) {
    TF_RETURN_IF_ERROR(
        AsFunctionDefInput(function_body_node->input(i), &func_def_input));
    function_body_node->set_input(i, func_def_input);
  }

  return Status::OK();
}

Status GrapplerFunctionItemInstantiation::GetTypeAttr(
    const string& type_attr_name, DataType* data_type) const {
  auto it = func_instantiation_attr_->find(type_attr_name);
  if (it == func_instantiation_attr_->end()) {
    return errors::InvalidArgument("Type attribute ", type_attr_name,
                                   " is not defined");
  } else if (it->second.type() == DT_INVALID) {
    return errors::InvalidArgument("Type attribute ", type_attr_name,
                                   " is not defined with a valid type");
  } else {
    *data_type = it->second.type();
  }
  return Status::OK();
}

Status GrapplerFunctionItemInstantiation::GetArgType(
    const OpDef::ArgDef& arg, DataType* data_type) const {
  if (arg.type() != DT_INVALID) {
    *data_type = arg.type();
  } else {
    if (!arg.type_list_attr().empty() || !arg.number_attr().empty()) {
      return errors::InvalidArgument(
          "Arguments with sequence of tensors are not supported. Unsupported "
          "argument name: ",
          arg.name());
    }
    TF_RETURN_IF_ERROR(GetTypeAttr(arg.type_attr(), data_type));
  }
  return Status::OK();
}

GrapplerFunctionItem::GrapplerFunctionItem(
    const string& func_name, const AttrValueMap& func_attr,
    const std::vector<InputArgExpansion>& input_arg_expansions,
    const std::vector<OutputArgExpansion>& output_arg_expansions,
    GraphDef&& function_body)
    : func_attr_(func_attr),
      input_arg_expansions_(input_arg_expansions),
      output_arg_expansions_(output_arg_expansions) {
  id = func_name;
  // Fill the feed nodes with input placeholders
  for (const InputArgExpansion& input_arg : input_arg_expansions_) {
    for (const string& placeholder : input_arg.placeholders) {
      feed.emplace_back(placeholder, Tensor());
      input_arg_placeholders_.insert(placeholder);
    }
  }
  // Fill the fetch nodes with outputs
  for (const OutputArgExpansion& output_arg : output_arg_expansions_) {
    for (const string& output_tensor : output_arg.output_tensors) {
      fetch.push_back(output_tensor);
    }
  }
  // Swap the graph body
  graph.Swap(&function_body);
}

const std::vector<InputArgExpansion>& GrapplerFunctionItem::inputs() const {
  return input_arg_expansions_;
}

const InputArgExpansion& GrapplerFunctionItem::input(int i) const {
  return input_arg_expansions_[i];
}

const std::size_t GrapplerFunctionItem::input_size() const {
  return input_arg_expansions_.size();
}

bool GrapplerFunctionItem::IsInputPlaceholder(const string& node_name) const {
  return input_arg_placeholders_.find(node_name) !=
         input_arg_placeholders_.end();
}

const std::vector<OutputArgExpansion>& GrapplerFunctionItem::outputs() const {
  return output_arg_expansions_;
}

const OutputArgExpansion& GrapplerFunctionItem::output(int i) const {
  return output_arg_expansions_[i];
}

const std::size_t GrapplerFunctionItem::output_size() const {
  return output_arg_expansions_.size();
}

const AttrValueMap& GrapplerFunctionItem::func_attr() const {
  return func_attr_;
}

const GraphDef& GrapplerFunctionItem::function_body() const { return graph; }

GraphDef& GrapplerFunctionItem::mutable_function_body() { return graph; }

GrapplerFunctionItem& GrapplerFunctionItem::SwapFunctionBody(GraphDef&& other) {
  graph.Swap(&other);
  return *this;
}

std::vector<string> OutputTensors(const GrapplerFunctionItem& item) {
  std::vector<string> output_tensors;
  for (const OutputArgExpansion& output : item.outputs()) {
    for (const string& tensor : output.output_tensors) {
      output_tensors.push_back(tensor);
    }
  }
  return output_tensors;
}

Status MakeGrapplerFunctionItem(const FunctionDef& func,
                                const AttrValueMap& func_instantiation_attr,
                                const FunctionLibraryDefinition& flib,
                                GrapplerFunctionItem* item) {
  const OpDef& signature = func.signature();

  if (signature.name().empty()) {
    return errors::InvalidArgument("Function name must be specified");
  }

  // Function types will be resolved from function instantiation attributes. All
  // other attributes will be lost during conversion to FunctionDef.
  for (const OpDef::AttrDef& attr : signature.attr()) {
    if (attr.type() != "type") {
      return errors::InvalidArgument(
          "Function signature must have only type attributes");
    }
  }

  // Helper methods to lookup function instantiation attributes
  GrapplerFunctionItemInstantiation instantiation(&func_instantiation_attr);

  // Mapping from FunctionDef input format (name[:output][:position]) to
  // GraphDef input format (name[:position])
  GrapplerFunctionConnectivity connectivity;

  std::vector<InputArgExpansion> inputs;
  std::vector<OutputArgExpansion> outputs;

  // Function body shares the library with the graph that instantiated it.
  GraphDef function_body;
  *function_body.mutable_library() = flib.ToProto();

  // TODO(ezhulenev): support functions with tensor sequence inputs/outputs

  // Make sure that there is no tensor sequences in outputs
  for (const OpDef::ArgDef& output : signature.output_arg()) {
    if (!output.type_list_attr().empty() || !output.number_attr().empty()) {
      return errors::InvalidArgument(
          "Outputs with sequence of tensors are not supported. Unsupported "
          "output: ",
          output.name());
    }
  }

  // For each input argument create a placeholder in function body.
  for (const OpDef::ArgDef& input : signature.input_arg()) {
    if (!input.type_list_attr().empty() || !input.number_attr().empty()) {
      return errors::InvalidArgument(
          "Inputs with sequence of tensors are not supported. Unsupported "
          "input: ",
          input.name());
    }

    DataType input_data_type;
    TF_RETURN_IF_ERROR(instantiation.GetArgType(input, &input_data_type));

    NodeDef* placeholder = function_body.add_node();
    placeholder->set_name(input.name());
    placeholder->set_op("Placeholder");
    (*placeholder->mutable_attr())["T"].set_type(input_data_type);

    InputArgExpansion input_expansion{/*input_name=*/input.name(),
                                      /*data_type=*/input_data_type,
                                      /*placeholders=*/{input.name()}};
    connectivity.RegisterInputArgExpansion(input_expansion);
    inputs.push_back(input_expansion);
  }

  // Add all function nodes to the function body
  for (const NodeDef& func_def_node : func.node_def()) {
    NodeDef* new_node = function_body.add_node();
    *new_node = func_def_node;

    // Resolve all placeholder values using function instantiation attributes.
    TF_RETURN_IF_ERROR(ResolveFunctionBodyNodeAttrPlaceholders(
        func_instantiation_attr, new_node));
    // Register node output range in a function connectivity.
    TF_RETURN_IF_ERROR(
        RegisterFunctionBodyOutputs(flib, func_def_node, &connectivity));
  }

  // Rewrite inputs to use GraphDef format
  for (NodeDef& node : *function_body.mutable_node()) {
    TF_RETURN_IF_ERROR(connectivity.ExpandNodeInputs(&node));
  }

  // Add function outputs
  for (const OpDef::ArgDef& out : signature.output_arg()) {
    std::vector<string> output_tensors;
    auto ret = func.ret().find(out.name());
    TF_RETURN_IF_ERROR(
        ret != func.ret().end()
            // Expand outputs using provided output mapping
            ? connectivity.ExpandFunctionDefInput(ret->second, &output_tensors)
            // Otherwise output must be one of the function inputs
            : connectivity.ExpandFunctionDefInput(out.name(), &output_tensors));

    DataType output_data_type;
    TF_RETURN_IF_ERROR(instantiation.GetArgType(out, &output_data_type));

    OutputArgExpansion output{/*output_name=*/out.name(),
                              /*data_type=*/output_data_type,
                              /*output_tensors=*/output_tensors};
    outputs.push_back(output);
  }

  *item = GrapplerFunctionItem(
      /*func_name=*/signature.name(),
      /*func_attr=*/AttrValueMap(func.attr().begin(), func.attr().end()),
      inputs, outputs, std::move(function_body));
  return Status::OK();
}

// Register GrapplerFunctionItem input arg expansion and function body outputs
// in the GrapplerFunctionConnectivity
Status RegisterGrapplerFunctionConnectivity(
    const GrapplerFunctionItem& item, const FunctionLibraryDefinition& flib,
    GrapplerFunctionConnectivity* connectivity) {
  for (const InputArgExpansion& input : item.inputs()) {
    connectivity->RegisterInputArgExpansion(input);
  }
  for (const NodeDef& func_body_node : item.function_body().node()) {
    TF_RETURN_IF_ERROR(
        RegisterFunctionBodyOutputs(flib, func_body_node, connectivity));
  }
  return Status::OK();
}

Status MakeSpecializedFunctionDef(const GrapplerFunctionItem& item,
                                  const FunctionLibraryDefinition& flib,
                                  FunctionDef* func) {
  func->mutable_signature()->set_name(item.id);

  // Build a GrapplerFunctionConnectivity from inputs and new function body.
  GrapplerFunctionConnectivity connectivity;
  TF_RETURN_IF_ERROR(
      RegisterGrapplerFunctionConnectivity(item, flib, &connectivity));

  // Add function input arguments.
  for (const InputArgExpansion& input_arg : item.inputs()) {
    OpDef::ArgDef arg_def;
    arg_def.set_name(input_arg.input_name);
    arg_def.set_type(input_arg.data_type);
    *func->mutable_signature()->add_input_arg() = arg_def;
  }

  // Add function output arguments.
  for (const OutputArgExpansion& output_arg : item.outputs()) {
    OpDef::ArgDef arg_def;
    arg_def.set_name(output_arg.output_name);
    arg_def.set_type(output_arg.data_type);
    *func->mutable_signature()->add_output_arg() = arg_def;

    CHECK(output_arg.output_tensors.size() == 1)  // do some sanity checking
        << "Outputs of tensor sequences are not supported";

    string ret;
    for (const string& output_tensor : output_arg.output_tensors) {
      TF_RETURN_IF_ERROR(connectivity.AsFunctionDefInput(output_tensor, &ret));
      (*func->mutable_ret())[output_arg.output_name] = ret;
    }
  }

  // Copy function definition specific attributes.
  for (const auto& attr : item.func_attr()) {
    const auto& attr_name = attr.first;
    const auto& attr_value = attr.second;
    (*func->mutable_attr())[attr_name] = attr_value;
  }

  // Copy function body nodes to the FunctionDef and update input format
  for (const NodeDef& func_body_node : item.function_body().node()) {
    // Do not copy input placeholders
    if (item.IsInputPlaceholder(func_body_node.name())) continue;

    NodeDef* func_def_node = func->add_node_def();
    *func_def_node = func_body_node;
    TF_RETURN_IF_ERROR(connectivity.AsFunctionDefNode(func_def_node));
  }

  return Status::OK();
}

}  // end namespace grappler
}  // end namespace tensorflow
