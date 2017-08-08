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

#include "tensorflow/compiler/xla/service/hlo_alias_analysis.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_buffer.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

using ::tensorflow::str_util::Join;
using ::tensorflow::strings::StrAppend;
using ::tensorflow::strings::StrCat;

HloAliasAnalysis::HloAliasAnalysis(HloModule* module) : module_(module) {}

void HloAliasAnalysis::InitializeBufferSets() {
  std::unordered_map<HloValue::Id, const HloBuffer*> value_to_buffer;

  // Initially define a buffer for every HloValue in the module.
  for (const HloValue* value : dataflow_analysis_->values()) {
    HloBuffer* buffer = NewHloBuffer();
    buffer->AddValue(*value);
    value_to_buffer[value->id()] = buffer;
  }

  // Construct the Instruction buffer set to contain the HloBuffers for each
  // HloValue in the InstructionValueSet.
  for (auto& computation : module_->computations()) {
    for (auto& instruction : computation->instructions()) {
      buffer_sets_.emplace(std::piecewise_construct,
                           std::forward_as_tuple(instruction.get()),
                           std::forward_as_tuple(instruction->shape()));
      dataflow_analysis_->GetInstructionValueSet(instruction.get())
          .ForEachElement(
              [this, &instruction, &value_to_buffer](
                  const ShapeIndex& index, const HloValueSet& value_set) {
                for (const HloValue* value : value_set.values()) {
                  const HloBuffer* buffer = value_to_buffer.at(value->id());
                  GetBufferSet(instruction.get(), index).AddBuffer(buffer);
                }
              });
    }
  }
}

void HloAliasAnalysis::CombineBuffers(
    tensorflow::gtl::ArraySlice<HloBuffer::Id> buffer_ids) {
  VLOG(4) << "Combining buffers: " << Join(buffer_ids, ", ");

  if (buffer_ids.size() < 2) {
    return;
  }

  // Merging buffers invalidates the buffer vector.
  buffers_vector_.clear();

  // Add all values from all buffers to the first buffer in the list.
  HloBuffer& unified_buffer = GetBuffer(buffer_ids[0]);
  for (int i = 1; i < buffer_ids.size(); ++i) {
    const HloBuffer::Id buffer_id = buffer_ids[i];
    const HloBuffer& buffer = GetBuffer(buffer_id);

    VLOG(4) << "Eliminating buffer: " << buffer_id;

    // Add all values held by the buffer-to-eliminate to the unified buffer.
    for (const HloValue* value : buffer.values()) {
      unified_buffer.AddValue(*value);
    }

    // Iterate through all positions where the buffer-to-eliminate exists and
    // replace it with the unified buffer.
    for (const HloPosition& position : buffer.positions()) {
      VLOG(4) << "Replacing in " << position;
      GetBufferSet(position.instruction, position.index)
          .RemoveBufferOrDie(buffer_id);
      GetBufferSet(position.instruction, position.index)
          .AddBuffer(&unified_buffer);
    }

    buffers_.erase(buffer_id);
  }

  TF_DCHECK_OK(Verify());
}

Status HloAliasAnalysis::Verify() const {
  // Verify every HloBuffer in buffers_ exists somewhere in an HloBufferSet and
  // verify that every HloBuffer in the HloBufferSets exists somewhere in
  // buffers_.
  tensorflow::gtl::FlatSet<HloBuffer::Id> buffers_in_sets;
  for (auto& pair : buffer_sets_) {
    const InstructionBufferSet& instruction_buffer_set = pair.second;
    TF_RETURN_IF_ERROR(instruction_buffer_set.ForEachElementWithStatus(
        [this, &buffers_in_sets](const ShapeIndex& index,
                                 const HloBufferSet& buffer_set) -> Status {
          for (const HloBuffer* buffer : buffer_set.buffers()) {
            TF_RET_CHECK(ContainsKey(buffers_, buffer->id()));
            buffers_in_sets.insert(buffer->id());
          }
          return Status::OK();
        }));
  }
  for (auto& pair : buffers_) {
    const HloBuffer::Id buffer_id = pair.first;
    const HloBuffer& buffer = pair.second;
    TF_RET_CHECK(buffer_id == buffer.id());
    TF_RET_CHECK(ContainsKey(buffers_in_sets, buffer_id));
  }
  return Status::OK();
}

void HloAliasAnalysis::FlattenInstructionBufferSets(
    tensorflow::gtl::ArraySlice<const HloInstruction*> instructions) {
  VLOG(4) << "Flattening buffer sets of instructions: "
          << Join(instructions, ", ",
                  [this](string* out, const HloInstruction* instruction) {
                    StrAppend(out, instruction->name());
                  });
  if (instructions.size() < 2) {
    return;
  }
  ShapeUtil::ForEachSubshape(
      instructions[0]->shape(),
      [this, instructions](const Shape& /*subshape*/, const ShapeIndex& index) {
        // Gather all HloBuffers contained in all the buffer sets of the
        // given instructions at the current index.
        std::vector<HloBuffer::Id> to_unify;
        for (const HloInstruction* instruction : instructions) {
          const HloBufferSet& buffer_set = GetBufferSet(instruction, index);
          for (const HloBuffer* buffer : buffer_set.buffers()) {
            to_unify.push_back(buffer->id());
          }
        }
        // Sort and uniquify buffer ids to combine.
        std::sort(to_unify.begin(), to_unify.end());
        to_unify.erase(std::unique(to_unify.begin(), to_unify.end()),
                       to_unify.end());

        CombineBuffers(to_unify);
      });
}

HloBuffer* HloAliasAnalysis::NewHloBuffer() {
  HloBuffer::Id buffer_id = next_buffer_id_++;
  auto emplaced = buffers_.emplace(std::piecewise_construct,
                                   std::forward_as_tuple(buffer_id),
                                   std::forward_as_tuple(buffer_id));
  CHECK(emplaced.second);
  return &emplaced.first->second;
}

string HloAliasAnalysis::ToString() const {
  string out = StrCat("HloAliasAnalysis, module ", module_->name(), "\n");
  StrAppend(&out, "  Instruction buffer sets:\n");
  for (const std::unique_ptr<HloComputation>& computation :
       module_->computations()) {
    for (const std::unique_ptr<HloInstruction>& instruction :
         computation->instructions()) {
      StrAppend(&out, "    ", instruction->name(), ":\n");
      if (ShapeUtil::IsTuple(instruction->shape())) {
        GetInstructionBufferSet(instruction.get())
            .ForEachElement([this, &out](const ShapeIndex& index,
                                         const HloBufferSet& buffer_set) {
              StrAppend(&out, "      tuple index ", index.ToString(), ":\n");
              StrAppend(&out, "        ", buffer_set.ToString(), "\n");
            });
      } else {
        const HloBufferSet top_level_buffer_set =
            GetBufferSet(instruction.get());
        StrAppend(&out, "      ", top_level_buffer_set.ToString(), "\n");
      }
    }
  }
  return out;
}

const InstructionBufferSet& HloAliasAnalysis::GetInstructionBufferSet(
    const HloInstruction* instruction) const {
  return buffer_sets_.at(instruction);
}

InstructionBufferSet& HloAliasAnalysis::GetInstructionBufferSet(
    const HloInstruction* instruction) {
  return buffer_sets_.at(instruction);
}

const HloBufferSet& HloAliasAnalysis::GetBufferSet(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  return buffer_sets_.at(instruction).element(index);
}

HloBufferSet& HloAliasAnalysis::GetBufferSet(const HloInstruction* instruction,
                                             const ShapeIndex& index) {
  return *buffer_sets_.at(instruction).mutable_element(index);
}

const std::vector<const HloBuffer*>& HloAliasAnalysis::buffers() const {
  if (buffers_vector_.empty()) {
    // Lazily construct vector of buffers.
    buffers_vector_.reserve(buffers_.size());
    for (auto& pair : buffers_) {
      buffers_vector_.push_back(&pair.second);
    }
    std::sort(buffers_vector_.begin(), buffers_vector_.end(),
              [](const HloBuffer* a, const HloBuffer* b) {
                return a->id() < b->id();
              });
  } else {
    CHECK_EQ(buffers_vector_.size(), buffers_.size());
    for (const HloBuffer* buffer : buffers_vector_) {
      DCHECK(ContainsKey(buffers_, buffer->id()));
      DCHECK(&GetBuffer(buffer->id()) == buffer);
    }
  }
  return buffers_vector_;
}

/* static */
StatusOr<std::unique_ptr<HloAliasAnalysis>> HloAliasAnalysis::Run(
    HloModule* module) {
  VLOG(1) << "HloAliasAnalysis::Run on module " << module->name();
  XLA_VLOG_LINES(2, module->ToString());

  auto alias_analysis = WrapUnique(new HloAliasAnalysis(module));
  TF_ASSIGN_OR_RETURN(
      alias_analysis->dataflow_analysis_,
      HloDataflowAnalysis::Run(module, /*ssa_form=*/true,
                               /*bitcast_defines_value=*/false));

  alias_analysis->InitializeBufferSets();
  VLOG(3) << "Initial state:\n" << alias_analysis->ToString();

  // The while instruction updates its state inplace, so the inputs to the while
  // alias the while instruction, the parameters of the subcomputations, and the
  // root of the body subcomputation.
  for (auto& computation : module->computations()) {
    for (auto& instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kWhile) {
        VLOG(4) << "Flattening buffer sets at kWhile instruction: "
                << instruction->name();
        alias_analysis->FlattenInstructionBufferSets(
            {instruction->operand(0),
             instruction->while_body()->parameter_instruction(0),
             instruction->while_body()->root_instruction(),
             instruction->while_condition()->parameter_instruction(0),
             instruction.get()});
      }
    }
  }
  VLOG(1) << alias_analysis->ToString();
  return std::move(alias_analysis);
}

}  // namespace xla
