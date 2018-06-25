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

#include "tensorflow/compiler/xla/service/hlo_domain_map.h"

#include <algorithm>

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

/* static */ StatusOr<std::unique_ptr<HloDomainMap>> HloDomainMap::Create(
    HloComputation* computation, string domain_kind) {
  auto domain_map = WrapUnique(new HloDomainMap(std::move(domain_kind)));
  TF_RETURN_IF_ERROR(domain_map->Populate(computation));
  return std::move(domain_map);
}

/* static */ StatusOr<std::unique_ptr<HloDomainMap>> HloDomainMap::Create(
    HloModule* module, string domain_kind) {
  auto domain_map = WrapUnique(new HloDomainMap(std::move(domain_kind)));
  for (HloComputation* computation : module->computations()) {
    TF_RETURN_IF_ERROR(domain_map->Populate(computation));
  }
  return std::move(domain_map);
}

bool HloDomainMap::InSameDomain(HloInstruction* instruction1,
                                HloInstruction* instruction2) const {
  int64 domain_id1 = FindOrDefault(instruction_to_domain_, instruction1, -1);
  int64 domain_id2 = FindOrDefault(instruction_to_domain_, instruction2, -1);
  return domain_id1 >= 0 && domain_id1 == domain_id2;
}

Status HloDomainMap::TryProcessEmptyDomain(HloInstruction* instruction) {
  TF_RET_CHECK(instruction->opcode() == HloOpcode::kDomain);
  // We only check operands, so we are sure to not process the empty domain from
  // both sides.
  for (HloInstruction* operand : instruction->unique_operands()) {
    if (IsDomainInstruction(operand)) {
      auto domain = MakeUnique<DomainMetadata::Domain>();
      domain->enter_domains.insert(operand);
      domain->exit_domains.insert(instruction);
      TF_RETURN_IF_ERROR(InsertDomain(std::move(domain)));
    }
  }
  return Status::OK();
}

Status HloDomainMap::Populate(HloComputation* computation) {
  for (HloInstruction* instruction : computation->instructions()) {
    if (IsDomainInstruction(instruction)) {
      // If this is a kDomain of the kind we are currently processing, check
      // whether this is an "empty domain".
      TF_RETURN_IF_ERROR(TryProcessEmptyDomain(instruction));
      continue;
    }
    int64 domain_id = FindOrDefault(instruction_to_domain_, instruction, -1);
    if (domain_id >= 0) {
      // We have already processed this instruction.
      continue;
    }
    TF_ASSIGN_OR_RETURN(std::unique_ptr<DomainMetadata::Domain> domain,
                        CreateDomain(instruction));
    TF_RETURN_IF_ERROR(InsertDomain(std::move(domain)));
  }
  return Status::OK();
}

Status HloDomainMap::InsertDomain(
    std::unique_ptr<DomainMetadata::Domain> domain) {
  int64 domain_id = instruction_domains_.size();
  instruction_domains_.push_back(std::move(domain));
  for (HloInstruction* instruction : instruction_domains_.back()->reach_set) {
    instruction_to_domain_[instruction] = domain_id;
  }
  return Status::OK();
}

Status HloDomainMap::ExpandDomain(HloInstruction* instruction,
                                  DomainMetadata::Domain* domain) const {
  std::vector<HloInstruction*> in_queue;
  in_queue.push_back(instruction);
  while (!in_queue.empty()) {
    HloInstruction* current_instruction = in_queue.back();
    in_queue.pop_back();
    if (domain->reach_set.insert(current_instruction).second) {
      // We should not be finding instructions with assigned domain here.
      // If we assigned a domain to the instruction, it means that all the
      // instructions reached by it, should have a domain as well.
      int64 domain_id =
          FindOrDefault(instruction_to_domain_, current_instruction, -1);
      TF_RET_CHECK(domain_id < 0)
          << "Instruction " << current_instruction->ToString()
          << " already has domain " << domain_id;
      for (HloInstruction* operand : current_instruction->operands()) {
        if (IsDomainInstruction(operand)) {
          // The reach set instruction is a user of the domain instruction
          // (the instruction sees the kDomain as operand).
          // IOW the dataflow enters the domain through the kDomain instruction.
          domain->enter_domains.insert(operand);
        } else {
          in_queue.push_back(operand);
        }
      }
      for (HloInstruction* user : current_instruction->users()) {
        if (IsDomainInstruction(user)) {
          // The reach set instruction is an operand of the domain instruction
          // (the instruction sees the kDomain as user).
          // IOW the dataflow exits the domain through the kDomain instruction.
          domain->exit_domains.insert(user);
        } else {
          in_queue.push_back(user);
        }
      }
    }
  }
  return Status::OK();
}

StatusOr<std::unique_ptr<DomainMetadata::Domain>> HloDomainMap::CreateDomain(
    HloInstruction* instruction) const {
  auto domain = MakeUnique<DomainMetadata::Domain>();
  TF_RETURN_IF_ERROR(ExpandDomain(instruction, domain.get()));
  domain->instructions = MakeNonDomainInstructions(domain->reach_set);
  return std::move(domain);
}

bool HloDomainMap::IsDomainInstruction(HloInstruction* instruction) const {
  if (instruction->opcode() != HloOpcode::kDomain) {
    return false;
  }
  if (!domain_kind_.empty()) {
    if (instruction->user_side_metadata().Kind() != domain_kind_) {
      return false;
    }
    // Both user and operand side of the metadata must be of the same kind.
    CHECK(instruction->operand_side_metadata().Kind() == domain_kind_)
        << "Instruction " << instruction->ToString()
        << " has mismatching metadata kinds";
  }
  return true;
}

/* static */ std::vector<HloInstruction*>
HloDomainMap::MakeNonDomainInstructions(
    const tensorflow::gtl::FlatSet<HloInstruction*>& instruction_set) {
  std::vector<HloInstruction*> instructions;
  instructions.reserve(instruction_set.size());
  for (HloInstruction* instruction : instruction_set) {
    if (instruction->opcode() != HloOpcode::kDomain) {
      instructions.push_back(instruction);
    }
  }
  std::sort(instructions.begin(), instructions.end(),
            [](HloInstruction* a, HloInstruction* b) {
              return a->unique_id() < b->unique_id();
            });
  return instructions;
}

}  // namespace xla
