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

#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"

namespace xla {

namespace {

template <typename FloatT>
void PopulateWithRandomFloatingPointData(Literal* literal) {
  CHECK_EQ(literal->shape().element_type(),
           primitive_util::NativeToPrimitiveType<FloatT>());
  std::minstd_rand0 engine;
  std::uniform_real_distribution<FloatT> generator(0.0f, 1.0f);
  TF_CHECK_OK(literal->Populate<FloatT>(
      [&](tensorflow::gtl::ArraySlice<int64> /*indices*/) {
        return generator(engine);
      }));
}

template <typename IntT>
void PopulateWithRandomIntegralData(Literal* literal) {
  CHECK_EQ(literal->shape().element_type(),
           primitive_util::NativeToPrimitiveType<IntT>());
  std::minstd_rand0 engine;
  std::uniform_int_distribution<IntT> generator(
      std::numeric_limits<IntT>::lowest(), std::numeric_limits<IntT>::max());
  TF_CHECK_OK(literal->Populate<IntT>(
      [&](tensorflow::gtl::ArraySlice<int64> /*indices*/) {
        return generator(engine);
      }));
}

// Matches binary addition computations.
bool LooksLikeSum(const HloComputation& computation) {
  const HloInstruction* const root = computation.root_instruction();
  return root->opcode() == HloOpcode::kAdd &&
         computation.num_parameters() == 2 &&
         root->operand(0)->opcode() == HloOpcode::kParameter &&
         root->operand(1)->opcode() == HloOpcode::kParameter &&
         root->operand(0) != root->operand(1);
}

// Reduce, ReduceWindow, and SelectAndScatter ops may use binary addition,
// which requires an init_value of 0 rather than a random value.
bool NeedsZeroInitValue(const HloUse& use) {
  const HloInstruction* const instruction = use.instruction;
  const HloOpcode opcode = instruction->opcode();
  const int64 op_num = use.operand_number;
  return (
      ((opcode == HloOpcode::kReduce || opcode == HloOpcode::kReduceWindow) &&
       op_num == 1 && LooksLikeSum(*instruction->to_apply())) ||
      (opcode == HloOpcode::kSelectAndScatter && op_num == 2 &&
       LooksLikeSum(*instruction->scatter())));
}

// Generate random values that are constrained to the input_shape minus the
// output_shape so as not to produce wrapping slices, for instance.
std::unique_ptr<Literal> MakeRandomNonwrappingSliceIndex(
    const Shape& input_shape, const Shape& slice_shape) {
  const int64 rank = ShapeUtil::Rank(input_shape);
  std::vector<int32> start_indices(rank);
  std::minstd_rand0 engine;
  for (int i = 0; i < rank; ++i) {
    const int32 upper_bound = ShapeUtil::GetDimension(input_shape, i) -
                              ShapeUtil::GetDimension(slice_shape, i);
    std::uniform_int_distribution<int32> generator(0, upper_bound);
    start_indices[i] = generator(engine);
  }
  return Literal::CreateR1<int32>(start_indices);
}

// Use dataflow analysis on each parameter to see if there are uses that would
// be problematic when generating input data.  Returns the list of instructions
// that correspond to their uses.
//
// Should be paired with the CreateLiteralForConstrainedUses() function below.
std::vector<HloInstruction*> FindConstrainedUses(
    const HloDataflowAnalysis& dataflow, const HloInstruction& param) {
  std::vector<HloInstruction*> constrained_uses;
  for (const auto& pair : dataflow.GetInstructionValueSet(&param)) {
    const HloValue& value = dataflow.GetUniqueValueAt(&param, pair.first);
    for (const HloUse& use : value.uses()) {
      HloInstruction* instruction = use.instruction;
      const HloOpcode opcode = instruction->opcode();
      const int64 op_num = use.operand_number;
      if ((opcode == HloOpcode::kDynamicSlice && op_num == 1) ||
          (opcode == HloOpcode::kDynamicUpdateSlice && op_num == 2)) {
        constrained_uses.push_back(instruction);
      } else if (opcode == HloOpcode::kFusion) {
        const HloInstruction* const to_analyze =
            instruction->fused_parameter(op_num);
        auto fused_uses = FindConstrainedUses(dataflow, *to_analyze);
        constrained_uses.insert(constrained_uses.end(), fused_uses.begin(),
                                fused_uses.end());
      } else if (NeedsZeroInitValue(use)) {
        constrained_uses.push_back(instruction);
      }
    }
  }
  return constrained_uses;
}

// Given a parameter, generate a random Literal to use as input if there exist
// no constrained uses in the dataflow graph.  If such constraints exist,
// generate a constrained literal (either bounded in the case of indices, or
// zero in the case of init_values for reductions).
StatusOr<std::unique_ptr<Literal>> CreateLiteralForConstrainedUses(
    const tensorflow::gtl::ArraySlice<HloInstruction*> constrained_uses,
    const HloInstruction& param) {
  const auto count = constrained_uses.size();
  if (count > 1) {
    return Unimplemented("multiple constrained uses not yet supported");
  }

  if (count == 0) {
    return MakeFakeLiteral(param.shape());
  }

  const HloInstruction* const use = constrained_uses[0];
  switch (use->opcode()) {
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
      return MakeRandomNonwrappingSliceIndex(use->operand(0)->shape(),
                                             use->shape());
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kSelectAndScatter:
      return Literal::CreateFromShape(param.shape());
    default:
      return Unimplemented("constrained use given; no equivalent literal");
  }
}

// Given a module entry parameter, use the dataflow analysis to see if a
// special case literal must be created, or if we can generate fake data.
StatusOr<std::unique_ptr<Literal>> MakeConstrainedArgument(
    const HloDataflowAnalysis& dataflow, const HloInstruction& param) {
  const auto constrained_uses = FindConstrainedUses(dataflow, param);
  return CreateLiteralForConstrainedUses(constrained_uses, param);
}

}  // namespace

StatusOr<std::unique_ptr<Literal>> MakeFakeLiteral(const Shape& shape) {
  if (ShapeUtil::IsTuple(shape)) {
    std::vector<std::unique_ptr<Literal>> elements;
    for (const Shape& element_shape : shape.tuple_shapes()) {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<Literal> element,
                          MakeFakeLiteral(element_shape));
      elements.push_back(std::move(element));
    }
    return Literal::MakeTupleOwned(std::move(elements));
  }
  std::unique_ptr<Literal> literal = Literal::CreateFromShape(shape);
  switch (shape.element_type()) {
    case F32:
      PopulateWithRandomFloatingPointData<float>(literal.get());
      break;
    case F64:
      PopulateWithRandomFloatingPointData<double>(literal.get());
      break;
    case S8:
      PopulateWithRandomIntegralData<int8>(literal.get());
      break;
    case U8:
      PopulateWithRandomIntegralData<uint8>(literal.get());
      break;
    case S16:
      PopulateWithRandomIntegralData<int16>(literal.get());
      break;
    case U16:
      PopulateWithRandomIntegralData<uint16>(literal.get());
      break;
    case S32:
      PopulateWithRandomIntegralData<int32>(literal.get());
      break;
    case U32:
      PopulateWithRandomIntegralData<uint32>(literal.get());
      break;
    case S64:
      PopulateWithRandomIntegralData<int64>(literal.get());
      break;
    case U64:
      PopulateWithRandomIntegralData<uint64>(literal.get());
      break;
    case PRED: {
      std::uniform_int_distribution<int> generator(0, 1);
      std::minstd_rand0 engine;
      TF_CHECK_OK(literal->Populate<bool>(
          [&](tensorflow::gtl::ArraySlice<int64> /*indices*/) {
            return generator(engine);
          }));
      break;
    }
    default:
      return Unimplemented("Unsupported type for fake literal generation: %s",
                           ShapeUtil::HumanString(shape).c_str());
  }
  return std::move(literal);
}

StatusOr<std::vector<std::unique_ptr<Literal>>> MakeFakeArguments(
    HloModule* const module) {
  TF_ASSIGN_OR_RETURN(auto dataflow, HloDataflowAnalysis::Run(module));
  const auto params = module->entry_computation()->parameter_instructions();
  std::vector<std::unique_ptr<Literal>> arguments(params.size());
  for (int i = 0; i < params.size(); ++i) {
    TF_ASSIGN_OR_RETURN(arguments[i],
                        MakeConstrainedArgument(*dataflow, *params[i]));
  }
  return std::move(arguments);
}

Status VerifyHloModule(const perftools::gputools::Platform& platform,
                       HloModule* const module) {
  return HloVerifier(
             std::bind(
                 &TransferManager::GetByteSizeRequirement,
                 TransferManager::GetForPlatform(&platform).ConsumeValueOrDie(),
                 std::placeholders::_1))
      .Run(module)
      .status();
}

}  // namespace xla
