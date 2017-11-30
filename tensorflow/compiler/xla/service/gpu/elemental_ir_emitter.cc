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

#include "tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.h"

#include <stddef.h>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
// IWYU pragma: no_include "llvm/IR/Attributes.gen.inc"
// IWYU pragma: no_include "llvm/IR/Intrinsics.gen.inc"
#include "llvm/ADT/APInt.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace xla {
namespace gpu {

using llvm_ir::IrArray;
using llvm_ir::IrName;
using llvm_ir::SetToFirstInsertPoint;
using tensorflow::strings::StrAppend;

// Returns whether operand is a floating-point literal with the given value.
bool IsFPLiteralWithValue(const HloInstruction* operand, float value) {
  return operand->opcode() == HloOpcode::kConstant &&
         operand->literal().IsAllFloat(value);
}

GpuElementalIrEmitter::GpuElementalIrEmitter(
    const HloModuleConfig& hlo_module_config, llvm::Module* module,
    llvm::IRBuilder<>* ir_builder, NestedComputer compute_nested)
    : ElementalIrEmitter(hlo_module_config, module, ir_builder),
      hlo_module_config_(hlo_module_config),
      compute_nested_(std::move(compute_nested)) {}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitLibdeviceMathCall(
    const string& callee_name,
    tensorflow::gtl::ArraySlice<llvm::Value*> operands,
    tensorflow::gtl::ArraySlice<PrimitiveType> input_types,
    PrimitiveType output_type) const {
  // The libdevice math functions differentiate between "double" and "float" by
  // appending an 'f' to the function's name.
  string munged_callee = callee_name;
  switch (output_type) {
    case F32:
      StrAppend(&munged_callee, "f");
      break;
    case F64:
      break;
    default:
      return Unimplemented("Bad type for libdevice math call: %s",
                           PrimitiveType_Name(output_type).c_str());
  }
  return EmitMathCall(munged_callee, operands, input_types, output_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitLlvmIntrinsicMathCall(
    const string& callee_name,
    tensorflow::gtl::ArraySlice<llvm::Value*> operands,
    tensorflow::gtl::ArraySlice<PrimitiveType> input_types,
    PrimitiveType output_type) const {
  // llvm intrinsics differentiate between float/double functions via the ".f32"
  // and ".f64" suffixes.
  string munged_callee = callee_name;
  switch (output_type) {
    case F32:
      StrAppend(&munged_callee, ".f32");
      break;
    case F64:
      StrAppend(&munged_callee, ".f64");
      break;
    default:
      return Unimplemented("Bad type for llvm intrinsic math call: %s",
                           PrimitiveType_Name(output_type).c_str());
  }
  return EmitMathCall(munged_callee, operands, input_types, output_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitMathCall(
    const string& callee_name,
    tensorflow::gtl::ArraySlice<llvm::Value*> operands,
    tensorflow::gtl::ArraySlice<PrimitiveType> input_types,
    PrimitiveType output_type) const {
  // Binary math functions transform are of type [T] -> T.
  for (PrimitiveType input_type : input_types) {
    if (output_type != input_type) {
      return Unimplemented("Input type ≠ output type: %s ≠ %s",
                           PrimitiveType_Name(input_type).c_str(),
                           PrimitiveType_Name(output_type).c_str());
    }
  }

  return EmitDeviceFunctionCall(
      callee_name, operands, input_types, output_type,
      {llvm::Attribute::ReadNone, llvm::Attribute::NoUnwind});
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitFloatBinaryOp(
    const HloInstruction* op, llvm::Value* lhs_value,
    llvm::Value* rhs_value) const {
  PrimitiveType lhs_input_type = op->operand(0)->shape().element_type();
  PrimitiveType rhs_input_type = op->operand(1)->shape().element_type();
  PrimitiveType output_type = op->shape().element_type();
  switch (op->opcode()) {
    case HloOpcode::kAtan2:
      return EmitLibdeviceMathCall("__nv_atan2", {lhs_value, rhs_value},
                                   {lhs_input_type, rhs_input_type},
                                   output_type);
    case HloOpcode::kRemainder: {
      return EmitLibdeviceMathCall("__nv_fmod", {lhs_value, rhs_value},
                                   {lhs_input_type, rhs_input_type},
                                   output_type);
    }
    case HloOpcode::kPower: {
      return EmitPowerOp(op, lhs_value, rhs_value);
    }
    default:
      return ElementalIrEmitter::EmitFloatBinaryOp(op, lhs_value, rhs_value);
  }
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitPowerOp(
    const HloInstruction* op, llvm::Value* lhs_value,
    llvm::Value* rhs_value) const {
  CHECK_EQ(op->opcode(), HloOpcode::kPower);
  PrimitiveType lhs_input_type = op->operand(0)->shape().element_type();
  PrimitiveType rhs_input_type = op->operand(1)->shape().element_type();
  PrimitiveType output_type = op->shape().element_type();
  llvm::Type* llvm_ty = lhs_value->getType();

  auto make_sqrt = [&, this]() -> StatusOr<llvm::Value*> {
    // NVPTX has four relevant square root instructions:
    //   sqrt.approx{.ftz}.f32
    //   sqrt.rn{.ftz}.f32
    //   sqrt.rn.f64
    //   rsqrt.approx.f64
    // We rely on LLVM's NVPTX backend to pick the right one based on our
    // fast-math options.  (If fast-math is enabled, llvm may compute the 64-bit
    // sqrt from the rsqrt approximation.)
    return EmitLlvmIntrinsicMathCall("llvm.sqrt", {lhs_value}, {lhs_input_type},
                                     output_type);
  };

  const HloInstruction* rhs = op->operand(1);
  if (IsFPLiteralWithValue(rhs, .5)) {
    VLOG(10) << "emitting pow(A, .5) as sqrt(A): " << op->ToString();
    return make_sqrt();
  }

  if (hlo_module_config_.debug_options().xla_enable_fast_math() &&
      IsFPLiteralWithValue(rhs, -.5)) {
    VLOG(10) << "emitting pow(A, -.5) as 1/sqrt(A): " << op->ToString();
    // LLVM's NVPTX backend knows how to transform 1/sqrt(A) into the NVPTX
    // rsqrt.approx instruction.
    TF_ASSIGN_OR_RETURN(auto* sqrt, make_sqrt());
    return ir_builder_->CreateFDiv(llvm::ConstantFP::get(llvm_ty, 1), sqrt);
  }

  VLOG(10) << "emitting pow as regular call to pow(): " << op->ToString();
  return EmitLibdeviceMathCall("__nv_pow", {lhs_value, rhs_value},
                               {lhs_input_type, rhs_input_type}, output_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitErfcInv(
    PrimitiveType prim_type, llvm::Value* value) const {
  return EmitLibdeviceMathCall("__nv_erfcinv", {value}, {prim_type}, prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitFloatUnaryOp(
    const HloInstruction* op, llvm::Value* operand_value) const {
  PrimitiveType input_type = op->operand(0)->shape().element_type();
  PrimitiveType output_type = op->shape().element_type();
  switch (op->opcode()) {
    case HloOpcode::kExp:
      return EmitLibdeviceMathCall("__nv_exp", {operand_value}, {input_type},
                                   output_type);
    case HloOpcode::kFloor:
      return EmitLibdeviceMathCall("__nv_floor", {operand_value}, {input_type},
                                   output_type);
    case HloOpcode::kCeil:
      return EmitLibdeviceMathCall("__nv_ceil", {operand_value}, {input_type},
                                   output_type);
    case HloOpcode::kLog:
      return EmitLibdeviceMathCall("__nv_log", {operand_value}, {input_type},
                                   output_type);
    case HloOpcode::kCos:
      return EmitLibdeviceMathCall("__nv_cos", {operand_value}, {input_type},
                                   output_type);
    case HloOpcode::kSin:
      return EmitLibdeviceMathCall("__nv_sin", {operand_value}, {input_type},
                                   output_type);
    case HloOpcode::kTanh:
      return EmitLibdeviceMathCall("__nv_tanh", {operand_value}, {input_type},
                                   output_type);
    default:
      return ElementalIrEmitter::EmitFloatUnaryOp(op, operand_value);
  }
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitComplexBinaryOp(
    const HloInstruction* op, llvm::Value* lhs_value,
    llvm::Value* rhs_value) const {
  PrimitiveType input_type = op->operand(0)->shape().element_type();
  TF_RET_CHECK(primitive_util::IsComplexType(input_type));
  PrimitiveType component_type =
      primitive_util::ComplexComponentType(input_type);
  switch (op->opcode()) {
    case HloOpcode::kPower: {
      // (a+bi)^(c+di) =
      //    (a*a+b*b)^(0.5c) * exp(-d*atan2(b,a)) * (cos(q) + i*sin(q)),
      //    where q = c*atan2(b,a)+0.5d*ln(a*a+b*b)
      auto a = EmitExtractReal(lhs_value);
      auto b = EmitExtractImag(lhs_value);
      auto c = EmitExtractReal(rhs_value);
      auto d = EmitExtractImag(rhs_value);
      auto aa_p_bb = ir_builder_->CreateFAdd(ir_builder_->CreateFMul(a, a),
                                             ir_builder_->CreateFMul(b, b));
      auto one_half = llvm::ConstantFP::get(a->getType(), 0.5);
      auto half_c = ir_builder_->CreateFMul(one_half, c);

      TF_ASSIGN_OR_RETURN(
          auto aa_p_bb_to_half_c,
          EmitLibdeviceMathCall("__nv_pow", {aa_p_bb, half_c},
                                {component_type, component_type},
                                component_type));
      auto neg_d = ir_builder_->CreateFNeg(d);
      TF_ASSIGN_OR_RETURN(
          auto arg_lhs, EmitLibdeviceMathCall("__nv_atan2", {b, a},
                                              {component_type, component_type},
                                              component_type));
      auto neg_d_arg_lhs = ir_builder_->CreateFMul(neg_d, arg_lhs);
      TF_ASSIGN_OR_RETURN(
          auto e_to_neg_d_arg_lhs,
          EmitLibdeviceMathCall("__nv_exp", {neg_d_arg_lhs}, {component_type},
                                component_type));
      auto coeff =
          ir_builder_->CreateFMul(aa_p_bb_to_half_c, e_to_neg_d_arg_lhs);
      TF_ASSIGN_OR_RETURN(
          auto ln_aa_p_bb,
          EmitLibdeviceMathCall("__nv_log", {aa_p_bb}, {component_type},
                                component_type));
      auto half_d = ir_builder_->CreateFMul(one_half, d);
      auto q =
          ir_builder_->CreateFAdd(ir_builder_->CreateFMul(c, arg_lhs),
                                  ir_builder_->CreateFMul(half_d, ln_aa_p_bb));
      TF_ASSIGN_OR_RETURN(
          auto cos_q, EmitLibdeviceMathCall("__nv_cos", {q}, {component_type},
                                            component_type));
      TF_ASSIGN_OR_RETURN(
          auto sin_q, EmitLibdeviceMathCall("__nv_sin", {q}, {component_type},
                                            component_type));
      return EmitComposeComplex(op, ir_builder_->CreateFMul(coeff, cos_q),
                                ir_builder_->CreateFMul(coeff, sin_q));
    }
    default:
      return ElementalIrEmitter::EmitComplexBinaryOp(op, lhs_value, rhs_value);
  }
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitComplexUnaryOp(
    const HloInstruction* op, llvm::Value* operand_value) const {
  PrimitiveType input_type = op->operand(0)->shape().element_type();
  PrimitiveType component_type =
      primitive_util::IsComplexType(input_type)
          ? primitive_util::ComplexComponentType(input_type)
          : input_type;

  switch (op->opcode()) {
    case HloOpcode::kLog: {
      // log(a+bi) = .5*log(a^2+b^2) + i*atan2(b, a)
      auto a = EmitExtractReal(operand_value);
      auto b = EmitExtractImag(operand_value);
      llvm::Type* llvm_ty = a->getType();
      auto sum_sq = ir_builder_->CreateFAdd(ir_builder_->CreateFMul(a, a),
                                            ir_builder_->CreateFMul(b, b));
      TF_ASSIGN_OR_RETURN(
          auto log_sum_sq,
          EmitLibdeviceMathCall("__nv_log", {sum_sq}, {component_type},
                                component_type));
      TF_ASSIGN_OR_RETURN(
          auto angle, EmitLibdeviceMathCall("__nv_atan2", {b, a},
                                            {component_type, component_type},
                                            component_type));
      auto one_half = llvm::ConstantFP::get(llvm_ty, 0.5);
      return EmitComposeComplex(
          op, ir_builder_->CreateFMul(one_half, log_sum_sq), angle);
    }
    case HloOpcode::kExp: {
      // e^(a+bi) = e^a*(cos(b)+sin(b)i)
      auto b = EmitExtractImag(operand_value);
      TF_ASSIGN_OR_RETURN(
          auto exp_a,
          EmitLibdeviceMathCall("__nv_exp", {EmitExtractReal(operand_value)},
                                {component_type}, component_type));
      TF_ASSIGN_OR_RETURN(
          auto cos_b, EmitLibdeviceMathCall("__nv_cos", {b}, {component_type},
                                            component_type));
      TF_ASSIGN_OR_RETURN(
          auto sin_b, EmitLibdeviceMathCall("__nv_sin", {b}, {component_type},
                                            component_type));
      return EmitComposeComplex(op, ir_builder_->CreateFMul(exp_a, cos_b),
                                ir_builder_->CreateFMul(exp_a, sin_b));
    }
    case HloOpcode::kCos: {
      // cos(a+bi) = .5(cos(a)*(e^-b+e^b) + i*sin(a)*(e^-b-e^b))
      auto a = EmitExtractReal(operand_value);
      auto llvm_ty = a->getType();
      TF_ASSIGN_OR_RETURN(
          auto exp_b,
          EmitLibdeviceMathCall("__nv_exp", {EmitExtractImag(operand_value)},
                                {component_type}, component_type));
      TF_ASSIGN_OR_RETURN(
          auto cos_a, EmitLibdeviceMathCall("__nv_cos", {a}, {component_type},
                                            component_type));
      TF_ASSIGN_OR_RETURN(
          auto sin_a, EmitLibdeviceMathCall("__nv_sin", {a}, {component_type},
                                            component_type));
      auto half_exp_b =
          ir_builder_->CreateFMul(llvm::ConstantFP::get(llvm_ty, 0.5), exp_b);
      auto half_exp_neg_b =
          ir_builder_->CreateFDiv(llvm::ConstantFP::get(llvm_ty, 0.5), exp_b);
      return EmitComposeComplex(
          op,
          ir_builder_->CreateFMul(
              cos_a, ir_builder_->CreateFAdd(half_exp_neg_b, half_exp_b)),
          ir_builder_->CreateFMul(
              sin_a, ir_builder_->CreateFSub(half_exp_neg_b, half_exp_b)));
    }

    case HloOpcode::kSin: {
      // sin(a+bi) = 0.5(sin(a)*(e^b+e^-b) + i*cos(a)*(e^b-e^-b)
      auto a = EmitExtractReal(operand_value);
      auto llvm_ty = a->getType();
      TF_ASSIGN_OR_RETURN(
          auto exp_b,
          EmitLibdeviceMathCall("__nv_exp", {EmitExtractImag(operand_value)},
                                {component_type}, component_type));
      TF_ASSIGN_OR_RETURN(
          auto cos_a, EmitLibdeviceMathCall("__nv_cos", {a}, {component_type},
                                            component_type));
      TF_ASSIGN_OR_RETURN(
          auto sin_a, EmitLibdeviceMathCall("__nv_sin", {a}, {component_type},
                                            component_type));
      auto half_exp_b =
          ir_builder_->CreateFMul(llvm::ConstantFP::get(llvm_ty, 0.5), exp_b);
      auto half_exp_neg_b =
          ir_builder_->CreateFDiv(llvm::ConstantFP::get(llvm_ty, 0.5), exp_b);
      return EmitComposeComplex(
          op,
          ir_builder_->CreateFMul(
              sin_a, ir_builder_->CreateFAdd(half_exp_b, half_exp_neg_b)),
          ir_builder_->CreateFMul(
              cos_a, ir_builder_->CreateFSub(half_exp_b, half_exp_neg_b)));
    }
    case HloOpcode::kTanh: {
      /*
      tanh=(exp(x)-exp(-x)) / (exp(x)+exp(-x))
      e^(a+bi) = e^a*(cos(b)+sin(b)i)
      so tanh=(((cos(b)+sin(b)i)e^a - (cos(-b)+sin(-b)i)e^-a)) /
              (((cos(b)+sin(b)i)e^a + (cos(-b)+sin(-b)i)e^-a))
      cos(b)=cos(-b), sin(-b)=-sin(b)
      so tanh=(((cos(b)+sin(b)i)e^a - (cos(b)-sin(b)i)e^-a)) /
              (((cos(b)+sin(b)i)e^a + (cos(b)-sin(b)i)e^-a))
             =(cos(b)e^a+i*sin(b)e^a + cos(b)(-e^-a)+i*sin(b)e^-a) /
              (cos(b)e^a+i*sin(b)e^a + cos(b)e^-a+i*sin(b)(-e^-a))
             =(cos(b)(e^a-e^-a) + i*sin(b)(e^a+e^-a)) /
              (cos(b)(e^a+e^-a) + i*sin(b)(e^a-e^-a))
      This is a complex division, so we can multiply by denom_conj/denom_conj
             =(cos(b)(e^a-e^-a) + i*sin(b)(e^a+e^-a)) *
              (cos(b)(e^a+e^-a) - i*sin(b)(e^a-e^-a)) /
              ((cos(b)(e^a+e^-a))^2 + (sin(b)(e^a-e^-a))^2)
             =(cos(b)^2(e^(2a)-e^(-2a)) + sin(b)^2(e^(2a)-e^(-2a)) +
               i*(cos(b)sin(b)(e^a+e^-a)^2 - cos(b)sin(b)(e^a-e^-a)^2)) /
              ((cos(b)(e^a+e^-a))^2 + (sin(b)(e^a-e^-a))^2)
      */
      auto a = EmitExtractReal(operand_value);
      auto b = EmitExtractImag(operand_value);
      TF_ASSIGN_OR_RETURN(
          auto exp_a, EmitLibdeviceMathCall("__nv_exp", {a}, {component_type},
                                            component_type));
      TF_ASSIGN_OR_RETURN(
          auto cos_b, EmitLibdeviceMathCall("__nv_cos", {b}, {component_type},
                                            component_type));
      TF_ASSIGN_OR_RETURN(
          auto sin_b, EmitLibdeviceMathCall("__nv_sin", {b}, {component_type},
                                            component_type));
      auto exp_neg_a = ir_builder_->CreateFDiv(
          llvm::ConstantFP::get(exp_a->getType(), 1), exp_a);
      auto exp_2a_minus_exp_neg_2a = ir_builder_->CreateFSub(
          ir_builder_->CreateFMul(exp_a, exp_a),
          ir_builder_->CreateFMul(exp_neg_a, exp_neg_a));
      auto cos_b_sq = ir_builder_->CreateFMul(cos_b, cos_b);
      auto sin_b_sq = ir_builder_->CreateFMul(sin_b, sin_b);
      auto real_num = ir_builder_->CreateFAdd(
          ir_builder_->CreateFMul(cos_b_sq, exp_2a_minus_exp_neg_2a),
          ir_builder_->CreateFMul(sin_b_sq, exp_2a_minus_exp_neg_2a));
      auto cos_b_sin_b = ir_builder_->CreateFMul(cos_b, sin_b);
      auto exp_a_plus_exp_neg_a = ir_builder_->CreateFAdd(exp_a, exp_neg_a);
      auto exp_a_plus_exp_neg_a_sq =
          ir_builder_->CreateFMul(exp_a_plus_exp_neg_a, exp_a_plus_exp_neg_a);
      auto exp_a_minus_exp_neg_a = ir_builder_->CreateFSub(exp_a, exp_neg_a);
      auto exp_a_minus_exp_neg_a_sq =
          ir_builder_->CreateFMul(exp_a_minus_exp_neg_a, exp_a_minus_exp_neg_a);
      auto imag_num = ir_builder_->CreateFMul(
          cos_b_sin_b, ir_builder_->CreateFSub(exp_a_plus_exp_neg_a_sq,
                                               exp_a_minus_exp_neg_a_sq));
      auto denom = ir_builder_->CreateFAdd(
          ir_builder_->CreateFMul(cos_b_sq, exp_a_plus_exp_neg_a_sq),
          ir_builder_->CreateFMul(sin_b_sq, exp_a_minus_exp_neg_a_sq));
      return EmitComposeComplex(op, ir_builder_->CreateFDiv(real_num, denom),
                                ir_builder_->CreateFDiv(imag_num, denom));
    }
    default:
      return ElementalIrEmitter::EmitComplexUnaryOp(op, operand_value);
  }
}

llvm::Value* GpuElementalIrEmitter::EmitDeviceFunctionCall(
    const string& callee_name,
    tensorflow::gtl::ArraySlice<llvm::Value*> operands,
    tensorflow::gtl::ArraySlice<PrimitiveType> input_types,
    PrimitiveType output_type,
    tensorflow::gtl::ArraySlice<llvm::Attribute::AttrKind> attributes) const {
  std::vector<llvm::Type*> ir_input_types;
  for (PrimitiveType input_type : input_types) {
    ir_input_types.push_back(
        llvm_ir::PrimitiveTypeToIrType(input_type, module_));
  }
  llvm::FunctionType* callee_type = llvm::FunctionType::get(
      llvm_ir::PrimitiveTypeToIrType(output_type, module_),  // Return type.
      ir_input_types,                                        // Parameter types.
      false);  // No variadic arguments.

  // Declares the callee if it is not declared already.
  llvm::Function* callee = llvm::cast<llvm::Function>(
      ir_builder_->GetInsertBlock()->getModule()->getOrInsertFunction(
          llvm_ir::AsStringRef(callee_name), callee_type));

  for (auto attribute : attributes) {
    callee->addFnAttr(attribute);
  }

  return ir_builder_->CreateCall(callee, llvm_ir::AsArrayRef(operands));
}

llvm::Value* GpuElementalIrEmitter::EmitThreadId() const {
  llvm::Value* block_id = ir_builder_->CreateIntCast(
      llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x,
                                   {}, {}, ir_builder_),
      ir_builder_->getIntNTy(128), /*isSigned=*/true, "block.id");
  llvm::Value* thread_id_in_block = ir_builder_->CreateIntCast(
      llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x,
                                   {}, {}, ir_builder_),
      ir_builder_->getIntNTy(128), /*isSigned=*/true, "thread.id");
  llvm::Value* threads_per_block = ir_builder_->CreateIntCast(
      llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x,
                                   {}, {}, ir_builder_),
      ir_builder_->getIntNTy(128), /*isSigned=*/true, "threads_per_block");
  return ir_builder_->CreateNSWAdd(
      ir_builder_->CreateNSWMul(block_id, threads_per_block),
      thread_id_in_block);
}

llvm_ir::ElementGenerator GpuElementalIrEmitter::MakeElementGenerator(
    const HloInstruction* hlo,
    const HloToElementGeneratorMap& operand_to_generator) const {
  switch (hlo->opcode()) {
    case HloOpcode::kMap:
      return [=, &operand_to_generator](
                 const IrArray::Index& index) -> StatusOr<llvm::Value*> {
        TF_RET_CHECK(!hlo->operands().empty())
            << "Zero operand map not implemented in GPU backend.";
        TF_RET_CHECK(hlo->to_apply()->num_parameters() > 0);
        std::vector<llvm::Value*> operand_elements;
        for (HloInstruction* operand : hlo->operands()) {
          TF_ASSIGN_OR_RETURN(llvm::Value * value,
                              operand_to_generator.at(operand)(index));
          operand_elements.push_back(value);
        }
        return compute_nested_(*hlo->to_apply(), operand_elements);
      };
    case HloOpcode::kReduceWindow:
      // Pseudocode:
      // for each index I in output
      //   value = init_value
      //   for each index W in window
      //     for each dimension i from 0 to rank - 1
      //       (input index I)[i] = O[i] * stride[i] + W[i] - pad_low[i]
      //     if I in bounds of input
      //       value = function(value, input[I])
      //     output[O] = value
      return [=, &operand_to_generator](
                 const IrArray::Index& index) -> StatusOr<llvm::Value*> {
        const HloInstruction* operand = hlo->operand(0);
        const Window& window = hlo->window();

        // TODO(b/31410564): Implement dilation for reduce-window.
        if (window_util::HasDilation(window)) {
          return Unimplemented(
              "Dilation for reduce-window not implemented on GPU. "
              "See b/31410564.");
        }

        PrimitiveType operand_element_type = operand->shape().element_type();
        llvm::Value* accum_ptr = llvm_ir::EmitAllocaAtFunctionEntry(
            llvm_ir::PrimitiveTypeToIrType(operand_element_type, module_),
            "reduce_window_accum_ptr", ir_builder_);
        {
          TF_ASSIGN_OR_RETURN(llvm::Value * init_value,
                              operand_to_generator.at(hlo->operand(1))({}));
          ir_builder_->CreateStore(init_value, accum_ptr);
        }

        llvm_ir::ForLoopNest loops(IrName(hlo), ir_builder_);
        std::vector<int64> window_size;
        for (const auto& dim : window.dimensions()) {
          window_size.push_back(dim.size());
        }
        const IrArray::Index window_index = loops.AddLoopsForShape(
            ShapeUtil::MakeShape(operand_element_type, window_size), "window");
        CHECK_EQ(window_index.size(), index.size());

        SetToFirstInsertPoint(loops.GetInnerLoopBodyBasicBlock(), ir_builder_);

        IrArray::Index input_index(index.size());
        llvm::Value* in_bounds = ir_builder_->getInt1(true);
        for (size_t i = 0; i < index.size(); ++i) {
          llvm::Value* stridden_index = ir_builder_->CreateNSWMul(
              index[i], ir_builder_->getInt64(window.dimensions(i).stride()));
          input_index[i] = ir_builder_->CreateNSWSub(
              ir_builder_->CreateNSWAdd(stridden_index, window_index[i]),
              ir_builder_->getInt64(window.dimensions(i).padding_low()));

          // We must check whether 0 ≤ input_index[i] < bound, as otherwise
          // we are in the pad and so can skip the computation. This
          // comparison is equivalent to the unsigned comparison
          // input_index[i] < bound, as a negative value wraps to a large
          // positive value.
          in_bounds = ir_builder_->CreateAnd(
              in_bounds,
              ir_builder_->CreateICmpULT(
                  input_index[i],
                  ir_builder_->getInt64(operand->shape().dimensions(i))));
        }

        llvm_ir::LlvmIfData if_data =
            llvm_ir::EmitIfThenElse(in_bounds, "in_bounds", ir_builder_);
        SetToFirstInsertPoint(if_data.true_block, ir_builder_);

        // We are not in pad, so do the computation.
        TF_ASSIGN_OR_RETURN(llvm::Value * input_value,
                            operand_to_generator.at(operand)(input_index));
        TF_ASSIGN_OR_RETURN(
            llvm::Value * accum_value,
            compute_nested_(*hlo->to_apply(),
                            {ir_builder_->CreateLoad(accum_ptr), input_value}));
        ir_builder_->CreateStore(accum_value, accum_ptr);

        SetToFirstInsertPoint(loops.GetOuterLoopExitBasicBlock(), ir_builder_);
        return ir_builder_->CreateLoad(accum_ptr);
      };
    case HloOpcode::kReduce:
      return [=, &operand_to_generator](
                 const IrArray::Index& output_index) -> StatusOr<llvm::Value*> {
        const HloInstruction* operand = hlo->operand(0);
        llvm::Value* accum_ptr =
            ir_builder()->CreateAlloca(llvm_ir::PrimitiveTypeToIrType(
                hlo->shape().element_type(), module_));
        TF_ASSIGN_OR_RETURN(llvm::Value * init_value,
                            operand_to_generator.at(hlo->operand(1))({}));
        ir_builder()->CreateStore(init_value, accum_ptr);

        llvm_ir::ForLoopNest loops(IrName(hlo), ir_builder_);
        IrArray::Index input_index = loops.AddLoopsForShapeOnDimensions(
            operand->shape(), hlo->dimensions(), "reduction_dim");
        if (!ShapeUtil::IsScalar(hlo->shape())) {
          // Here only input_index[hlo->dimensions()] are non-null, so we must
          // set the rest.
          size_t j = 0;
          for (size_t i = 0; i < input_index.size(); ++i) {
            if (input_index[i] == nullptr) {
              input_index[i] = output_index[j++];
            }
          }
          CHECK_EQ(output_index.size(), j);
        }

        SetToFirstInsertPoint(loops.GetInnerLoopBodyBasicBlock(), ir_builder());
        TF_ASSIGN_OR_RETURN(
            llvm::Value * input_value,
            operand_to_generator.at(hlo->operand(0))(input_index));
        TF_ASSIGN_OR_RETURN(
            llvm::Value * accum_value,
            compute_nested_(
                *hlo->to_apply(),
                {ir_builder()->CreateLoad(accum_ptr), input_value}));
        ir_builder()->CreateStore(accum_value, accum_ptr);
        SetToFirstInsertPoint(loops.GetOuterLoopExitBasicBlock(), ir_builder());
        return ir_builder()->CreateLoad(accum_ptr);
      };
    default:
      return ElementalIrEmitter::MakeElementGenerator(hlo,
                                                      operand_to_generator);
  }
}

}  // namespace gpu
}  // namespace xla
