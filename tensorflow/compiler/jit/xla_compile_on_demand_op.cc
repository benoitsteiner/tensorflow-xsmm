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

// Defines the XlaCompileOnDemandOp.

#include "tensorflow/compiler/jit/xla_compile_on_demand_op.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"

namespace tensorflow {

namespace {
std::map<int, OptionalTensor> GetVariables(OpKernelContext* ctx) {
  std::map<int, OptionalTensor> variables;
  for (int64 i = 0; i < ctx->num_inputs(); ++i) {
    if (ctx->input(i).dtype() == DT_RESOURCE) {
      Var* variable = nullptr;
      ResourceHandle handle = HandleFromInput(ctx, i);
      OptionalTensor& optional = variables[i];
      optional.name = handle.name();
      if (LookupResource(ctx, handle, &variable).ok()) {
        tf_shared_lock lock(*variable->mu());
        optional.present = true;
        optional.value = *variable->tensor();
      }
    }
  }
  return variables;
}
}  // namespace

Status XlaCompileOnDemandOp::Run(OpKernelContext* ctx,
                                 const XlaDevice::Metadata& metadata,
                                 const XlaCompiler::CompilationResult* result,
                                 xla::LocalExecutable* executable) {
  std::map<int, OptionalTensor> variables = GetVariables(ctx);
  int64 num_resource_args = variables.size();

  xla::LocalClient* client = metadata.client();
  XlaTensorInfoManager* tensor_info_manager = &metadata.tensor_info_manager();

  // Builds an XLA allocator for the device.
  XlaAllocator xla_allocator(client->platform(), ctx);
  XlaComputationLaunchContext launch_context(
      num_resource_args, client, &xla_allocator, tensor_info_manager);

  launch_context.PopulateInputs(ctx, result, variables);

  perftools::gputools::Stream* stream =
      ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr;
  TF_RET_CHECK(stream);

  VLOG(2) << "Executing computation.";
  xla::ExecutableRunOptions run_options;
  run_options.set_stream(stream);
  run_options.set_allocator(&xla_allocator);
  run_options.set_intra_op_thread_pool(&ctx->eigen_cpu_device());

  auto run_result = executable->Run(launch_context.arguments(), run_options);
  TF_RETURN_IF_ERROR(run_result.status());

  launch_context.PopulateOutputs(ctx, result, run_result.ConsumeValueOrDie());
  return Status::OK();
}

bool XlaCompileOnDemandOp::MustArgumentBeConstant(const OpKernel* op_kernel,
                                                  int64 argument_idx) {
  // TODO(jmolloy): This could be expensive, so memoize.
  auto* constant_inputs = tensorflow::XlaOpRegistry::CompileTimeConstantInputs(
      op_kernel->def().op());
  CHECK(constant_inputs);
  std::set<int64> constant_input_indices;
  for (const auto& name : *constant_inputs) {
    int start, stop;
    TF_CHECK_OK(op_kernel->InputRange(name, &start, &stop));
    for (int i = start; i < stop; ++i) {
      constant_input_indices.insert(i);
    }
  }
  return constant_input_indices.count(argument_idx) > 0;
}

bool XlaCompileOnDemandOp::ShouldArgumentBeConstant(const OpKernel* op_kernel,
                                                    int64 argument_idx) {
  // Right now we only create kConstant arguments when absolutely required, but
  // there may be benefit in eagerly constant-folding a larger subset of
  // arguments in the future.
  return MustArgumentBeConstant(op_kernel, argument_idx);
}

Status XlaCompileOnDemandOp::Compile(
    OpKernelContext* ctx, const XlaDevice::Metadata& metadata,
    const XlaCompiler::CompilationResult** result,
    xla::LocalExecutable** executable) {
  XlaTensorInfoManager* tensor_info_manager = &metadata.tensor_info_manager();

  std::map<int, Tensor> constant_arguments;
  for (int64 i = 0; i < ctx->num_inputs(); ++i) {
    const Tensor& device_tensor = ctx->input(i);
    if (const XlaTensorInfo* tensor_info =
            tensor_info_manager->GetTensorInfo(device_tensor)) {
      if (tensor_info->has_host_tensor() &&
          ShouldArgumentBeConstant(&ctx->op_kernel(), i)) {
        constant_arguments[i] = tensor_info->host_tensor();
      }
    }
    if (constant_arguments.count(i) == 0 &&
        MustArgumentBeConstant(&ctx->op_kernel(), i)) {
      // Slow path; the argument is not available as a host constant so we must
      // fetch it synchronously.
      Tensor host_tensor;
      TF_RETURN_IF_ERROR(ctx->allocate_temp(
          device_tensor.dtype(), device_tensor.shape(), &host_tensor));
      Notification n;
      ctx->op_device_context()->CopyDeviceTensorToCPU(
          &device_tensor, "ConstantArgument",
          reinterpret_cast<Device*>(ctx->device()), &host_tensor,
          [&](Status status) { n.Notify(); });
      n.WaitForNotification();
      constant_arguments[i] = host_tensor;
    }
  }

  // We store information about the JIT-compiled XLA computation
  // in the ResourceMgr.
  ResourceMgr* rm = ctx->resource_manager();
  CHECK(rm);

  XlaCompilationCache* cache;
  TF_RETURN_IF_ERROR(rm->LookupOrCreate<XlaCompilationCache>(
      rm->default_container(), "xla_cache", &cache,
      [&](XlaCompilationCache** cache) {
        *cache = new XlaCompilationCache(metadata.client(),
                                         metadata.jit_device_type());
        return Status::OK();
      }));
  // Hold the reference to the JIT during evaluation. (We could probably
  // free it sooner because the ResourceMgr will retain a reference, but
  // this is more obviously correct.)
  core::ScopedUnref cache_ref(cache);

  XlaCompiler::Options options;
  DeviceType device_type = metadata.jit_device_type();
  options.device_type = &device_type;
  options.client = metadata.client();
  options.flib_def =
      new FunctionLibraryDefinition(OpRegistry::Global(), FunctionDefLibrary{});

  std::map<int, OptionalTensor> variable_args = GetVariables(ctx);
  return cache->CompileSingleOp(options, constant_arguments, variable_args, ctx,
                                result, executable,
                                /*compile_options=*/nullptr);
}

void XlaCompileOnDemandOp::Compute(OpKernelContext* ctx) {
  const XlaCompiler::CompilationResult* result;
  xla::LocalExecutable* executable;
  const XlaDevice::Metadata* metadata;
  OP_REQUIRES_OK(ctx, XlaDevice::GetMetadata(ctx, &metadata));
  OP_REQUIRES_OK(ctx, Compile(ctx, *metadata, &result, &executable));
  OP_REQUIRES_OK(ctx, Run(ctx, *metadata, result, executable));
}

}  // namespace tensorflow
