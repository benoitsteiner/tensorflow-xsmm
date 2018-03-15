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
#ifndef TENSORFLOW_C_EAGER_C_API_INTERNAL_H_
#define TENSORFLOW_C_EAGER_C_API_INTERNAL_H_

#include "tensorflow/c/eager/c_api.h"

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/runtime.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/version.h"

// A unit of execution for the TFE_Executor class below. Example subclasses
// encapsulate execution of a TFE_Op, or copying a TFE_TensorHandle from one
// device to another.
class TFE_Node {
 public:
  explicit TFE_Node(tensorflow::uint64 id);

  virtual ~TFE_Node() {}

  // Runs the computation corresponding to this node and blocks till the
  // execution is done.
  virtual tensorflow::Status Run() = 0;

  // An id unique to the TFE_Context under which this node is created. Allocated
  // monotonically.
  const tensorflow::uint64 id;
};

// A class for handling async execution (see TFE_ContextSetAsync).
// Note that this class is thread-safe.
// TODO(agarwal): TFE_OpAddInput may currently block if it tries to access the
// device of the input handle. Fix that.
// TODO(agarwal): On error, mark all affected handles as corrupted.
// TODO(agarwal): Implement support for control dependencies.
// TODO(agarwal): Support out-of-order execution and dispatching multiple
// TFE_Node in parallel.
// TODO(agarwal): Implement optimizations over TFE_Node traces.
class TFE_Executor {
 public:
  ~TFE_Executor();

  // This is called whenever async mode is enabled. Note that it may be called
  // multiple times as different calling threads may switch async mode on or off
  // independently.
  void EnableAsync();

  // Helper function to create monotonically increasing ids unique to this
  // object.
  tensorflow::uint64 NextId();

  // Schedules `node` for execution.
  // Note that Add must be called in monotonically increasing order of node->id.
  void Add(TFE_Node* node);

  // Causes the caller to block till node with id `node_id` has finished
  // execution.
  tensorflow::Status WaitFor(tensorflow::uint64 node_id);

  // Blocks till all currently pending ops are done.
  tensorflow::Status WaitForAllPendingNodes();

  // Clears all currently set errors which re-enables async execution.
  void ClearError();

  // Returns Status based on any errors that occurred during async execution.
  tensorflow::Status status();

 private:
  // Starts execution of pending TFE_Nodes. This function loops till
  // thread_done_ is set to true. If any errors are encontered, these are set
  // inside `status_`. The loop blocks anytime there are no pending nodes, or if
  // `status_` is not ok.
  void Run();

  tensorflow::Status WaitImpl(bool wait_all, tensorflow::uint64 node_id);

  tensorflow::mutex node_queue_mutex_;

  // Used to signal that some TFE_Nodes are pending execution.
  tensorflow::condition_variable nodes_pending_ GUARDED_BY(node_queue_mutex_);

  // Queue of pending TFE_Nodes.
  std::queue<TFE_Node*> node_queue_ GUARDED_BY(node_queue_mutex_);

  // `status_` is set based on any errors raised during execution of a TFE_Node.
  // It remains set until ClearError is called.
  tensorflow::Status status_ GUARDED_BY(node_queue_mutex_);

  // Map from id of a TFE_Node to condition_variables (not owned by the map).
  // These condition_variables are notified and removed when that TFE_Node is
  // done executing, or if an error is found in execution of any TFE_Node.
  std::multimap<tensorflow::uint64, tensorflow::condition_variable*>
      node_done_notifications_ GUARDED_BY(node_queue_mutex_);

  // Thread object that calls the `Run` method. Currently we use only one thread
  // for executing the TFE_Nodes one-by-one.
  std::unique_ptr<tensorflow::Thread> thread_ GUARDED_BY(node_queue_mutex_);

  // Indicates that `thread_` should stop as soon as it is done executing the
  // current TFE_Node.
  bool thread_done_ GUARDED_BY(node_queue_mutex_) = false;

  tensorflow::mutex next_id_mutex_;
  tensorflow::uint64 next_id_ GUARDED_BY(next_id_mutex_) = 1;
};

struct TFE_ContextOptions {
  TF_SessionOptions session_options;
  // true if async execution is enabled.
  bool async = false;
  TFE_ContextDevicePlacementPolicy policy{
      TFE_DEVICE_PLACEMENT_SILENT_FOR_INT32};
};

TFE_ContextDevicePlacementPolicy PlacementPolicy(
    bool soft_placement, TFE_ContextDevicePlacementPolicy original_policy);

struct TFE_Context {
  explicit TFE_Context(const TFE_ContextOptions& opts,
                       std::unique_ptr<tensorflow::DeviceMgr> device_mgr,
                       tensorflow::Rendezvous* rendezvous)
      : soft_placement(
            opts.session_options.options.config.allow_soft_placement()),
        policy(PlacementPolicy(soft_placement, opts.policy)),
        device_manager(std::move(device_mgr)),
        devices(device_manager->ListDevices()),
        rendezvous(rendezvous),
        pflr(new tensorflow::ProcessFunctionLibraryRuntime(
            device_manager.get(), opts.session_options.options.env,
            TF_GRAPH_DEF_VERSION, &func_lib_def, {})),
        log_device_placement(
            opts.session_options.options.config.log_device_placement()),
        async_default(opts.async) {
    if (async_default) executor.EnableAsync();
  }

  const bool soft_placement;
  const TFE_ContextDevicePlacementPolicy policy;

  // Note: we cannot use C++11 thread_local here as there is no concept of a
  // thread-local-object-local variable in C++11.
  tensorflow::mutex policy_map_mu;
  std::unordered_map<std::thread::id, TFE_ContextDevicePlacementPolicy>
      thread_local_policies GUARDED_BY(policy_map_mu);

  std::unique_ptr<tensorflow::DeviceMgr> device_manager;
  // Devices owned by device_manager
  const std::vector<tensorflow::Device*> devices;
  tensorflow::Rendezvous* const rendezvous;

  tensorflow::mutex functions_mu;
  tensorflow::FunctionLibraryDefinition func_lib_def GUARDED_BY(functions_mu){
      tensorflow::OpRegistry::Global(), {}};

  // One FunctionLibraryRuntime per device.
  // func_libs[i] is the FunctionLibraryRuntime corresponding to
  // session->devices[i].
  const std::unique_ptr<tensorflow::ProcessFunctionLibraryRuntime> pflr;

  tensorflow::mutex cache_mu;
  std::unordered_map<tensorflow::Fprint128, tensorflow::KernelAndDevice*,
                     tensorflow::Fprint128Hasher>
      kernel_cache GUARDED_BY(cache_mu);

  tensorflow::FunctionLibraryRuntime* func_lib(tensorflow::Device* d) const {
    return pflr->GetFLR(d->name());
  }

  // Whether we should compute RunMetadata.
  std::atomic<bool> should_store_metadata{false};
  tensorflow::mutex metadata_mu;
  tensorflow::RunMetadata run_metadata GUARDED_BY(metadata_mu);
  const bool log_device_placement;
  // TFE_Executor for async execution.
  TFE_Executor executor;

  // True if running in asynchronous mode.
  bool Async() const;

  // True if the default value for execution mode is async. Note that this value
  // can be overridden per thread based on `thread_local_async` overrides.
  const bool async_default;
  mutable tensorflow::mutex async_map_mu;
  std::unordered_map<std::thread::id, bool> thread_local_async
      GUARDED_BY(async_map_mu);
};

struct TFE_TensorHandle : public tensorflow::core::RefCounted {
 public:
  TFE_TensorHandle(const tensorflow::Tensor& t, tensorflow::Device* d,
                   tensorflow::Device* op_device)
      : dtype(t.dtype()),
        node_id(0),
        tensor_(t),
        device_(d),
        op_device_(op_device),
        ctx_(nullptr) {}

  TFE_TensorHandle(tensorflow::uint64 node_id, tensorflow::DataType dtype,
                   TFE_Context* ctx)
      : dtype(dtype),
        node_id(node_id),
        tensor_(dtype),
        device_(nullptr),
        op_device_(nullptr),
        ctx_(ctx) {
    DCHECK_GT(node_id, 0);
  }

  ~TFE_TensorHandle() override {}

  tensorflow::Status Tensor(const tensorflow::Tensor** t);

  tensorflow::Status Device(tensorflow::Device** d);

  tensorflow::Status OpDevice(tensorflow::Device** d);

  tensorflow::Status TensorAndDevice(const tensorflow::Tensor** tensor,
                                     tensorflow::Device** device,
                                     tensorflow::Device** op_device);

  // Note that this can be called at most once, and only on non-ready handles,
  // and makes them ready.
  void SetTensorAndDevice(const tensorflow::Tensor& tensor,
                          tensorflow::Device* device,
                          tensorflow::Device* op_device);

  // dtype for the handle. It must be the same as t.dtype() once the handle is
  // ready.
  const tensorflow::DataType dtype;

 private:
  // If the contents of the Tensor pointed to by this handle is yet to be
  // computed by a TFE_Node, this function will block till that compuatation is
  // done and the handle is "ready".
  tensorflow::Status WaitReady();

  bool IsReady();

  // Id for the TFE_Node that will compute the value pointed to by this handle.
  // If the value is 0, the handle is already ready, but not vice-versa.
  const tensorflow::uint64 node_id;

  tensorflow::Tensor tensor_;

  // TODO(ashankar): device_ == nullptr iff local CPU
  // This was expedient, but perhaps worth revisiting ('device_' should always
  // be a valid pointer?)
  // This can be done if TFE_NewOp() and the TFE_TensorHandle constructors are
  // provided with the appropriate TFE_Context.
  //
  // TODO(ashankar): Reference count TFE_Context to ensure that 'device_' of a
  // TFE_TensorHandle does not outlive the TFE_Context from which it came?
  tensorflow::Device* device_;

  // Device in which the op producing this tensor was executed. Equals to
  // device_ for constant tensors.
  tensorflow::Device* op_device_;

  tensorflow::mutex ctx_mutex_;

  // `ctx` is only guaranteed to be set if the handle is not "ready". This is
  // typically true when the handle was produced during async execution.
  // `ctx` object is not owned and should outlive this handle.
  TFE_Context* ctx_ GUARDED_BY(ctx_mutex_);
};

struct TFE_Op {
  // t is NULL iff the TFE_Op corresponds to a TensorFlow function instead of a
  // primitive operation.
  TFE_Op(TFE_Context* ctx, const char* op, const tensorflow::AttrTypeMap* t)
      : ctx(ctx), name(op), attrs(op), attr_types(t), device(nullptr) {}

  ~TFE_Op();

  bool const is_function() const { return attr_types == nullptr; }

  TFE_Context* ctx;  // Must outlive the TFE_Op.
  const tensorflow::string name;
  tensorflow::AttrBuilder attrs;
  const tensorflow::AttrTypeMap* attr_types;
  tensorflow::gtl::InlinedVector<TFE_TensorHandle*, 4> inputs;
  tensorflow::Device* device;
  bool use_xla = false;
};

namespace tensorflow {
// Set an AttrValue on the op. Doesn't handle the list types.
void SetOpAttrValueScalar(TFE_Context* ctx, TFE_Op* op,
                          const tensorflow::AttrValue& default_value,
                          const char* attr_name, TF_Status* status);
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_C_API_INTERNAL_H_
