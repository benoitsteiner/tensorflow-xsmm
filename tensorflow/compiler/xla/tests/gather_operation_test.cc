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

#include "tensorflow/compiler/xla/execution_options_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tools/parser/hlo_parser.h"

// NB!  TODO(b/74360564): These tests do not test out of bounds behavior since
// that hasn't been specced yet.

namespace xla {
namespace {

using tensorflow::gtl::nullopt;

class GatherOperationTest : public HloTestBase {
 protected:
  void RunTest(const string& hlo_text, Literal* operand,
               Literal* gather_indices) {
    RunTest(hlo_text, {operand, gather_indices});
  }

  void RunTest(const string& hlo_text,
               tensorflow::gtl::ArraySlice<Literal*> args) {
    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsForTest());
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            tools::Parse(hlo_text, config));
    EXPECT_TRUE(RunAndCompare(std::move(module), args, nullopt));
  }
};

XLA_TEST_F(GatherOperationTest, TensorFlowGatherV1) {
  const string hlo_text = R"(
HloModule TensorFlowGatherV1

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[2,3] gather(operand, indices),
      output_window_dims={1},
      elided_window_dims={0},
      gather_dims_to_operand_dims={0},
      index_vector_dim=1,
      window_bounds={1, 3}
}
)";
  std::unique_ptr<Literal> operand =
      Literal::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  std::unique_ptr<Literal> gather_indices = Literal::CreateR1<int32>({0, 2});
  RunTest(hlo_text, operand.get(), gather_indices.get());
}

XLA_TEST_F(GatherOperationTest, TensorFlowGatherV2) {
  const string hlo_text = R"(
HloModule TensorFlowGatherV2

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[3,2] gather(operand, indices),
      output_window_dims={0},
      elided_window_dims={1},
      gather_dims_to_operand_dims={1},
      index_vector_dim=1,
      window_bounds={3, 1}
}
)";
  std::unique_ptr<Literal> operand =
      Literal::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  std::unique_ptr<Literal> gather_indices = Literal::CreateR1<int32>({0, 2});
  RunTest(hlo_text, operand.get(), gather_indices.get());
}

XLA_TEST_F(GatherOperationTest, TensorFlowGatherMultipleBatchDims) {
  const string hlo_text = R"(
HloModule TensorFlowGatherMultipleBatchDims

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,3,2] gather(operand, indices),
      output_window_dims={1},
      elided_window_dims={1},
      gather_dims_to_operand_dims={1},
      index_vector_dim=2,
      window_bounds={3, 1}
}
)";
  std::unique_ptr<Literal> operand =
      Literal::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  std::unique_ptr<Literal> gather_indices =
      Literal::CreateR2<int32>({{0, 2}, {2, 1}});
  RunTest(hlo_text, operand.get(), gather_indices.get());
}

XLA_TEST_F(GatherOperationTest, TensorFlowGatherNdMultipleBatchDims_0) {
  const string hlo_text = R"(
HloModule TensorFlowGatherNdMultipleBatchDims

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2,2] parameter(1)
  ROOT gather = s32[2,2] gather(operand, indices),
      output_window_dims={},
      elided_window_dims={0,1},
      gather_dims_to_operand_dims={0,1},
      index_vector_dim=2,
      window_bounds={1, 1}
}
)";
  std::unique_ptr<Literal> operand =
      Literal::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  std::unique_ptr<Literal> gather_indices =
      Literal::CreateR3<int32>({{{0, 2}, {2, 1}}, {{1, 2}, {2, 0}}});
  RunTest(hlo_text, operand.get(), gather_indices.get());
}

XLA_TEST_F(GatherOperationTest, TensorFlowGatherNdMultipleBatchDims_1) {
  const string hlo_text = R"(
HloModule TensorFlowGatherNdMultipleBatchDims

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2,2] parameter(1)
  ROOT gather = s32[2,1,1,2] gather(operand, indices),
      output_window_dims={1,2},
      elided_window_dims={},
      gather_dims_to_operand_dims={0,1},
      index_vector_dim=2,
      window_bounds={1, 1}
}
)";
  std::unique_ptr<Literal> operand =
      Literal::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  std::unique_ptr<Literal> gather_indices =
      Literal::CreateR3<int32>({{{0, 2}, {2, 1}}, {{1, 2}, {2, 0}}});
  RunTest(hlo_text, operand.get(), gather_indices.get());
}

XLA_TEST_F(GatherOperationTest, TensorFlowGatherNd) {
  const string hlo_text = R"(
HloModule TensorFlowGatherNd

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,2] gather(operand, indices),
      output_window_dims={1},
      elided_window_dims={0,1},
      gather_dims_to_operand_dims={0,1},
      index_vector_dim=1,
      window_bounds={1,1,2}
}
)";
  std::unique_ptr<Literal> operand =
      Literal::CreateR3<int32>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                {{-7, 7}, {-8, 8}, {-9, 9}}});
  std::unique_ptr<Literal> gather_indices =
      Literal::CreateR2<int32>({{0, 0}, {1, 0}});
  RunTest(hlo_text, operand.get(), gather_indices.get());
}

XLA_TEST_F(GatherOperationTest, TensorFlowGatherNdNonDefaultIndexVectorDim) {
  const string hlo_text = R"(
HloModule TensorFlowGatherNd

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,2] gather(operand, indices),
      output_window_dims={1},
      elided_window_dims={0,1},
      gather_dims_to_operand_dims={0,1},
      index_vector_dim=0,
      window_bounds={1,1,2}
}
)";
  std::unique_ptr<Literal> operand =
      Literal::CreateR3<int32>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                {{-7, 7}, {-8, 8}, {-9, 9}}});
  std::unique_ptr<Literal> gather_indices =
      Literal::CreateR2<int32>({{0, 0}, {1, 0}});
  RunTest(hlo_text, operand.get(), gather_indices.get());
}

XLA_TEST_F(GatherOperationTest, DynamicSlice) {
  const char* hlo_text = R"(
HloModule DynamicSlice

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[1,1] gather(operand, indices),
      output_window_dims={0,1},
      elided_window_dims={},
      gather_dims_to_operand_dims={0,1},
      index_vector_dim=0,
      window_bounds={1,1}
}
)";
  std::unique_ptr<Literal> operand =
      Literal::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  std::unique_ptr<Literal> gather_indices = Literal::CreateR1<int32>({1, 1});
  RunTest(hlo_text, operand.get(), gather_indices.get());
}

XLA_TEST_F(GatherOperationTest, BatchDynamicSlice) {
  const string hlo_text = R"(
HloModule BatchDynamicSlice

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,1,1] gather(operand, indices),
      output_window_dims={1,2},
      elided_window_dims={},
      gather_dims_to_operand_dims={0,1},
      index_vector_dim=0,
      window_bounds={1,1}
}
)";
  std::unique_ptr<Literal> operand =
      Literal::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  std::unique_ptr<Literal> gather_indices =
      Literal::CreateR2<int32>({{2, 1}, {1, 1}});
  RunTest(hlo_text, operand.get(), gather_indices.get());
}

XLA_TEST_F(GatherOperationTest, ZeroDimBounds) {
  const char* hlo_text = R"(
HloModule TensorFlowGatherV1

ENTRY main {
  operand = s32[3,0] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[2,0] gather(operand, indices),
      output_window_dims={1},
      elided_window_dims={0},
      gather_dims_to_operand_dims={0},
      index_vector_dim=1,
      window_bounds={1, 0}
}
)";
  std::unique_ptr<Literal> operand = Literal::CreateR2<int32>({{}, {}, {}});
  std::unique_ptr<Literal> gather_indices = Literal::CreateR1<int32>({0, 2});
  RunTest(hlo_text, operand.get(), gather_indices.get());
}

XLA_TEST_F(GatherOperationTest, OutOfBoundsIndex) {
  // Out of bounds indices must not crash, and the indices in range should
  // produce the same values across all backends.
  //
  // TODO(b/74360564): Once we have a well defined semantics for OOB accesses,
  // we should get rid of the mask and check that backends produce the same
  // value for OOB indices too.

  const string hlo_text = R"(
HloModule BatchDynamicSlice

ENTRY main {
  operand = s32[3,3]{1,0} parameter(0)
  indices = s32[6,2]{1,0} parameter(1)
  gather = s32[6,1,1]{2,1,0} gather(operand, indices),
      output_window_dims={1,2},
      elided_window_dims={},
      gather_dims_to_operand_dims={0,1},
      index_vector_dim=1,
      window_bounds={1,1}
  gather_reshaped = s32[6]{0} reshape(gather)
  in_bounds_mask = s32[6]{0} parameter(2)
  ROOT result = s32[6]{0} multiply(gather_reshaped, in_bounds_mask)
}
)";
  std::unique_ptr<Literal> operand =
      Literal::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  std::unique_ptr<Literal> gather_indices = Literal::CreateR2<int32>(
      {{2, 7}, {2, 1}, {1, 1}, {5, 1}, {2147483647, 1}, {1, 2}});
  std::unique_ptr<Literal> in_bounds_mask =
      Literal::CreateR1<int32>({0, 1, 1, 0, 0, 1});

  RunTest(hlo_text,
          {operand.get(), gather_indices.get(), in_bounds_mask.get()});
}

XLA_TEST_F(GatherOperationTest, NegativeIndex) {
  // Negative indices must not crash, and the indices in range should produce
  // the same values across all backends.
  //
  // TODO(b/74360564): Once we have a well defined semantics for negative
  // accesses, we should get rid of the mask and check that backends produce the
  // same value for negative indices too.

  const string hlo_text = R"(
HloModule BatchDynamicSlice

ENTRY main {
  operand = s32[3,3]{1,0} parameter(0)
  indices = s32[6,2]{1,0} parameter(1)
  gather = s32[6,1,1]{2,1,0} gather(operand, indices),
      output_window_dims={1,2},
      elided_window_dims={},
      gather_dims_to_operand_dims={0,1},
      index_vector_dim=1,
      window_bounds={1,1}
  gather_reshaped = s32[6]{0} reshape(gather)
  in_bounds_mask = s32[6]{0} parameter(2)
  ROOT result = s32[6]{0} multiply(gather_reshaped, in_bounds_mask)
}
)";
  std::unique_ptr<Literal> operand =
      Literal::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  std::unique_ptr<Literal> gather_indices = Literal::CreateR2<int32>(
      {{2, -1}, {2, 1}, {1, 1}, {-500, 1}, {-2147483648, 1}, {1, 2}});
  std::unique_ptr<Literal> in_bounds_mask =
      Literal::CreateR1<int32>({0, 1, 1, 0, 0, 1});

  RunTest(hlo_text,
          {operand.get(), gather_indices.get(), in_bounds_mask.get()});
}

XLA_TEST_F(GatherOperationTest, OneScalarIndex) {
  const char* hlo_text = R"(
HloModule OneScalarIndex

ENTRY main {
  operand = s32[2,3,2]{2,1,0} parameter(0)
  index = s32[] parameter(1)
  ROOT gather = s32[1,3,2]{2,1,0} gather(operand, index),
      output_window_dims={0,1,2},
      elided_window_dims={},
      gather_dims_to_operand_dims={0},
      index_vector_dim=0,
      window_bounds={1,3,2}
}
)";
  std::unique_ptr<Literal> operand = Literal::CreateR3<int32>(
      {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}});
  std::unique_ptr<Literal> gather_indices = Literal::CreateR0<int32>(1);
  RunTest(hlo_text, operand.get(), gather_indices.get());
}

XLA_TEST_F(GatherOperationTest, ScalarResult) {
  const char* hlo_text = R"(
HloModule ScalarResult

ENTRY main {
  operand = s32[4]{0} parameter(0)
  index = s32[] parameter(1)
  ROOT gather = s32[] gather(operand, index),
      output_window_dims={},
      elided_window_dims={0},
      gather_dims_to_operand_dims={0},
      index_vector_dim=0,
      window_bounds={1}
}
)";
  std::unique_ptr<Literal> operand = Literal::CreateR1<int32>({1, 2, 3, 4});
  std::unique_ptr<Literal> gather_indices = Literal::CreateR0<int32>(1);
  RunTest(hlo_text, operand.get(), gather_indices.get());
}

XLA_TEST_F(GatherOperationTest, ZeroSizedResult) {
  const string hlo_text = R"(
HloModule ZeroSizedResult

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[0] parameter(1)
  ROOT gather = s32[0,3] gather(operand, indices),
      output_window_dims={1},
      elided_window_dims={0},
      gather_dims_to_operand_dims={0},
      index_vector_dim=1,
      window_bounds={1, 3}
}
)";
  std::unique_ptr<Literal> operand =
      Literal::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  std::unique_ptr<Literal> gather_indices = Literal::CreateR1<int32>({});
  RunTest(hlo_text, operand.get(), gather_indices.get());
}

class GatherClientLibraryTest : public ClientLibraryTestBase {};

// TODO(b/30671675): Asynchronous execution on stream is not yet supported on
// GPU and CPU_PARALLEL.
XLA_TEST_F(GatherClientLibraryTest,
           DISABLED_ON_CPU_PARALLEL(DISABLED_ON_GPU(Basic))) {
  // We create this HLO, but using the XlaBuilder API.
  //
  // ENTRY main {
  //   operand = s32[3,3] parameter(0)
  //   indices = s32[2] parameter(1)
  //   ROOT gather = s32[2,3] gather(operand, indices),
  //       output_window_dims={1},
  //       elided_window_dims={0},
  //       gather_dims_to_operand_dims={0},
  //       index_vector_dim=1,
  //       window_bounds={1, 3}
  // }

  XlaBuilder builder("gather_basic");

  Shape operand_shape = ShapeUtil::MakeShape(S32, {3, 3});
  Shape indices_shape = ShapeUtil::MakeShape(S32, {2});

  auto operand = builder.Parameter(0, operand_shape, "operand");
  auto indices = builder.Parameter(1, indices_shape, "indices");
  GatherDimensionNumbers dim_numbers;
  dim_numbers.add_output_window_dims(1);
  dim_numbers.add_elided_window_dims(0);
  dim_numbers.add_gather_dims_to_operand_dims(0);
  dim_numbers.set_index_vector_dim(1);
  builder.Gather(operand, indices, dim_numbers, {1, 3});

  std::vector<int32> expected = {};
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> operand_arg,
                          client_->TransferToServer(*Literal::CreateR2<int32>(
                              {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}})));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GlobalData> indices_arg,
      client_->TransferToServer(*Literal::CreateR1<int32>({0, 2})));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<xla::DeviceHandle> devices,
                          client_->GetDeviceHandles(1));
  xla::ExecutionOptions execution_options = CreateDefaultExecutionOptions();
  *execution_options.add_device_handles() = devices[0];
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());
  std::vector<xla::Client::XlaComputationInstance> computation_instances = {
      {computation,
       {operand_arg.get(), indices_arg.get()},
       execution_options,
       /*execution_profile=*/nullptr}};
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<xla::GlobalData>> result_data,
      client_->ExecuteParallel(computation_instances));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Literal> result_literal,
                          client_->Transfer(*(result_data[0])));
  LiteralTestUtil::ExpectEqual(
      *result_literal, *Literal::CreateR2<int32>({{1, 2, 3}, {7, 8, 9}}));
}
}  // namespace
}  // namespace xla
