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

#include <initializer_list>
#include <memory>
#include <string>

#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_computation.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class CompilationCacheTest : public ClientLibraryTestBase {
 public:
  void ExecuteComputationR0F32(
      const XlaComputation& computation,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments, float expected_result,
      bool expect_cache_hit) {
    ExecutionProfile execution_profile;
    std::unique_ptr<Literal> result =
        client_
            ->ExecuteAndTransfer(computation, arguments,
                                 /*execution_options=*/&execution_options_,
                                 &execution_profile)
            .ConsumeValueOrDie();
    EXPECT_TRUE(LiteralTestUtil::Near(
        *Literal::CreateR0<float>(expected_result), *result, error_spec_));
    EXPECT_EQ(expect_cache_hit, execution_profile.compilation_cache_hit());
  }

  void ExecuteComputationR2F32(
      const XlaComputation& computation,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments,
      std::initializer_list<std::initializer_list<float>> expected_result,
      bool expect_cache_hit) {
    ExecutionProfile execution_profile;
    auto data_handle = client_
                           ->Execute(computation, arguments,
                                     &execution_options_, &execution_profile)
                           .ConsumeValueOrDie();
    std::unique_ptr<Literal> result =
        client_->Transfer(*data_handle).ConsumeValueOrDie();
    EXPECT_TRUE(LiteralTestUtil::Near(
        *Literal::CreateR2<float>(expected_result), *result, error_spec_));
    EXPECT_EQ(expect_cache_hit, execution_profile.compilation_cache_hit());
  }

  ErrorSpec error_spec_{0.0001};
};

// TODO(b/74197823): Disabled because there is no cache in the new design.
XLA_TEST_F(CompilationCacheTest, DISABLED_ComputationCalledMultipleTimes) {
  XlaBuilder builder(TestName());
  builder.Neg(builder.ConstantR0<float>(42.0));
  XlaComputation computation = builder.Build().ConsumeValueOrDie();

  ExecuteComputationR0F32(computation, {}, -42.0, /*expect_cache_hit=*/false);
  ExecuteComputationR0F32(computation, {}, -42.0, /*expect_cache_hit=*/true);
  ExecuteComputationR0F32(computation, {}, -42.0, /*expect_cache_hit=*/true);
}

// TODO(b/74197823): Disabled because there is no cache in the new design.
XLA_TEST_F(CompilationCacheTest,
           DISABLED_ComputationCalledWithDifferentParameters) {
  std::unique_ptr<GlobalData> data_42 =
      client_->TransferToServer(*Literal::CreateR0<float>(42.0f))
          .ConsumeValueOrDie();
  std::unique_ptr<GlobalData> data_123 =
      client_->TransferToServer(*Literal::CreateR0<float>(123.0f))
          .ConsumeValueOrDie();
  std::unique_ptr<GlobalData> data_456 =
      client_->TransferToServer(*Literal::CreateR0<float>(456.0f))
          .ConsumeValueOrDie();

  XlaBuilder builder(TestName());
  builder.Neg(builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "param"));
  XlaComputation computation = builder.Build().ConsumeValueOrDie();

  ExecuteComputationR0F32(computation, {data_42.get()}, -42.0,
                          /*expect_cache_hit=*/false);
  ExecuteComputationR0F32(computation, {data_123.get()}, -123.0,
                          /*expect_cache_hit=*/true);
  ExecuteComputationR0F32(computation, {data_456.get()}, -456.0,
                          /*expect_cache_hit=*/true);
  ExecuteComputationR0F32(computation, {data_42.get()}, -42.0,
                          /*expect_cache_hit=*/true);
}

// TODO(b/74197823): Disabled because there is no cache in the new design.
XLA_TEST_F(CompilationCacheTest, DISABLED_MultipleComputations) {
  XlaBuilder builder_neg(TestName() + "_neg");
  builder_neg.Neg(builder_neg.ConstantR0<float>(42.0));
  XlaComputation computation_neg = builder_neg.Build().ConsumeValueOrDie();

  XlaBuilder builder_exp(TestName() + "_exp");
  builder_exp.Exp(builder_exp.ConstantR0<float>(1.0));
  XlaComputation computation_exp = builder_exp.Build().ConsumeValueOrDie();

  XlaBuilder builder_add(TestName() + "_add");
  builder_add.Add(builder_add.ConstantR0<float>(2.0),
                  builder_add.ConstantR0<float>(3.0));
  XlaComputation computation_add = builder_add.Build().ConsumeValueOrDie();

  ExecuteComputationR0F32(computation_neg, {}, -42.0,
                          /*expect_cache_hit=*/false);
  ExecuteComputationR0F32(computation_exp, {}, 2.7182817,
                          /*expect_cache_hit=*/false);
  ExecuteComputationR0F32(computation_add, {}, 5.0,
                          /*expect_cache_hit=*/false);
  ExecuteComputationR0F32(computation_neg, {}, -42.0,
                          /*expect_cache_hit=*/true);
}

// TODO(b/74197823): Disabled because there is no cache in the new design.
XLA_TEST_F(CompilationCacheTest, DISABLED_DifferentParameterLayouts) {
  // Create two GlobalData arrays with the same shape but different
  // layouts. Use these arrays as parameters to a simple computation. If the
  // layout of the array changes then computation should be recompiled (cache
  // miss).
  auto rowmaj_array = Literal::CreateR2WithLayout(
      {{1.0f, 2.0f}, {3.0f, 4.0f}}, LayoutUtil::MakeLayout({1, 0}));
  auto rowmaj_handle =
      client_->TransferToServer(*rowmaj_array).ConsumeValueOrDie();

  auto colmaj_array = Literal::CreateR2WithLayout(
      {{1.0f, 2.0f}, {3.0f, 4.0f}}, LayoutUtil::MakeLayout({0, 1}));
  auto colmaj_handle =
      client_->TransferToServer(*colmaj_array).ConsumeValueOrDie();

  XlaBuilder builder(TestName());
  builder.Parameter(0, ShapeUtil::MakeShape(F32, {2, 2}), "param0");
  XlaComputation computation = builder.Build().ConsumeValueOrDie();

  ExecuteComputationR2F32(computation, {colmaj_handle.get()},
                          {{1.0f, 2.0f}, {3.0f, 4.0f}},
                          /*expect_cache_hit=*/false);
  ExecuteComputationR2F32(computation, {colmaj_handle.get()},
                          {{1.0f, 2.0f}, {3.0f, 4.0f}},
                          /*expect_cache_hit=*/true);
  ExecuteComputationR2F32(computation, {rowmaj_handle.get()},
                          {{1.0f, 2.0f}, {3.0f, 4.0f}},
                          /*expect_cache_hit=*/false);
  ExecuteComputationR2F32(computation, {rowmaj_handle.get()},
                          {{1.0f, 2.0f}, {3.0f, 4.0f}},
                          /*expect_cache_hit=*/true);
  ExecuteComputationR2F32(computation, {colmaj_handle.get()},
                          {{1.0f, 2.0f}, {3.0f, 4.0f}},
                          /*expect_cache_hit=*/true);
}

}  // namespace
}  // namespace xla
