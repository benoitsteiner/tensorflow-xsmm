/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/c_api.h"

#include "tensorflow/c/c_test_util.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

// Specification for expected input/output and its type.
// DataType value of DT_INVALID signifies that we don't want to
// check the data type.
typedef std::pair<string, DataType> IOSpec;

std::vector<IOSpec> M(const std::initializer_list<string>& names) {
  std::vector<IOSpec> v;
  for (const string& name : names) {
    v.push_back(IOSpec(name, DT_INVALID));
  }
  return v;
}

// Specification for an expected edge.
// src is either:
// - input name (as it appears in FunctionDef)
// - name of output tensor (in nested "add:z:0" format)
// dst is either:
// - output name (as it appears in FunctionDef)
// - <name_of_node>:<index_of_this_input_into_node> (this looks the same as
//      output tensor naming, but it the index is actually an input index)
struct EdgeSpec : public std::pair<string, string> {
  typedef std::pair<string, string> Base;

  // Inherit the set of constructors
  using Base::pair;

  string ToString() const { return strings::StrCat(first, "->", second); }
};

class CApiFunctionTest : public ::testing::Test {
 protected:
  CApiFunctionTest()
      : s_(TF_NewStatus()),
        func_graph_(TF_NewGraph()),
        host_graph_(TF_NewGraph()),
        func_(nullptr) {}

  void SetUp() override {}

  ~CApiFunctionTest() override {
    TF_DeleteFunction(func_);
    TF_DeleteGraph(host_graph_);
    TF_DeleteGraph(func_graph_);
    TF_DeleteStatus(s_);
  }

  void Run(const std::vector<std::pair<TF_Operation*, TF_Tensor*>>& inputs,
           TF_Operation* output, int32_t expected_result) {
    Run(inputs, {{output, 0}}, {expected_result});
  }

  // Run the host graph, which now contains a function and check that
  // outputs are as expected.
  // 'T' stands for 'tensor' since the outputs are tensors, not scalars.
  void RunT(const std::vector<std::pair<TF_Operation*, TF_Tensor*>>& inputs,
            std::initializer_list<TF_Output> outputs,
            const std::vector<std::vector<int32_t>>& expected_results) {
    // Create a session for this graph
    CSession csession(host_graph_, s_);
    ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);

    // Run
    csession.SetInputs(inputs);
    csession.SetOutputs(outputs);
    csession.Run(s_);
    ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);

    // Check results
    for (int i = 0; i < expected_results.size(); ++i) {
      TF_Tensor* out = csession.output_tensor(i);
      ASSERT_TRUE(out != nullptr);
      EXPECT_EQ(TF_INT32, TF_TensorType(out));
      EXPECT_EQ(1, TF_NumDims(out));
      CompareInt32Tensor(expected_results[i], out);
    }
  }

  // Run the host graph, which now contains a function and check that
  // outputs are as expected.
  void Run(const std::vector<std::pair<TF_Operation*, TF_Tensor*>>& inputs,
           std::initializer_list<TF_Output> outputs,
           const std::vector<int32_t>& expected_results) {
    // Create a session for this graph.
    CSession csession(host_graph_, s_);
    ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);

    csession.SetInputs(inputs);
    csession.SetOutputs(outputs);
    csession.Run(s_);
    ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);

    for (int i = 0; i < expected_results.size(); ++i) {
      TF_Tensor* out = csession.output_tensor(i);
      ASSERT_TRUE(out != nullptr);
      EXPECT_EQ(TF_INT32, TF_TensorType(out));
      EXPECT_EQ(0, TF_NumDims(out));  // scalar
      ASSERT_EQ(sizeof(int32_t), TF_TensorByteSize(out));
      int32_t* output_contents = static_cast<int32_t*>(TF_TensorData(out));
      EXPECT_EQ(expected_results[i], *output_contents);
    }
  }

  void CompareInt32Tensor(const std::vector<int32_t>& expected, TF_Tensor* t) {
    int32_t* data = static_cast<int32_t*>(TF_TensorData(t));
    size_t size = TF_TensorByteSize(t);
    ASSERT_EQ(expected.size() * sizeof(int32_t), size);
    for (int i = 0; i < expected.size(); ++i) {
      ASSERT_EQ(expected[i], data[i]) << "Different data at index " << i;
    }
  }

  std::vector<TF_Output> ToOutput(const std::vector<TF_Operation*> ops) {
    std::vector<TF_Output> out;
    for (auto op : ops) {
      out.push_back({op, 0});
    }
    return out;
  }

  void Define(int num_opers, const std::vector<TF_Operation*>& opers,
              const std::vector<TF_Operation*>& inputs,
              const std::vector<TF_Operation*>& outputs,
              const char** output_names, bool expect_failure = false) {
    DefineT(num_opers, opers, ToOutput(inputs), ToOutput(outputs), output_names,
            expect_failure);
  }

  // An explicit `num_opers` is needed so that we can distinguish between the
  // case of no operations specified (-1) and the case of an empty set of
  // operations specified (0).
  void DefineT(int num_opers, const std::vector<TF_Operation*>& opers,
               const std::vector<TF_Output>& inputs,
               const std::vector<TF_Output>& outputs, const char** output_names,
               bool expect_failure = false) {
    ASSERT_EQ(func_, nullptr);
    func_ = TF_GraphToFunction(func_graph_, func_name_, num_opers,
                               num_opers == -1 ? nullptr : opers.data(),
                               inputs.size(), inputs.data(), outputs.size(),
                               outputs.data(), output_names,
                               /*opts=*/nullptr, s_);
    if (expect_failure) {
      ASSERT_EQ(func_, nullptr);
      return;
    }

    ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
    ASSERT_NE(func_, nullptr);
    TF_GraphAddFunction(host_graph_, func_, s_);
    ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  }

  TF_Operation* Use(const std::vector<TF_Operation*>& inputs) {
    return UseT(ToOutput(inputs));
  }

  TF_Operation* UseT(const std::vector<TF_Output>& inputs) {
    TF_Operation* op;
    UseHelper(inputs, &op);
    return op;
  }

  // All the *Helper methods are used as a workaround for the restrictions that
  // one cannot call ASSERT_* methods in non-void-returning functions (when
  // exceptions are disabled during compilation)
  void UseHelper(const std::vector<TF_Output>& inputs, TF_Operation** op) {
    TF_OperationDescription* desc =
        TF_NewOperation(host_graph_, func_name_, func_node_name_);
    for (auto input : inputs) {
      TF_AddInput(desc, input);
    }
    // Set device to CPU because some ops inside the function might not be
    // available on GPU.
    TF_SetDevice(desc, "/cpu:0");
    *op = TF_FinishOperation(desc, s_);
    ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
    ASSERT_NE(*op, nullptr);
  }

  FunctionDef fdef() {
    tensorflow::FunctionDef fdef;
    EXPECT_TRUE(GetFunctionDef(func_, &fdef));
    return fdef;
  }

  // logging utility
  template <class Container>
  string ToString(const Container& v) {
    std::stringstream ss;
    ss << "{";
    size_t i = 0;
    for (const auto& e : v) {
      if (i != 0) {
        ss << ", ";
      }
      ss << e.ToString();
      ++i;
    }
    ss << "}";
    return ss.str();
  }

  void VerifyFDefNodes(const tensorflow::FunctionDef& fdef,
                       const std::unordered_set<string>& nodes) {
    ASSERT_EQ(nodes.size(), fdef.node_def_size())
        << "Got unexpected number of nodes. Expected: ["
        << str_util::Join(nodes, ", ")
        << "] Actual nodes in fdef: " << fdef.DebugString();
    for (const NodeDef& node_def : fdef.node_def()) {
      ASSERT_TRUE(nodes.find(node_def.name()) != nodes.end())
          << "Got unexpected node: " << node_def.name()
          << " in fdef: " << fdef.DebugString();
    }
  }

  void VerifyFDefInputs(const tensorflow::FunctionDef& fdef,
                        const std::vector<IOSpec>& inputs) {
    const OpDef& signature = fdef.signature();
    ASSERT_EQ(inputs.size(), signature.input_arg_size());
    for (int i = 0; i < inputs.size(); ++i) {
      const OpDef::ArgDef& arg = signature.input_arg(i);
      const IOSpec& in = inputs[i];
      if (in.second != DT_INVALID) {
        ASSERT_EQ(arg.type(), in.second)
            << "Got unexpected type for input " << i
            << ". fdef: " << fdef.DebugString();
      }
      ASSERT_EQ(arg.name(), in.first) << "Got unexpected name for input " << i
                                      << ". fdef: " << fdef.DebugString();
    }
  }

  void VerifyFDefOutputs(const tensorflow::FunctionDef& fdef,
                         const std::vector<IOSpec>& outputs) {
    const OpDef& signature = fdef.signature();
    ASSERT_EQ(outputs.size(), signature.output_arg_size());
    for (int i = 0; i < outputs.size(); ++i) {
      const OpDef::ArgDef& arg = signature.output_arg(i);
      const IOSpec& out = outputs[i];
      if (out.second != DT_INVALID) {
        ASSERT_EQ(arg.type(), out.second)
            << "Got unexpected type for output " << i
            << ". fdef: " << fdef.DebugString();
      }
      ASSERT_EQ(arg.name(), out.first) << "Got unexpected name for output " << i
                                       << ". fdef: " << fdef.DebugString();
    }
  }

  void VerifyFDefEdges(
      const tensorflow::FunctionDef& fdef,
      const std::vector<EdgeSpec>& e_edges,  // expected edges
      const std::vector<EdgeSpec>& c_edges,  // expected ctrl edges
      bool is_exact_edges = true) {
    // Build a set of edges from fdef
    std::set<EdgeSpec> a_edges;  // actual edges
    // Get edges from inputs to body nodes and between body nodes
    for (const NodeDef& node_def : fdef.node_def()) {
      for (int i = 0; i < node_def.input_size(); ++i) {
        const string& in = node_def.input(i);
        const auto& v =
            a_edges.insert({in, strings::StrCat(node_def.name(), ":", i)});
        ASSERT_TRUE(v.second) << "Duplicate edge " << in << " -> "
                              << strings::StrCat(node_def.name(), ":", i)
                              << ". fdef: " << fdef.DebugString();
      }
    }
    // Get edges from body nodes to outputs and from inputs to outputs
    for (const OpDef::ArgDef& arg : fdef.signature().output_arg()) {
      const auto& iter = fdef.ret().find(arg.name());
      if (iter != fdef.ret().end()) {
        const auto& v = a_edges.insert({iter->second, arg.name()});
        ASSERT_TRUE(v.second) << "Duplicate edge " << iter->second << " -> "
                              << arg.name() << ". fdef: " << fdef.DebugString();
      } else {
        const auto& v = a_edges.insert({arg.name(), arg.name()});
        ASSERT_TRUE(v.second) << "Duplicate edge " << arg.name() << " -> "
                              << arg.name() << ". fdef: " << fdef.DebugString();
      }
    }

    // Verify edges
    for (const EdgeSpec& e : e_edges) {
      ASSERT_TRUE(a_edges.find(e) != a_edges.end())
          << "Failed to find expected edge " << e.ToString()
          << " in fdef: " << fdef.DebugString();
    }

    // If caller specified all edges, check that we have seen all
    if (is_exact_edges) {
      ASSERT_EQ(e_edges.size() + c_edges.size(), a_edges.size())
          << "Expected edges: " << ToString(e_edges)
          << " Expected Control edges: " << ToString(c_edges)
          << " Actual edges: " << ToString(a_edges)
          << " in fdef: " << fdef.DebugString();
    }
  }

  void VerifyFDef(const std::unordered_set<string>& nodes,
                  const std::vector<IOSpec>& inputs,
                  const std::vector<IOSpec>& outputs,
                  const std::vector<EdgeSpec>& e_edges,  // expected edges
                  const std::vector<EdgeSpec>& c_edges,  // expected ctrl edges
                  bool is_exact_edges = true) {
    tensorflow::FunctionDef fdef;
    ASSERT_TRUE(GetFunctionDef(func_, &fdef));
    VerifyFDefNodes(fdef, nodes);
    VerifyFDefInputs(fdef, inputs);
    VerifyFDefOutputs(fdef, outputs);
    VerifyFDefEdges(fdef, e_edges, c_edges, is_exact_edges);
  }

  const char* func_name_ = "MyFunc";
  const char* func_node_name_ = "MyFunc_0";
  TF_Status* s_;
  TF_Graph* func_graph_;
  TF_Graph* host_graph_;
  TF_Function* func_;

  // Workaround for not being able to initialize empty map using {}
  std::unordered_set<string> empty_;
};

TEST_F(CApiFunctionTest, OneOp_ZeroInputs_OneOutput) {
  /*
   *                constant
   *                   |
   *                   v
   */
  // Define
  TF_Operation* c = ScalarConst(10, func_graph_, s_, "scalar10");
  Define(-1, {}, {}, {c}, nullptr);

  // Use, run, and verify
  TF_Operation* func_op = Use({});
  Run({}, func_op, 10);
  VerifyFDef({"scalar10_0"}, {}, {{"scalar10", DT_INT32}},
             {{"scalar10_0:output:0", "scalar10"}}, {});
}

TEST_F(CApiFunctionTest, OneOp_OneInput_OneOutput) {
  /*
   *                   |
   *                   v
   *                 negate
   *                   |
   *                   v
   */
  // Define
  TF_Operation* feed = Placeholder(func_graph_, s_);
  TF_Operation* neg = Neg(feed, func_graph_, s_);
  Define(-1, {}, {feed}, {neg}, nullptr);

  // Use, run, and verify
  TF_Operation* func_feed = Placeholder(host_graph_, s_);
  TF_Operation* func_op = Use({func_feed});
  Run({{func_feed, Int32Tensor(3)}}, func_op, -3);
  VerifyFDef({"neg_0"}, {{"feed", DT_INT32}}, {{"neg", DT_INT32}},
             {{"feed", "neg_0:0"}, {"neg_0:y:0", "neg"}}, {});
}

TEST_F(CApiFunctionTest, ZeroOps_Identity) {
  /*
   *                   |
   *                   |
   *                   |
   *                   v
   */
  // Define
  TF_Operation* feed = Placeholder(func_graph_, s_);
  Define(-1, {}, {feed}, {feed}, nullptr);

  // Use, run, and verify
  TF_Operation* func_feed = Placeholder(host_graph_, s_);
  TF_Operation* func_op = Use({func_feed});
  Run({{func_feed, Int32Tensor(3)}}, func_op, 3);
  VerifyFDef(empty_, {{"feed", DT_INT32}}, {{"feed_0", DT_INT32}},
             {{"feed", "feed_0"}}, {});
}

TEST_F(CApiFunctionTest, ZeroOps_Permutation) {
  /*
   *                   |   |
   *                   \  /
   *                    \/
   *                    x
   *                   /\
   *                  /  \
   *                 |   |
   *                 v   v
   */
  // Define
  TF_Operation* feed1 = Placeholder(func_graph_, s_, "feed1");
  TF_Operation* feed2 = Placeholder(func_graph_, s_, "feed2");
  Define(-1, {}, {feed1, feed2}, {feed2, feed1}, nullptr);

  // Use, run, and verify
  TF_Operation* two = ScalarConst(2, host_graph_, s_);
  TF_Operation* func_feed = Placeholder(host_graph_, s_);
  TF_Operation* func_op = Use({two, func_feed});
  Run({{func_feed, Int32Tensor(3)}}, {{func_op, 0}, {func_op, 1}}, {3, 2});
  VerifyFDef(empty_, M({{"feed1"}, {"feed2"}}), M({{"feed2_0"}, {"feed1_0"}}),
             {{"feed1", "feed1_0"}, {"feed2", "feed2_0"}}, {});
}

TEST_F(CApiFunctionTest, OneOp_TwoInputs_OneOutput) {
  /*
   *                  |  |
   *                  v  v
   *                  add
   *                   |
   *                   v
   */
  // Define
  TF_Operation* feed1 = Placeholder(func_graph_, s_, "feed1");
  TF_Operation* feed2 = Placeholder(func_graph_, s_, "feed2");
  TF_Operation* add = Add(feed1, feed2, func_graph_, s_);
  Define(-1, {}, {feed1, feed2}, {add}, nullptr);

  // Use, run, and verify
  TF_Operation* two = ScalarConst(2, host_graph_, s_);
  TF_Operation* func_feed = Placeholder(host_graph_, s_);
  TF_Operation* func_op = Use({two, func_feed});
  Run({{func_feed, Int32Tensor(3)}}, func_op, 2 + 3);
  VerifyFDef(
      {"add_0"}, M({{"feed1"}, {"feed2"}}), M({{"add"}}),
      {{"feed1", "add_0:0"}, {"feed2", "add_0:1"}, {"add_0:sum:0", "add"}}, {});
}

TEST_F(CApiFunctionTest, OneOp_TwoInputs_ZeroOutputs) {
  /*
   *                  |  |
   *                  v  v
   *                  add
   *
   *            (output ignored)
   */
  // Define
  TF_Operation* feed1 = Placeholder(func_graph_, s_, "feed1");
  TF_Operation* feed2 = Placeholder(func_graph_, s_, "feed2");
  Add(feed1, feed2, func_graph_, s_);
  Define(-1, {}, {feed1, feed2}, {}, nullptr);

  // Use, run, and verify
  TF_Operation* two = ScalarConst(2, host_graph_, s_);
  TF_Operation* func_feed = Placeholder(host_graph_, s_);
  Use({two, func_feed});
  VerifyFDef({"add"}, M({{"feed1"}, {"feed2"}}), {},
             {{"feed1", "add:0"}, {"feed2", "add:1"}}, {});
}

TEST_F(CApiFunctionTest, TwoOps_ThreeInputs_OneOutput) {
  /*
   *                  |  |   |
   *                  v  v   /
   *                  add1  /
   *                   |   |
   *                   v   v
   *                   add2
   *                    |
   *                    v
   */
  // Define
  TF_Operation* feed1 = Placeholder(func_graph_, s_, "feed1");
  TF_Operation* feed2 = Placeholder(func_graph_, s_, "feed2");
  TF_Operation* feed3 = Placeholder(func_graph_, s_, "feed3");
  TF_Operation* add1 = Add(feed1, feed2, func_graph_, s_, "add1");
  TF_Operation* add2 = Add(add1, feed3, func_graph_, s_, "add2");
  Define(-1, {}, {feed1, feed2, feed3}, {add2}, nullptr);

  // Use, run, and verify
  TF_Operation* two = ScalarConst(2, host_graph_, s_, "two");
  TF_Operation* ten = ScalarConst(10, host_graph_, s_, "ten");
  TF_Operation* func_feed = Placeholder(host_graph_, s_);
  TF_Operation* func_op = Use({two, ten, func_feed});
  Run({{func_feed, Int32Tensor(3)}}, func_op, 2 + 10 + 3);
  VerifyFDef({"add1", "add2_0"}, M({{"feed1"}, {"feed2"}, {"feed3"}}),
             M({{"add2"}}),
             {{"feed1", "add1:0"},
              {"feed2", "add1:1"},
              {"add1:sum:0", "add2_0:0"},
              {"feed3", "add2_0:1"},
              {"add2_0:sum:0", "add2"}},
             {});
}

TEST_F(CApiFunctionTest, OneOp_TwoInputs_TwoDuplicateOutputs) {
  /*
   *                  |  |
   *                  v  v
   *                  add
   *                   |
   *                 +-+-+
   *                 |   |
   *                 v   v
   */
  // Define
  TF_Operation* feed1 = Placeholder(func_graph_, s_, "feed1");
  TF_Operation* feed2 = Placeholder(func_graph_, s_, "feed2");
  TF_Operation* add = Add(feed1, feed2, func_graph_, s_);
  Define(-1, {}, {feed1, feed2}, {add, add}, nullptr);

  // Use, run, and verify
  TF_Operation* two = ScalarConst(2, host_graph_, s_);
  TF_Operation* func_feed = Placeholder(host_graph_, s_);
  TF_Operation* func_op = Use({two, func_feed});
  Run({{func_feed, Int32Tensor(3)}}, {{func_op, 0}, {func_op, 1}}, {5, 5});
  VerifyFDef({"add_1"}, M({{"feed1"}, {"feed2"}}), M({{"add"}, {"add_0"}}),
             {{"feed1", "add_1:0"},
              {"feed2", "add_1:1"},
              {"add_1:sum:0", "add"},
              {"add_1:sum:0", "add_0"}},
             {});
}

TEST_F(CApiFunctionTest, TwoOps_ThreeInputs_TwoOutputs) {
  /*
   *                  |  |  |
   *                  v  v  /
   *                  add  /
   *                   |  |
   *                 +-+  |
   *                 | |  |
   *                 | v  v
   *                 | add
   *                 |  |
   *                 v  v
   */
  // Define
  TF_Operation* feed1 = Placeholder(func_graph_, s_, "feed1");
  TF_Operation* feed2 = Placeholder(func_graph_, s_, "feed2");
  TF_Operation* feed3 = Placeholder(func_graph_, s_, "feed3");
  TF_Operation* add1 = Add(feed1, feed2, func_graph_, s_, "add1");
  TF_Operation* add2 = Add(add1, feed3, func_graph_, s_, "add2");
  Define(-1, {}, {feed1, feed2, feed3}, {add1, add2}, nullptr);

  // Use, run, and verify
  TF_Operation* two = ScalarConst(2, host_graph_, s_, "two");
  TF_Operation* ten = ScalarConst(10, host_graph_, s_, "ten");
  TF_Operation* func_feed = Placeholder(host_graph_, s_);
  TF_Operation* func_op = Use({two, ten, func_feed});
  Run({{func_feed, Int32Tensor(3)}}, {{func_op, 0}, {func_op, 1}}, {12, 15});
  VerifyFDef({"add1_0", "add2_0"}, M({{"feed1"}, {"feed2"}, {"feed3"}}),
             M({{"add1"}, {"add2"}}),
             {{"feed1", "add1_0:0"},
              {"feed2", "add1_0:1"},
              {"add1_0:sum:0", "add2_0:0"},
              {"feed3", "add2_0:1"},
              {"add1_0:sum:0", "add1"},
              {"add2_0:sum:0", "add2"}},
             {});
}

TEST_F(CApiFunctionTest, FromSubsetOfOps) {
  /*
   *                  |  |  |
   *                  v  v  /
   *                  add  /
   *                   |  |
   *               +---+--+---+
   *  Ops used     |   |  |   |
   *  for func     |   v  v   |
   *     |         |   add    |
   *     +-------> |    |     |
   *               |    v     |
   *               |          |
   *               +----------+
   */
  // Define
  TF_Operation* feed1 = Placeholder(func_graph_, s_, "feed1");
  TF_Operation* feed2 = Placeholder(func_graph_, s_, "feed2");
  TF_Operation* feed3 = Placeholder(func_graph_, s_, "feed3");
  TF_Operation* add1 = Add(feed1, feed2, func_graph_, s_, "add1");
  TF_Operation* add2 = Add(add1, feed3, func_graph_, s_, "add2");
  Define(1, {add2}, {add1, feed3}, {add2}, nullptr);

  // Use, run, and verify
  TF_Operation* two = ScalarConst(2, host_graph_, s_, "two");
  TF_Operation* func_feed = Placeholder(host_graph_, s_);
  TF_Operation* func_op = Use({two, func_feed});
  Run({{func_feed, Int32Tensor(3)}}, func_op, 2 + 3);
  VerifyFDef(
      {"add2_0"}, M({{"add1"}, {"feed3"}}), M({{"add2"}}),
      {{"add1", "add2_0:0"}, {"feed3", "add2_0:1"}, {"add2_0:sum:0", "add2"}},
      {});
}

TEST_F(CApiFunctionTest, UsingOneOutputOfSplit) {
  /*
   *                      feed
   *                       |
   *             +---------+---+
   *             | const0  |   |
   *             |    |    |   |
   *             |    v    /   |
   *             |    split    |
   *             |   |  |  |   |
   *             |   v  |  v   |
   *             |      |      |
   *             +------+------+
   *                    |
   *                    v
   *
   *  Only the second output from split is used as function output
   */
  // Define
  TF_Operation* feed = Placeholder(func_graph_, s_);
  TF_Operation* split = Split3(feed, func_graph_, s_);
  DefineT(-1, {}, {{feed, 0}}, {{split, 1}}, nullptr);

  // Use, run, and verify
  TF_Operation* func_feed = Placeholder(host_graph_, s_);
  TF_Operation* func_op = Use({func_feed});
  RunT({{func_feed, Int32Tensor({1, 2, 3, 4, 5, 6})}}, {{func_op, 0}},
       {{3, 4}});
  VerifyFDef({"split3_const0", "split3_0"}, M({{"feed"}}), M({{"split3"}}),
             {{"split3_const0:output:0", "split3_0:0"},
              {"feed", "split3_0:1"},
              {"split3_0:output:1", "split3"}},
             {});
}

TEST_F(CApiFunctionTest, UsingTwoOutputsOfSplit) {
  /*
   *                      feed
   *                       |
   *             +---------+---+
   *             | const0  |   |
   *             |    |    |   |
   *             |    v    /   |
   *             |    split    |
   *             |   |  |  |   |
   *             |   |  v  |   |
   *             |   |     |   |
   *             +---+-----+---+
   *                 |     |
   *                 v     v
   *
   *  Second output from split is not used as function output
   */
  // Define
  TF_Operation* feed = Placeholder(func_graph_, s_);
  TF_Operation* split = Split3(feed, func_graph_, s_);
  DefineT(-1, {}, {{feed, 0}}, {{split, 0}, {split, 2}}, nullptr);

  // Use, run, and verify
  TF_Operation* func_feed = Placeholder(host_graph_, s_);
  TF_Operation* func_op = Use({func_feed});
  RunT({{func_feed, Int32Tensor({1, 2, 3, 4, 5, 6})}},
       {{func_op, 0}, {func_op, 1}}, {{1, 2}, {5, 6}});
  VerifyFDef({"split3_const0", "split3_1"}, M({{"feed"}}),
             M({{"split3"}, {"split3_0"}}),
             {{"split3_const0:output:0", "split3_1:0"},
              {"feed", "split3_1:1"},
              {"split3_1:output:0", "split3"},
              {"split3_1:output:2", "split3_0"}},
             {});
}

TEST_F(CApiFunctionTest, UsingTwoOutputsOfSplitAsInputs) {
  /*
   *                    |
   *                    v
   *                  split
   *                 |  |  |
   *                 |  v  |
   *                 |     |
   *             +---+-----+---+
   *             |   |     |   |
   *             |   v     v   |
   *             |     add     |
   *             |      |      |
   *             |      |      |
   *             +------+------+
   *                    |
   *                    v
   */
  // Define
  TF_Operation* feed = Placeholder(func_graph_, s_);
  TF_Operation* split = Split3(feed, func_graph_, s_);
  TF_Operation* add = Add({split, 0}, {split, 2}, func_graph_, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  DefineT(1, {add}, {{split, 0}, {split, 2}}, {{add, 0}}, nullptr);

  // Use, run, and verify
  TF_Operation* two = ScalarConst(2, host_graph_, s_, "two");
  TF_Operation* func_feed = Placeholder(host_graph_, s_);
  TF_Operation* func_op = Use({two, func_feed});
  Run({{func_feed, Int32Tensor(3)}}, func_op, 2 + 3);
  VerifyFDef(
      {"add_0"}, M({{"split3"}, {"split3_0"}}), M({{"add"}}),
      {{"split3", "add_0:0"}, {"split3_0", "add_0:1"}, {"add_0:sum:0", "add"}},
      {});
}

TEST_F(CApiFunctionTest, NodesUsedInInputsMustHaveSingleOutput) {
  /*
   *                    |
   *                    v
   *                  split
   *                 |  |  |
   *                 |  v  |
   *                 |     |
   *       input --->|     |<--- input
   *                 |     |
   *                 v     v
   *                   add
   *                    |
   *                    |
   *                    v
   */
  // Define
  TF_Tensor* tensor_123 = Int32Tensor({1, 2, 3});
  TF_Operation* c = Const(tensor_123, func_graph_, s_, "const_array");
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  TF_Operation* split = Split3(c, func_graph_, s_);
  TF_Operation* add = Add({split, 0}, {split, 2}, func_graph_, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  DefineT(-1, {}, {{split, 0}, {split, 2}}, {{add, 0}}, nullptr, true);
  EXPECT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(s_));
  EXPECT_EQ(string("When `num_opers` is set to -1, nodes referenced in "
                   "`inputs` must have a single output. Node split3 has "
                   "3 outputs. Encountered while creating function 'MyFunc'"),
            string(TF_Message(s_)));

  TF_DeleteTensor(tensor_123);
}

TEST_F(CApiFunctionTest, FunctionWithWhileLoop) {
  // Inputs to the while loop and the function as a whole
  TF_Operation* feed1 = Placeholder(func_graph_, s_, "feed1");
  TF_Operation* feed2 = Placeholder(func_graph_, s_, "feed2");

  // Outputs of the while loop corresponding to the two inputs above
  // The first one will the function's output
  std::vector<TF_Output> outputs;

  // Add while loop to func_graph_
  {
    // The inputs to the while loop
    std::vector<TF_Output> inputs = {{feed1, 0}, {feed2, 0}};
    std::unique_ptr<TF_WhileParams> params(new TF_WhileParams(
        TF_NewWhile(func_graph_, &inputs[0], inputs.size(), s_)));
    ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
    params->name = "test_loop";

    // Initialize outputs so we can easily detect errors/bugs
    outputs.resize(2, {nullptr, -1});

    // Create loop: while (input1 < input2) input1 += input2 + 1
    TF_Operation* less_than = LessThan(
        params->cond_inputs[0], params->cond_inputs[1], params->cond_graph, s_);
    ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
    params->cond_output = {less_than, 0};

    TF_Operation* add1 = Add(params->body_inputs[0], params->body_inputs[1],
                             params->body_graph, s_, "add1");
    ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
    TF_Operation* one = ScalarConst(1, params->body_graph, s_);
    ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
    TF_Operation* add2 = Add(add1, one, params->body_graph, s_, "add2");
    ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
    params->body_outputs[0] = {add2, 0};
    params->body_outputs[1] = params->body_inputs[1];

    // Finalize while loop
    TF_FinishWhile(params.get(), s_, &outputs[0]);
    EXPECT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  }

  // Define function, use it in graph, and run
  DefineT(-1, {}, {{feed1, 0}, {feed2, 0}}, {outputs[0]}, nullptr);
  TF_Operation* five = ScalarConst(5, host_graph_, s_, "five");
  TF_Operation* func_feed = Placeholder(host_graph_, s_);
  TF_Operation* func_op = Use({func_feed, five});
  Run({{func_feed, Int32Tensor(2)}}, func_op, 2 /*+=*/ + 5 + 1);

  // Verify input, output, and subset of edges in fdef.
  // The subset of edges we verify is a chain between feed1 and output to
  // make sure that the correct output is picked.
  tensorflow::FunctionDef fdef;
  ASSERT_TRUE(GetFunctionDef(func_, &fdef));
  VerifyFDefInputs(fdef, M({{"feed1"}, {"feed2"}}));
  VerifyFDefOutputs(fdef, M({{"test_loop_exit"}}));
  VerifyFDefEdges(fdef,
                  {{"feed1", "test_loop/Enter:0"},
                   {"test_loop/Enter:output:0", "test_loop/Merge:0"},
                   {"test_loop/Merge:output:0", "test_loop/Switch:0"},
                   {"test_loop/Switch:output_false:0", "test_loop/Exit:0"},
                   {"test_loop/Exit:output:0", "test_loop_exit"}},
                  {}, false);
}

TEST_F(CApiFunctionTest, ControlDependency) {
  /*
   *                  |  |    scalar
   *                  |  |    .
   *                  v  v   . <---- control dependency
   *                  add < -
   *                   |
   *                   v
   */
  // Define
  TF_Operation* feed1 = Placeholder(func_graph_, s_, "feed1");
  TF_Operation* feed2 = Placeholder(func_graph_, s_, "feed2");
  TF_Operation* five = ScalarConst(5, func_graph_, s_);
  TF_Operation* add =
      AddWithCtrlDependency(feed1, feed2, func_graph_, five, s_);
  EXPECT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  Define(-1, {}, {feed1, feed2}, {add}, nullptr);

  // Use, run, and verify
  TF_Operation* two = ScalarConst(2, host_graph_, s_);
  TF_Operation* func_feed = Placeholder(host_graph_, s_);
  TF_Operation* func_op = Use({two, func_feed});
  Run({{func_feed, Int32Tensor(3)}}, func_op, 2 + 3);
  VerifyFDef(
      {"add_0", "scalar"}, M({{"feed1"}, {"feed2"}}), M({{"add"}}),
      {{"feed1", "add_0:0"}, {"feed2", "add_0:1"}, {"add_0:sum:0", "add"}},
      {{"scalar", "add_0"}});
}

TEST_F(CApiFunctionTest, ControlDependencyOutsideOfBody) {
  /*
   *                  |  |    scalar
   *                  |  |    .
   *                  v  v   . <---- control dependency
   *                  add < -
   *                   |
   *                   v
   */
  // Define
  TF_Operation* feed1 = Placeholder(func_graph_, s_, "feed1");
  TF_Operation* feed2 = Placeholder(func_graph_, s_, "feed2");
  TF_Operation* five = ScalarConst(5, func_graph_, s_);
  TF_Operation* add =
      AddWithCtrlDependency(feed1, feed2, func_graph_, five, s_);
  EXPECT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  Define(1, {add}, {feed1, feed2}, {add}, nullptr, true);
  EXPECT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(s_));
  EXPECT_EQ(string("The source of control edge [id=3 scalar:-1 -> add:-1] "
                   "is not in the body. Encountered while creating "
                   "function 'MyFunc'"),
            string(TF_Message(s_)));
}

TEST_F(CApiFunctionTest, ControlDependencyOutsideOfBody_FromInputNode) {
  /*
   *                  |  |.
   *                  |  |  .
   *                  |  |   .
   *                  v  v   . <---- control dependency
   *                  add < -
   *                   |
   *                   v
   */
  // Define
  TF_Operation* feed1 = Placeholder(func_graph_, s_, "feed1");
  TF_Operation* feed2 = Placeholder(func_graph_, s_, "feed2");
  TF_Operation* add =
      AddWithCtrlDependency(feed1, feed2, func_graph_, feed1, s_);
  EXPECT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  Define(-1, {}, {feed1, feed2}, {add}, nullptr, true);
  EXPECT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(s_));
  EXPECT_EQ(string("The source of control edge [id=3 feed1:-1 -> add:-1] "
                   "is not in the body. Encountered while creating "
                   "function 'MyFunc'"),
            string(TF_Message(s_)));
}

TEST_F(CApiFunctionTest, DuplicateInputsAreNotAllowed) {
  /*
   *                  feed
   *                   |
   *                  +++
   *                  | |
   *              +---+-+---+
   *              |   | |   |
   *              |   v v   |
   *              |   add   |
   *              |    |    |
   *              |    |    |
   *              +----+----+
   *                   |
   *                   v
   */
  TF_Operation* feed1 = Placeholder(func_graph_, s_, "feed1");
  TF_Operation* add = Add(feed1, feed1, func_graph_, s_);
  Define(-1, {}, {feed1, feed1}, {add}, nullptr, true);
  EXPECT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(s_));
  EXPECT_EQ(
      string("TF_Output feed1:0 appears more than once in the input list"),
      string(TF_Message(s_)));
}

TEST_F(CApiFunctionTest, InvalidInputTensor_HighIndex) {
  /*
   *                  |  |
   *                  v  v
   *                  add
   *                   |
   *                   v
   */
  TF_Operation* feed1 = Placeholder(func_graph_, s_, "feed1");
  TF_Operation* feed2 = Placeholder(func_graph_, s_, "feed2");
  TF_Operation* add = Add(feed1, feed2, func_graph_, s_);
  DefineT(-1, {}, {{feed1, 0}, {feed2, 2}}, {{add, 0}}, nullptr, true);
  EXPECT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(s_));
  EXPECT_EQ(string("Node 'feed2' (type: 'Placeholder', num of outputs: 1) does "
                   "not have output 2\n\tEncountered while processing "
                   "input 1 into function 'MyFunc'"),
            string(TF_Message(s_)));
}

TEST_F(CApiFunctionTest, InvalidInputTensor_BadNodePtr) {
  /*
   *                  |  |
   *                  v  v
   *                  add
   *                   |
   *                   v
   */
  TF_Operation* feed1 = Placeholder(func_graph_, s_, "feed1");
  TF_Operation* feed2 = Placeholder(func_graph_, s_, "feed2");
  TF_Operation* add = Add(feed1, feed2, func_graph_, s_);
  DefineT(-1, {}, {{feed1, 0}, {nullptr, 0}}, {{add, 0}}, nullptr, true);
  EXPECT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(s_));
  EXPECT_EQ(string("Node is null\n\tEncountered while processing input 1 "
                   "into function 'MyFunc'"),
            string(TF_Message(s_)));
}

TEST_F(CApiFunctionTest, InvalidOutputTensor_HighIndex) {
  /*
   *                  |  |
   *                  v  v
   *                  add
   *                   |
   *                   v
   */
  TF_Operation* feed1 = Placeholder(func_graph_, s_, "feed1");
  TF_Operation* feed2 = Placeholder(func_graph_, s_, "feed2");
  TF_Operation* add = Add(feed1, feed2, func_graph_, s_);
  DefineT(-1, {}, {{feed1, 0}, {feed2, 0}}, {{add, 3}}, nullptr, true);
  EXPECT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(s_));
  EXPECT_EQ(string("Node 'add' (type: 'AddN', num of outputs: 1) does "
                   "not have output 3\n\tEncountered while processing "
                   "output 0 from function 'MyFunc'"),
            string(TF_Message(s_)));
}

TEST_F(CApiFunctionTest, InvalidOutputTensor_BadNodePtr) {
  /*
   *                  |  |
   *                  v  v
   *                  add
   *                   |
   *                   v
   */
  TF_Operation* feed1 = Placeholder(func_graph_, s_, "feed1");
  TF_Operation* feed2 = Placeholder(func_graph_, s_, "feed2");
  Add(feed1, feed2, func_graph_, s_);
  DefineT(-1, {}, {{feed1, 0}, {feed2, 0}}, {{nullptr, 3}}, nullptr, true);
  EXPECT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(s_));
  EXPECT_EQ(string("Node is null\n\tEncountered while processing output 0 "
                   "from function 'MyFunc'"),
            string(TF_Message(s_)));
}

TEST_F(CApiFunctionTest, NodeMissingInput) {
  /*
   *        input---> |  | <----missing input
   *                  v  v
   *        body----> add
   *                   |
   *                   v
   */
  TF_Operation* feed1 = Placeholder(func_graph_, s_, "feed1");
  TF_Operation* feed2 = Placeholder(func_graph_, s_, "feed2");
  TF_Operation* add = Add(feed1, feed2, func_graph_, s_);
  DefineT(1, {add}, {{feed1, 0}}, {{add, 0}}, nullptr, true);
  EXPECT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(s_));
  EXPECT_EQ(string("Input 1, 'feed2:0', of node 'add' in function 'MyFunc' "
                   "is not available. You might need to include it in inputs "
                   "or include its source node in the body"),
            string(TF_Message(s_)));
}

TEST_F(CApiFunctionTest, OutputOpNotInBody) {
  /*
   *                  |  |
   *                  v  v
   *                  add    scalar    (scalar not included in body)
   *                   |       |
   *                   v       v       (function has two outputs)
   */
  // Define
  TF_Operation* feed1 = Placeholder(func_graph_, s_, "feed1");
  TF_Operation* feed2 = Placeholder(func_graph_, s_, "feed2");
  TF_Operation* scalar = ScalarConst(2, func_graph_, s_);
  TF_Operation* add = Add(feed1, feed2, func_graph_, s_);
  Define(1, {add}, {feed1, feed2}, {add, scalar}, nullptr, true);
  EXPECT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(s_));
  EXPECT_EQ(string("TF_Output scalar:0 is neither in the function body nor "
                   "among function inputs. Encountered while creating "
                   "function 'MyFunc'"),
            string(TF_Message(s_)));
}

}  // namespace
}  // namespace tensorflow
