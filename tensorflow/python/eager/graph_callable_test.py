# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import graph_callable
from tensorflow.python.eager import test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope


class GraphCallableTest(test.TestCase):

  def testBasic(self):

    @graph_callable.graph_callable(
        [graph_callable.ShapeAndDtype(shape=(), dtype=dtypes.float32)])
    def my_function(x):
      v = variable_scope.get_variable(
          "v", initializer=init_ops.zeros_initializer(), shape=())
      return v + x

    self.assertEqual(2,
                     my_function(ops.EagerTensor(2,
                                                 dtype=dtypes.float32)).numpy())
    my_function.variables[0].assign(1.)
    self.assertEqual(3,
                     my_function(ops.EagerTensor(2,
                                                 dtype=dtypes.float32)).numpy())

  def testMismatchingNumArgs(self):
    # pylint: disable=anomalous-backslash-in-string
    with self.assertRaisesRegexp(TypeError,
                                 "The number of arguments accepted by the "
                                 "decorated function `my_function` \(2\) must "
                                 "match the number of ShapeAndDtype objects "
                                 "passed to the graph_callable\(\) decorator "
                                 "\(1\)."):
      @graph_callable.graph_callable([
          graph_callable.ShapeAndDtype(shape=(), dtype=dtypes.float32)])
      def my_function(x, y):  # pylint: disable=unused-variable
        return x + y
    # pylint: enable=anomalous-backslash-in-string

  def testPureFunction(self):

    @graph_callable.graph_callable(
        [graph_callable.ShapeAndDtype(shape=(), dtype=dtypes.int32)])
    def f(x):
      return math_ops.add(x, ops.EagerTensor(3))

    self.assertAllEqual(5, f(ops.EagerTensor(2)).numpy())

  def testNestedFunction(self):

    # TensorFlow function (which is what would be used in TensorFlow graph
    # construction).
    @function.Defun(dtypes.int32, dtypes.int32)
    def add(a, b):
      return math_ops.add(a, b)

    # A graph_callable that will invoke the TensorFlow function.
    @graph_callable.graph_callable(
        [graph_callable.ShapeAndDtype(shape=(), dtype=dtypes.int32)])
    def add_one(x):
      return add(x, 1)

    self.assertAllEqual(3, add_one(ops.EagerTensor(2)).numpy())

  # TODO(ashankar): Make this work.
  # The problem is that the two graph_callables (for add_one and add_two)
  # are both trying to register the FunctionDef corresponding to "add".
  def DISABLED_testRepeatedUseOfSubFunction(self):

    @function.Defun(dtypes.int32, dtypes.int32)
    def add(a, b):
      return math_ops.add(a, b)

    @graph_callable.graph_callable(
        [graph_callable.ShapeAndDtype(shape=(), dtype=dtypes.int32)])
    def add_one(x):
      return add(x, 1)

    @graph_callable.graph_callable(
        [graph_callable.ShapeAndDtype(shape=(), dtype=dtypes.int32)])
    def add_two(x):
      return add(x, 2)

    two = ops.EagerTensor(2)
    self.assertAllEqual(3, add_one(two).numpy())
    self.assertAllEqual(4, add_two(two).numpy())

  def testNestedSequenceInputs(self):
    sd = graph_callable.ShapeAndDtype(shape=(), dtype=dtypes.float32)
    @graph_callable.graph_callable([[sd, tuple([sd, sd]), sd]])
    def my_op(inputs):
      a, b, c = inputs
      e, f = b
      v = variable_scope.get_variable(
          "my_v", initializer=init_ops.zeros_initializer(), shape=())
      return [a + a + v, tuple([e + e, f + f]), c + c], a + e + f + c + v

    inputs = [ops.EagerTensor(1.), [ops.EagerTensor(2.), ops.EagerTensor(3.)],
              ops.EagerTensor(4.)]
    ret = my_op(inputs)
    self.assertEqual(len(ret), 2.)
    self.assertEqual(ret[1].numpy(), 10.)

    my_op.variables[0].assign(1.)
    ret = my_op(inputs)
    self.assertEqual(ret[1].numpy(), 11.)


if __name__ == "__main__":
  test.main()
