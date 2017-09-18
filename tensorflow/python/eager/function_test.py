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

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.eager import tape
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function as tf_function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops


class FunctionTest(test.TestCase):

  def testBasic(self):
    matmul = function.defun(math_ops.matmul)
    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    sq = matmul(t, t, transpose_a=True)
    sq2 = matmul(sq, t, transpose_a=True)
    self.assertAllEqual(sq.numpy().reshape(-1), [10, 14, 14, 20])
    self.assertAllEqual(sq2.numpy().reshape(-1), [52, 76, 74, 108])

  def testBasicGraphMode(self):
    matmul = function.defun(math_ops.matmul)

    @function.defun
    def sq(a):
      return matmul(a, a)

    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    out = sq(t)
    self.assertAllEqual(out.numpy(), math_ops.matmul(t, t).numpy())

  def testGraphModeWithGradients(self):
    v = resource_variable_ops.ResourceVariable(1.0)

    @function.defun
    def step():
      def inner():
        tape.watch_variable(v)
        return v * v

      return backprop.implicit_grad(inner)()[0][0]

    self.assertAllEqual(step().numpy(), 2.0)

  def testTensorConversionWithDefun(self):

    @function.defun
    def f(x):
      return math_ops.add(x, constant_op.constant(3))

    self.assertAllEqual(5, f(constant_op.constant(2)).numpy())

  def testTensorConversionCall(self):

    @function.defun
    def f(x):
      return math_ops.add(x, constant_op.constant(3))

    @function.defun
    def g(x):
      return f(f(x))

    self.assertAllEqual(8, g(constant_op.constant(2)).numpy())

  def testDefunCallBackprop(self):

    @function.defun
    def f(x):
      return math_ops.add(x, x)

    @function.defun
    def g(x):
      return backprop.gradients_function(f, [0])(x)[0]

    self.assertAllEqual(2, g(constant_op.constant(2)).numpy())

  def testCallShape(self):

    @function.defun
    def f(x):
      return x + 1

    @function.defun
    def g(x):
      x = f(x)
      self.assertEqual(x.shape.as_list(), [])
      return None

    g(constant_op.constant(1.0))

  def testGradientTensorConversionWithDefun(self):
    three = resource_variable_ops.ResourceVariable(3.0)

    @function.defun
    def f(x):
      return math_ops.add(x, three)

    def g(x):
      tape.watch_variable(three)
      return f(x)

    g = backprop.implicit_grad(g)(constant_op.constant(1.0))[0][0]
    self.assertEqual(g.numpy(), 1.0)

  def testGradient(self):
    matmul = function.defun(math_ops.matmul)

    def sq(x):
      return matmul(x, x, transpose_a=True)

    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    grad_t, = backprop.gradients_function(sq, [0])(t)
    self.assertAllEqual(grad_t.numpy(), [[6, 6], [14, 14]])

  def testGradientInFunction(self):

    @function.defun
    def f(x):
      return backprop.gradients_function(lambda y: y * y, [0])(x)[0]

    self.assertEqual(f(constant_op.constant(1.0)).numpy(), 2.0)

  def testFunctionOnDevice(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')

    x = constant_op.constant([1.]).as_gpu_tensor()
    f = function.defun(math_ops.add)
    y = f(x, x).as_cpu_tensor()
    self.assertAllEqual(y.numpy(), [2.])

  def testFunctionHandlesInputsOnDifferentDevices(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')

    # The Reshape op requires the shape tensor to be placed in host memory.
    reshape = function.defun(array_ops.reshape)
    value = constant_op.constant([1., 2.]).as_gpu_tensor()
    shape = constant_op.constant([2, 1])
    reshaped = reshape(value, shape).as_cpu_tensor()
    self.assertAllEqual(reshaped.numpy(), [[1], [2]])

  def testFunctionHandlesInputsPlacedOnTheWrongDeviceGracefully(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')

    # The Reshape op requires the shape tensor to be placed in host memory.
    reshape = function.defun(array_ops.reshape)
    value = constant_op.constant([1., 2.]).as_gpu_tensor()
    shape = constant_op.constant([2, 1]).as_gpu_tensor()
    with self.assertRaises(errors.InvalidArgumentError):
      reshape(value, shape)

  def testDifferentiableFunctionNoneOutputs(self):

    @function.defun
    def my_function(x):
      return x, None

    def wrapper(x):
      return my_function(x)[0]

    g = backprop.gradients_function(wrapper, [0])(constant_op.constant(0.0))
    self.assertAllEqual(g[0].numpy(), 1.)

  def testNoneOutput(self):

    @function.defun
    def my_function(_):
      return None

    self.assertAllEqual(my_function(1), None)

  def testNestedFunctions(self):
    # TensorFlow function (which is what would be used in TensorFlow graph
    # construction).
    @tf_function.Defun(dtypes.int32, dtypes.int32)
    def add(a, b):
      return math_ops.add(a, b)

    @function.defun
    def add_one(x):
      return add(x, 1)

    self.assertAllEqual(3, add_one(constant_op.constant(2)).numpy())

  def testSequenceInputs(self):
    clip_by_global_norm = function.defun(clip_ops.clip_by_global_norm)
    t_list = [constant_op.constant(1.0), constant_op.constant(2.0)]
    clipped_list, global_norm = clip_by_global_norm(t_list,
                                                    constant_op.constant(.2))
    for t in clipped_list:
      self.assertTrue(isinstance(t, ops.Tensor))
    self.assertTrue(isinstance(global_norm, ops.Tensor))

  def testNestedSequenceInputs(self):

    def my_op(inputs):
      a, b, c = inputs
      e, f = b
      g, h = e
      return [a + a, [tuple([f + f, g + g]), h + h], c + c], a + f + g + h + c

    my_eager_op = function.defun(my_op)
    ret = my_eager_op([
        constant_op.constant(1), [(constant_op.constant(2),
                                   constant_op.constant(3)),
                                  constant_op.constant(4)],
        constant_op.constant(5)
    ])
    self.assertEqual(len(ret), 2)
    self.assertEqual(ret[0][0].numpy(), 2)
    self.assertEqual(ret[0][1][0][0].numpy(), 8)
    self.assertEqual(ret[0][1][0][1].numpy(), 4)
    self.assertTrue(isinstance(ret[0][1][0], tuple))
    self.assertEqual(ret[0][1][1].numpy(), 6)
    self.assertEqual(ret[0][2].numpy(), 10)
    self.assertEqual(ret[1].numpy(), 15)


if __name__ == '__main__':
  test.main()
