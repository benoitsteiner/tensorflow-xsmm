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
"""Tests for core."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import numpy as np

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.eager import execute
from tensorflow.python.eager import tensor
from tensorflow.python.eager import test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util


def truncated_normal(shape):
  return execute.execute(
      'TruncatedNormal',
      1,
      inputs=[shape],
      attrs=('dtype', dtypes.float32.as_datatype_enum, 'T',
             shape.dtype.as_datatype_enum, 'seed', 0, 'seed2', 0))[0]


class TFETest(test_util.TensorFlowTestCase):

  def testContext(self):
    ctx = context.Context()
    self.assertFalse(ctx.in_graph_mode())
    self.assertTrue(ctx.in_eager_mode())
    self.assertEqual('', ctx.scope_name)
    self.assertEqual(-1, ctx._device_index)  # pylint: disable=protected-access
    self.assertFalse(ctx.recording_summaries)
    self.assertIsNone(ctx.summary_writer_resource)
    del ctx

  def testDefaultContext(self):
    orig = context.get_default_context()
    self.assertIs(context.get_default_context(), orig)
    c0 = context.Context()
    self.assertIs(context.get_default_context(), orig)
    context_manager_0 = c0.as_default()
    self.assertIs(context.get_default_context(), orig)
    with context_manager_0 as c0:
      self.assertIs(context.get_default_context(), c0)
      with context.Context().as_default() as c1:
        self.assertIs(context.get_default_context(), c1)
      self.assertIs(context.get_default_context(), c0)
    self.assertIs(context.get_default_context(), orig)

  def testContextWithThreads(self):

    def run_fn(ctx1):
      ctx2 = context.get_default_context()
      # Default context created in different threads are different.
      self.assertIsNot(ctx1, ctx2)
      # Check that default values of the context created in a different thread
      # are set correctly.
      self.assertFalse(ctx2.in_graph_mode())
      self.assertTrue(ctx2.in_eager_mode())
      self.assertEqual('', ctx2.scope_name)
      self.assertEqual(-1, ctx2._device_index)  # pylint: disable=protected-access
      self.assertFalse(ctx2.recording_summaries)
      self.assertIsNone(ctx2.summary_writer_resource)

    ctx1 = context.get_default_context()
    t = threading.Thread(target=run_fn, args=(ctx1,))
    t.start()
    t.join()

  def testScalarTensor(self):
    t = tensor.Tensor(3)
    self.assertEqual(t.numpy(), tensor.Tensor(np.array(3)).numpy())
    self.assertEqual(dtypes.int32, t.dtype)
    self.assertEqual(0, t.shape.ndims)
    self.assertAllEqual([], t.shape.as_list())

  def testTensorAndNumpyMatrix(self):
    expected = np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)
    actual = tensor.Tensor([[1.0, 2.0], [3.0, 4.0]])
    self.assertAllEqual(expected, actual.numpy())
    self.assertEqual(np.float32, actual.numpy().dtype)
    self.assertEqual(dtypes.float32, actual.dtype)
    self.assertAllEqual([2, 2], actual.shape.as_list())

  def testFloatDowncast(self):
    # Unless explicitly specified, float64->float32
    t = tensor.Tensor(3.0)
    self.assertEqual(dtypes.float32, t.dtype)
    t = tensor.Tensor(3.0, dtype=dtypes.float64)
    self.assertEqual(dtypes.float64, t.dtype)

  def testBool(self):
    t = tensor.Tensor(False)
    if t:
      self.assertFalse(True)

  def testIntDowncast(self):
    t = tensor.Tensor(3)
    self.assertEqual(dtypes.int32, t.dtype)
    t = tensor.Tensor(3, dtype=dtypes.int64)
    self.assertEqual(dtypes.int64, t.dtype)
    t = tensor.Tensor(2**33)
    self.assertEqual(dtypes.int64, t.dtype)

  def testTensorCreationFailure(self):
    with self.assertRaises(Exception):
      # Should fail because the each row of the Python object has a different
      # number of columns.
      self.assertEqual(None, tensor.Tensor([[1], [1, 2]]))

  def testTensorPlacement(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')

    x = tensor.Tensor(1.).as_gpu_tensor()
    with context.device('gpu:0'):
      y = tensor.Tensor(2.)
    # Add would fail if t2 were not on GPU
    result = execute.execute(
        'Add', 1, inputs=[x, y],
        attrs=('T', x.dtype.as_datatype_enum))[0].as_cpu_tensor().numpy()
    self.assertEqual(3, result)

  def testNumpyOrderHandling(self):
    n = np.array([[1, 2], [3, 4]], order='F')
    t = tensor.Tensor(n)
    self.assertAllEqual([[1, 2], [3, 4]], t.numpy())

  def testCopyBetweenDevices(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')

    x = tensor.Tensor([[1., 2.], [3., 4.]])
    x = x.as_cpu_tensor()
    x = x.as_gpu_tensor()
    x = x.as_gpu_tensor()
    x = x.as_cpu_tensor()

    # Invalid device
    with self.assertRaises(errors.InvalidArgumentError):
      x.as_gpu_tensor(context.context().num_gpus() + 1)

  def testNumpyForceCPU(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')

    cpu = tensor.Tensor([[1., 2.], [3., 4.]])
    c2g = cpu.as_gpu_tensor()
    self.assertAllEqual(c2g.numpy(), cpu.numpy())

  def testCopyFromCPUToCPU(self):
    ta = tensor.Tensor([[1, 2], [3, 4]])
    tb = ta.as_cpu_tensor()

    self.assertNotEqual(ta._handle, tb._handle)
    self.assertAllEqual(ta.numpy(), tb.numpy())

  def testRegisterExceptionClass(self):
    with self.assertRaises(TypeError):
      pywrap_tensorflow.TFE_Py_RegisterExceptionClass(str)
    pywrap_tensorflow.TFE_Py_RegisterExceptionClass(core._NotOkStatusException)  # pylint: disable=protected-access

  # TODO(agarwal): add tests passing incorrect typed values to attrs.
  def testExecuteBasic(self):
    three = tensor.Tensor(3)
    five = tensor.Tensor(5)
    product = execute.execute(
        'Mul',
        num_outputs=1,
        inputs=[three, five],
        attrs=('T', three.dtype.as_datatype_enum))[0]
    self.assertEqual(15, product.numpy())

  def testExecuteTooManyNumOutputs(self):
    # num_outputs provided is 50, but only one output is produced.
    # That should be okay.
    product = execute.execute(
        'Mul',
        num_outputs=50,
        inputs=[tensor.Tensor(3), tensor.Tensor(5)],
        attrs=('T', dtypes.int32.as_datatype_enum))[0]
    self.assertEqual(15, product.numpy())

  def testMatMulGPU(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')
    three = tensor.Tensor([[3.]]).as_gpu_tensor()
    five = tensor.Tensor([[5.]]).as_gpu_tensor()
    product = execute.execute(
        'MatMul',
        num_outputs=1,
        inputs=[three, five],
        attrs=('transpose_a', False, 'transpose_b', False, 'T',
               three.dtype.as_datatype_enum))[0]
    self.assertEqual([[15.0]], product.numpy())

  def testExecuteStringAttr(self):
    checked_three = execute.execute(
        'CheckNumerics',
        num_outputs=1,
        inputs=[tensor.Tensor(3.)],
        attrs=('message', 'just checking', 'T',
               dtypes.float32.as_datatype_enum))[0]
    self.assertEqual([[3]], checked_three.numpy())

  def testExecuteStringAttrBadValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      _ = execute.execute(
          'CheckNumerics',
          num_outputs=1,
          inputs=[tensor.Tensor(3.)],
          attrs=('message', 1, 'T', dtypes.float32.as_datatype_enum))

  def testExecuteFloatAttr(self):
    almost_equal = execute.execute(
        'ApproximateEqual',
        num_outputs=1,
        inputs=[tensor.Tensor(3.0), tensor.Tensor(2.9)],
        attrs=('tolerance', 0.3, 'T', dtypes.float32.as_datatype_enum))[0]
    self.assertTrue(almost_equal.numpy())

  def testExecuteFloatAttrBadValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      _ = execute.execute(
          'ApproximateEqual',
          num_outputs=1,
          inputs=[tensor.Tensor(3.0), tensor.Tensor(2.9)],
          attrs=('tolerance', '0.3', 'T', dtypes.float32.as_datatype_enum))

  def testExecuteIntAttr(self):
    total = execute.execute(
        'AddN',
        num_outputs=1,
        inputs=[tensor.Tensor(3), tensor.Tensor(4)],
        attrs=('T', dtypes.int32.as_datatype_enum, 'N', 2))[0]
    self.assertEqual(7, total.numpy())

  def testExecuteIntAttrBadValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      _ = execute.execute(
          'AddN',
          num_outputs=1,
          inputs=[tensor.Tensor(3), tensor.Tensor(4)],
          attrs=('T', dtypes.int32.as_datatype_enum, 'N', '2'))

  # Looks like we don't have an existing op with list(bool) attrs.
  def testExecuteBoolAttr(self):
    product = execute.execute(
        'MatMul',
        num_outputs=1,
        inputs=[tensor.Tensor([[3]]),
                tensor.Tensor([[5]])],
        attrs=('transpose_a', True, 'transpose_b', False, 'T',
               dtypes.int32.as_datatype_enum))[0]
    self.assertEqual([[15]], product.numpy())

  def testExecuteShapeAttr(self):
    execute.execute(
        'VarHandleOp',
        num_outputs=1,
        inputs=[],
        attrs=('shape', [1, 2], 'dtype', dtypes.int32.as_datatype_enum,
               'container', '', 'shared_name', ''))

  def testExecuteShapeAttrBadValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      execute.execute(
          'VarHandleOp',
          num_outputs=1,
          inputs=[],
          attrs=('shape', 1, 'dtype', dtypes.int32.as_datatype_enum,
                 'container', '', 'shared_name', ''))

  def testExecuteListStringAttr(self):
    execute.execute(
        'TensorSummary',
        num_outputs=1,
        inputs=[tensor.Tensor(3.0)],
        attrs=('T', dtypes.float32.as_datatype_enum, 'description',
               'tensor_summary', 'labels', ['3',
                                            'summary'], 'display_name', 'test'))

  def testExecuteListStringAttrBadValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      execute.execute(
          'TensorSummary',
          num_outputs=1,
          inputs=[tensor.Tensor(3.0)],
          attrs=('T', dtypes.float32.as_datatype_enum, 'description', '',
                 'labels', 3, 'display_name', 'test'))

  def testExecuteListStringAttrBadListValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      execute.execute(
          'TensorSummary',
          num_outputs=1,
          inputs=[tensor.Tensor(3.0)],
          attrs=('T', dtypes.float32.as_datatype_enum, 'description', '',
                 'labels', [3], 'display_name', 'test'))

  def testExecuteListFloatAttr(self):
    b = execute.execute(
        'Bucketize',
        num_outputs=1,
        inputs=[tensor.Tensor([3.0, 5.0, 7.0])],
        attrs=('T', dtypes.float32.as_datatype_enum, 'boundaries', [4.0,
                                                                    6.0]))[0]
    self.assertAllEqual([0, 1, 2], b.numpy())

  def testExecuteListFloatAttrBadValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      execute.execute(
          'Bucketize',
          num_outputs=1,
          inputs=[tensor.Tensor([3.0, 5.0, 7.0])],
          attrs=('T', dtypes.float32.as_datatype_enum, 'boundaries', 4.0))

  def testExecuteListFloatAttrBadListValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      execute.execute(
          'Bucketize',
          num_outputs=1,
          inputs=[tensor.Tensor([3.0, 5.0, 7.0])],
          attrs=('T', dtypes.float32.as_datatype_enum, 'boundaries',
                 ['4.0', '6.0']))

  def testExecuteListIntAttr(self):
    b = execute.execute(
        'Squeeze',
        num_outputs=1,
        inputs=[tensor.Tensor([[[3.0]]])],
        attrs=('T', dtypes.float32.as_datatype_enum, 'squeeze_dims', [0, 2]))[0]
    self.assertAllEqual([3], b.numpy())

  def testExecuteListIntAttrBadValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      execute.execute(
          'Squeeze',
          num_outputs=1,
          inputs=[tensor.Tensor([[[3.0]]])],
          attrs=('T', dtypes.float32.as_datatype_enum, 'squeeze_dims', 0))

  def testExecuteListIntAttrBadListValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      execute.execute(
          'Squeeze',
          num_outputs=1,
          inputs=[tensor.Tensor([[[3.0]]])],
          attrs=('T', dtypes.float32.as_datatype_enum, 'squeeze_dims',
                 ['0', '2']))

  def testExecuteListTypeListShapeAttr(self):
    execute.execute(
        'Barrier',
        num_outputs=1,
        inputs=[],
        attrs=('component_types', [dtypes.float64.as_datatype_enum], 'shapes',
               [[1, 2]], 'capacity', -1, 'container', '', 'shared_name', ''))

  def testExecuteListTypeAttrBadValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      execute.execute(
          'Barrier',
          num_outputs=1,
          inputs=[],
          attrs=('component_types', dtypes.float64.as_datatype_enum, 'shapes',
                 [[1, 2]], 'capacity', -1, 'container', '', 'shared_name', ''))

  def testExecuteListTypeAttrBadListValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      execute.execute(
          'Barrier',
          num_outputs=1,
          inputs=[],
          attrs=('component_types', '1', 'shapes', [[1, 2]], 'capacity', -1,
                 'container', '', 'shared_name', ''))

  def testExecuteListShapeAttrBadValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      execute.execute(
          'Barrier',
          num_outputs=1,
          inputs=[],
          attrs=('component_types', [dtypes.float64.as_datatype_enum], 'shapes',
                 [1, 2], 'capacity', -1, 'container', '', 'shared_name', ''))

  def testExecuteListShapeAttrBadListValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      execute.execute(
          'Barrier',
          num_outputs=1,
          inputs=[],
          attrs=('component_types', [dtypes.float64.as_datatype_enum], 'shapes',
                 [1], 'capacity', -1, 'container', '', 'shared_name', ''))

  def testExecuteMultipleOutputs(self):
    split_dim = 1
    value = [[0, 1, 2], [3, 4, 5]]
    x1, x2, x3 = execute.execute(
        'Split',
        num_outputs=3,
        inputs=[tensor.Tensor(split_dim),
                tensor.Tensor(value)],
        attrs=('num_split', 3, 'T', dtypes.int32.as_datatype_enum))
    self.assertAllEqual([[0], [3]], x1.numpy())
    self.assertAllEqual([[1], [4]], x2.numpy())
    self.assertAllEqual([[2], [5]], x3.numpy())

  def testExecuteBadNumOutputsArgument(self):
    with self.assertRaises(TypeError):
      execute.execute(
          'Relu', [],
          inputs=[tensor.Tensor(3.0)],
          attrs=('T', dtypes.float32.as_datatype_enum))

  def testExecuteUnknownOp(self):
    with self.assertRaises(errors.NotFoundError):
      execute.execute('BlahBlahBlah', num_outputs=1, inputs=[], attrs=None)

  def testExecuteUnknownAttr(self):
    with self.assertRaises(errors.InvalidArgumentError):
      execute.execute(
          'Identity',
          num_outputs=1,
          inputs=[tensor.Tensor(3)],
          attrs=('T', dtypes.int32.as_datatype_enum, 'unknown_attr', 'blah'))

  def testComposition(self):

    def add(x, y):
      return execute.execute(
          'Add',
          num_outputs=1,
          inputs=[x, y],
          attrs=('T', dtypes.int32.as_datatype_enum))[0]

    x = tensor.Tensor(1)
    three_x = add(add(x, x), x)
    self.assertEquals(dtypes.int32, three_x.dtype)
    self.assertEquals(3, three_x.numpy())

  def testOperationWithNoInputsRunsOnDevice(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')
    shape = tensor.Tensor([], dtype=dtypes.int32)

    # x: Run the "TruncatedNormal" op CPU and copy result to GPU.
    x = truncated_normal(shape).as_gpu_tensor()
    # y: Explicitly run the "TruncatedNormal" op on GPU.
    with context.device('gpu:0'):
      y = truncated_normal(shape)
    # Add would fail if x and y were not on the same device.
    execute.execute(
        'Add', 1, inputs=[x, y], attrs=('T', x.dtype.as_datatype_enum))

  def testInvalidDevice(self):
    with self.assertRaises(ValueError):
      with context.device('pu:0'):
        _ = tensor.Tensor(1)


if __name__ == '__main__':
  test.main()
