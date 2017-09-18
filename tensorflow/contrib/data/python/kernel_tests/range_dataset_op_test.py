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
"""Test RangeDataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.contrib.data.python.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test


class RangeDatasetTest(test.TestCase):

  def tearDown(self):
    # Remove all checkpoint files.
    prefix = self._iterator_checkpoint_prefix()
    pattern = prefix + "*"
    files = gfile.Glob(pattern)
    map(gfile.Remove, files)

  def testStop(self):
    stop = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = dataset_ops.Dataset.range(stop).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op, feed_dict={stop: 5})
      for i in range(5):
        self.assertEqual(i, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testStartStop(self):
    start = array_ops.placeholder(dtypes.int64, shape=[])
    stop = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = dataset_ops.Dataset.range(start,
                                         stop).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op, feed_dict={start: 2, stop: 5})
      for i in range(2, 5):
        self.assertEqual(i, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testStartStopStep(self):
    start = array_ops.placeholder(dtypes.int64, shape=[])
    stop = array_ops.placeholder(dtypes.int64, shape=[])
    step = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = dataset_ops.Dataset.range(start, stop,
                                         step).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op, feed_dict={start: 2, stop: 10, step: 2})
      for i in range(2, 10, 2):
        self.assertEqual(i, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testZeroStep(self):
    start = array_ops.placeholder(dtypes.int64, shape=[])
    stop = array_ops.placeholder(dtypes.int64, shape=[])
    step = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = dataset_ops.Dataset.range(start, stop,
                                         step).make_initializable_iterator()
    init_op = iterator.initializer

    with self.test_session() as sess:
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(init_op, feed_dict={start: 2, stop: 10, step: 0})

  def testNegativeStep(self):
    start = array_ops.placeholder(dtypes.int64, shape=[])
    stop = array_ops.placeholder(dtypes.int64, shape=[])
    step = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = dataset_ops.Dataset.range(start, stop,
                                         step).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op, feed_dict={start: 2, stop: 10, step: -1})
      # This for loop is a no-op but will ensure that the implementation is
      # consistent with range if it ever changes.
      for i in range(2, 10, -1):
        self.assertEqual(i, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testStopLessThanStart(self):
    start = array_ops.placeholder(dtypes.int64, shape=[])
    stop = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = dataset_ops.Dataset.range(start,
                                         stop).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op, feed_dict={start: 10, stop: 2})
      # This for loop is a no-op but will ensure that the implementation is
      # consistent with range if it ever changes.
      for i in range(10, 2):
        self.assertEqual(i, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testStopLessThanStartWithPositiveStep(self):
    start = array_ops.placeholder(dtypes.int64, shape=[])
    stop = array_ops.placeholder(dtypes.int64, shape=[])
    step = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = dataset_ops.Dataset.range(start, stop,
                                         step).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op, feed_dict={start: 10, stop: 2, step: 2})
      # This for loop is a no-op but will ensure that the implementation is
      # consistent with range if it ever changes.
      for i in range(10, 2, 2):
        self.assertEqual(i, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testStopLessThanStartWithNegativeStep(self):
    start = array_ops.placeholder(dtypes.int64, shape=[])
    stop = array_ops.placeholder(dtypes.int64, shape=[])
    step = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = dataset_ops.Dataset.range(start, stop,
                                         step).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op, feed_dict={start: 10, stop: 2, step: -1})
      for i in range(10, 2, -1):
        self.assertEqual(i, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testEnumerateDataset(self):
    components = (["a", "b"], [1, 2], [37.0, 38])
    start = constant_op.constant(20, dtype=dtypes.int64)

    iterator = (dataset_ops.Dataset.from_tensor_slices(components).enumerate(
        start=start).make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual(dtypes.int64, get_next[0].dtype)
    self.assertEqual((), get_next[0].shape)
    self.assertEqual([tensor_shape.TensorShape([])] * 3,
                     [t.shape for t in get_next[1]])

    with self.test_session() as sess:
      sess.run(init_op)
      self.assertEqual((20, (b"a", 1, 37.0)), sess.run(get_next))
      self.assertEqual((21, (b"b", 2, 38.0)), sess.run(get_next))

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def _iterator_checkpoint_prefix(self):
    return os.path.join(self.get_temp_dir(), "iterator")

  def testSaveRestore(self):

    def _build_graph(start, stop):
      iterator = dataset_ops.Dataset.range(start,
                                           stop).make_initializable_iterator()
      init_op = iterator.initializer
      get_next = iterator.get_next()
      path = self._iterator_checkpoint_prefix()
      save_op = gen_dataset_ops.save_iterator(iterator._iterator_resource, path)
      restore_op = gen_dataset_ops.restore_iterator(iterator._iterator_resource,
                                                    path)
      return init_op, get_next, save_op, restore_op

    # Saving and restoring in different sessions.
    start = 2
    stop = 10
    break_point = 5
    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, _ = _build_graph(start, stop)
      with self.test_session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        for i in range(start, break_point):
          self.assertEqual(i, sess.run(get_next))
        sess.run(save_op)

    with ops.Graph().as_default() as g:
      init_op, get_next, _, restore_op = _build_graph(start, stop)
      with self.test_session(graph=g) as sess:
        sess.run(init_op)
        sess.run(restore_op)
        for i in range(break_point, stop):
          self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

    # Saving and restoring in same session.
    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, restore_op = _build_graph(start, stop)
      with self.test_session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        for i in range(start, break_point):
          self.assertEqual(i, sess.run(get_next))
        sess.run(save_op)
        sess.run(restore_op)
        for i in range(break_point, stop):
          self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  def testMultipleSaves(self):

    def _build_graph(start, stop):
      iterator = dataset_ops.Dataset.range(start,
                                           stop).make_initializable_iterator()
      init_op = iterator.initializer
      get_next = iterator.get_next()
      path = self._iterator_checkpoint_prefix()
      save_op = gen_dataset_ops.save_iterator(iterator._iterator_resource, path)
      restore_op = gen_dataset_ops.restore_iterator(iterator._iterator_resource,
                                                    path)
      return init_op, get_next, save_op, restore_op

    start = 2
    stop = 10
    break_point1 = 5
    break_point2 = 7

    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, _ = _build_graph(start, stop)
      with self.test_session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        for i in range(start, break_point1):
          self.assertEqual(i, sess.run(get_next))
        sess.run(save_op)

    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, restore_op = _build_graph(start, stop)
      with self.test_session(graph=g) as sess:
        sess.run(init_op)
        sess.run(restore_op)
        for i in range(break_point1, break_point2):
          self.assertEqual(i, sess.run(get_next))
        sess.run(save_op)

    break_point2 = 7
    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, restore_op = _build_graph(start, stop)
      with self.test_session(graph=g) as sess:
        sess.run(init_op)
        sess.run(restore_op)
        for i in range(break_point2, stop):
          self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  def testSaveRestoreWithRepeat(self):

    def _build_graph(start, stop, num_epochs):
      iterator = dataset_ops.Dataset.range(
          start, stop).repeat(num_epochs).make_initializable_iterator()
      init_op = iterator.initializer
      get_next = iterator.get_next()
      path = self._iterator_checkpoint_prefix()
      save_op = gen_dataset_ops.save_iterator(iterator._iterator_resource, path)
      restore_op = gen_dataset_ops.restore_iterator(iterator._iterator_resource,
                                                    path)
      return init_op, get_next, save_op, restore_op

    start = 2
    stop = 10
    num_epochs = 5
    break_range = 5
    break_epoch = 3
    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, restore_op = _build_graph(
          start, stop, num_epochs)
      with self.test_session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        # Note: There is no checkpoint saved currently so a NotFoundError is
        # raised.
        with self.assertRaises(errors.NotFoundError):
          sess.run(restore_op)
        for _ in range(break_epoch - 1):
          for i in range(start, stop):
            self.assertEqual(i, sess.run(get_next))
        for i in range(start, break_range):
          self.assertEqual(i, sess.run(get_next))
        sess.run(save_op)

    with ops.Graph().as_default() as g:
      init_op, get_next, _, restore_op = _build_graph(start, stop, num_epochs)
      with self.test_session(graph=g) as sess:
        sess.run(init_op)
        sess.run(restore_op)
        for i in range(break_range, stop):
          self.assertEqual(i, sess.run(get_next))
        for _ in range(break_epoch, num_epochs):
          for i in range(start, stop):
            self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  def testSaveRestoreExhaustedIterator(self):

    def _build_graph(start, stop, num_epochs):
      iterator = dataset_ops.Dataset.range(
          start, stop).repeat(num_epochs).make_initializable_iterator()
      init_op = iterator.initializer
      get_next = iterator.get_next()
      path = self._iterator_checkpoint_prefix()
      save_op = gen_dataset_ops.save_iterator(iterator._iterator_resource, path)
      restore_op = gen_dataset_ops.restore_iterator(iterator._iterator_resource,
                                                    path)
      return init_op, get_next, save_op, restore_op

    start = 2
    stop = 10
    num_epochs = 5
    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, restore_op = _build_graph(
          start, stop, num_epochs)
      with self.test_session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        # Note: There is no checkpoint saved currently so a NotFoundError is
        # raised.
        with self.assertRaises(errors.NotFoundError):
          sess.run(restore_op)
        for _ in range(num_epochs):
          for i in range(start, stop):
            self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)
        sess.run(save_op)

    with ops.Graph().as_default() as g:
      init_op, get_next, _, restore_op = _build_graph(start, stop, num_epochs)
      with self.test_session(graph=g) as sess:
        sess.run(init_op)
        sess.run(restore_op)
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)


if __name__ == "__main__":
  test.main()
