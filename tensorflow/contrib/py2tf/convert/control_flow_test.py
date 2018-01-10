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
"""Tests for control_flow module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.py2tf.convert import control_flow
from tensorflow.contrib.py2tf.pyct import compiler
from tensorflow.contrib.py2tf.pyct import parser
from tensorflow.contrib.py2tf.pyct.static_analysis import access
from tensorflow.contrib.py2tf.pyct.static_analysis import live_values
from tensorflow.contrib.py2tf.pyct.static_analysis import type_info
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import test


class TestNamer(control_flow.SymbolNamer):

  def new_symbol(self, name_root, _):
    return name_root


class ControlFlowTest(test.TestCase):

  def _parse_and_analyze(self, test_fn, namespace):
    node = parser.parse_object(test_fn)
    node = access.resolve(node)
    node = live_values.resolve(node, namespace, {})
    node = type_info.resolve(node, None)
    return node

  def test_simple_while(self):

    def test_fn(n):
      i = 0
      s = 0
      while i < n:
        s += i
        i += 1
      return s, i, n

    node = self._parse_and_analyze(test_fn, {})
    node = control_flow.transform(node, TestNamer())
    result = compiler.ast_to_object(node)
    setattr(result, 'tf', control_flow_ops)

    with self.test_session() as sess:
      self.assertEqual((10, 5, 5),
                       sess.run(result.test_fn(constant_op.constant(5))))

  def test_while_single_var(self):

    def test_fn(n):
      while n > 0:
        n -= 1
      return n

    node = self._parse_and_analyze(test_fn, {})
    node = control_flow.transform(node, TestNamer())
    result = compiler.ast_to_object(node)
    setattr(result, 'tf', control_flow_ops)

    with self.test_session() as sess:
      self.assertEqual(0, sess.run(result.test_fn(constant_op.constant(5))))


if __name__ == '__main__':
  test.main()
