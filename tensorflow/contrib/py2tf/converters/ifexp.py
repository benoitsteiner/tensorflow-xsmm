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
"""Canonicalizes the ternary conditional operator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.py2tf.pyct import templates
from tensorflow.contrib.py2tf.pyct import transformer


class IfExp(transformer.Base):
  """Canonicalizes all IfExp nodes into plain conditionals."""

  def visit_IfExp(self, node):
    template = """
        py2tf_utils.run_cond(test, lambda: body, lambda: orelse)
    """
    desugared_ifexp = templates.replace_as_expression(
        template, test=node.test, body=node.body, orelse=node.orelse)
    return desugared_ifexp


def transform(node, context):
  """Desugar IfExp nodes into plain conditionals.

  Args:
     node: an AST node to transform
     context: a context object

  Returns:
     new_node: an AST with no IfExp nodes, only conditionals.
  """

  node = IfExp(context).visit(node)
  return node
