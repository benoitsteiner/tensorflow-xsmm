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
"""Canonicalizes for loops into while loops.

This canonicalizer uses the len function on its argument. That should be
converted to a tf.shape separately.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.py2tf.pyct import anno
from tensorflow.contrib.py2tf.pyct import templates
from tensorflow.contrib.py2tf.pyct import transformer
from tensorflow.contrib.py2tf.pyct.static_analysis.annos import NodeAnno


class ForLoopCanonicalizationTransformer(transformer.Base):
  """Canonicalizes for loops (e.g. into while loops)."""

  def __init__(self, context):
    super(ForLoopCanonicalizationTransformer, self).__init__(context)

  def visit_For(self, node):
    self.generic_visit(node)
    body_scope = anno.getanno(node, NodeAnno.BODY_SCOPE)
    i_var = self.context.namer.new_symbol('i', body_scope.referenced)
    n_var = self.context.namer.new_symbol('n', body_scope.referenced)
    iterated_var = self.context.namer.new_symbol('iterated',
                                                 body_scope.referenced)
    # TODO(mdan): Use TensorListFromTensor(loop_iter) here.
    if anno.hasanno(node, 'extra_cond'):
      template = """
        i = 0
        iterated = loop_iter
        n = len(iterated)
        while i < n and extra_cond:
          target = iterated[i]
          body
          i += 1
      """
      return templates.replace(
          template,
          loop_iter=node.iter,
          target=node.target,
          body=node.body,
          i=i_var,
          n=n_var,
          iterated=iterated_var,
          extra_cond=anno.getanno(node, 'extra_cond'))
    else:
      template = """
        i = 0
        iterated = loop_iter
        n = len(iterated)
        while i < n:
          target = iterated[i]
          body
          i += 1
      """
      repl = templates.replace(
          template,
          loop_iter=node.iter,
          target=node.target,
          body=node.body,
          i=i_var,
          n=n_var,
          iterated=iterated_var)
      return repl

  def visit_Continue(self, node):
    assert False, 'continue statement should be desugared at this point'

  def visit_Break(self, node):
    assert False, 'break statement should be desugared at this point'


def transform(node, context):
  return ForLoopCanonicalizationTransformer(context).visit(node)
