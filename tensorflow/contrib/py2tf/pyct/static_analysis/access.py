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
"""Access information (reads, writes) resolution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.contrib.py2tf.pyct import anno

# TODO(mdan): Add support for PY3 (e.g. Param vs arg).


class Scope(object):
  """Encloses local symbol definition and usage information.

  This can track for instance whether a symbol is modified in the current scope.
  Note that scopes do not necessarily align with Python's scopes. For example,
  the body of an if statement may be considered a separate scope.

  Attributes:
    modified: identifiers modified in this scope
    created: identifiers created in this scope
    used: identifiers referenced in this scope
  """

  def __init__(self, parent, isolated=True):
    """Create a new scope.

    Args:
      parent: A Scope or None.
      isolated: Whether the scope is isolated, that is, whether variables
          created in this scope should be visible to the parent scope.
    """
    self.isolated = isolated
    self.parent = parent
    self.modified = set()
    self.created = set()
    self.used = set()

  def __repr__(self):
    return 'Scope{r=%s, c=%s, w=%s}' % (tuple(self.used), tuple(self.created),
                                        tuple(self.modified))

  def has(self, name):
    if name in self.modified:
      return True
    elif self.parent is not None:
      return self.parent.has(name)
    return False

  def mark_read(self, name):
    self.used.add(name)
    if self.parent is not None and self.parent.has(
        name) and name not in self.created:
      self.parent.mark_read(name)

  def mark_write(self, name):
    self.modified.add(name)
    if self.parent is not None and self.parent.has(name):
      self.parent.mark_write(name)
    else:
      if not self.isolated:
        self.parent.mark_write(name)
      self.created.add(name)


class AccessResolver(gast.NodeTransformer):
  """Annotates nodes with local scope information. See Scope."""

  def __init__(self):
    self.scope = Scope(None)

  def visit_Name(self, node):
    # TODO(mdan): This is insufficient for object fields, e.g. hp.learning_rate.
    self.generic_visit(node)
    if isinstance(node.ctx, gast.Store):
      self.scope.mark_write(node.id)
    elif isinstance(node.ctx, gast.Load):
      anno.setanno(node, 'is_local', self.scope.has(node.id))
      self.scope.mark_read(node.id)
    elif isinstance(node.ctx, gast.Param):
      # Param contexts appear in function defs, so they have the meaning of
      # defining a variable.
      # TODO(mdan): This bay be incorrect with nested functions.
      # For nested functions, we'll have to add the notion of hiding args from
      # the parent scope, not writing to them.
      self.scope.mark_write(node.id)
    else:
      raise ValueError('Unknown context %s for node %s.' % (type(node.ctx),
                                                            node.id))
    return node

  def visit_Print(self, node):
    current_scope = self.scope
    args_scope = Scope(current_scope)
    self.scope = args_scope
    for n in node.values:
      self.visit(n)
    anno.setanno(node, 'args_scope', args_scope)
    self.scope = current_scope
    return node

  def visit_Call(self, node):
    current_scope = self.scope
    args_scope = Scope(current_scope)
    self.scope = args_scope
    for n in node.args:
      self.visit(n)
    # TODO(mdan): Account starargs, kwargs
    for n in node.keywords:
      self.visit(n)
    anno.setanno(node, 'args_scope', args_scope)
    self.scope = current_scope
    self.visit(node.func)
    return node

  def visit_For(self, node):
    raise NotImplementedError()

  def visit_While(self, node):
    self.visit(node.test)
    current_scope = self.scope
    anno.setanno(node, 'parent_scope', current_scope)
    body_scope = Scope(current_scope, isolated=False)
    self.scope = body_scope
    for n in node.body:
      self.visit(n)
    anno.setanno(node, 'body_scope', body_scope)
    if node.orelse:
      raise NotImplementedError()
      # TODO(mdan): Add support for orelse.
    self.scope = current_scope
    return node


def resolve(node):
  return AccessResolver().visit(node)
