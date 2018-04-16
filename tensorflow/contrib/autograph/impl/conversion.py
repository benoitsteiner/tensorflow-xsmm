# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""High level conversion support."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.contrib.autograph import operators
from tensorflow.contrib.autograph import utils
from tensorflow.contrib.autograph.converters import asserts
from tensorflow.contrib.autograph.converters import break_statements
from tensorflow.contrib.autograph.converters import builtin_functions
from tensorflow.contrib.autograph.converters import call_trees
from tensorflow.contrib.autograph.converters import continue_statements
from tensorflow.contrib.autograph.converters import control_flow
from tensorflow.contrib.autograph.converters import decorators
from tensorflow.contrib.autograph.converters import ifexp
from tensorflow.contrib.autograph.converters import lists
from tensorflow.contrib.autograph.converters import logical_expressions
from tensorflow.contrib.autograph.converters import name_scopes
from tensorflow.contrib.autograph.converters import side_effect_guards
from tensorflow.contrib.autograph.converters import single_return
from tensorflow.contrib.autograph.impl import config
from tensorflow.contrib.autograph.impl import naming
from tensorflow.contrib.autograph.pyct import context
from tensorflow.contrib.autograph.pyct import inspect_utils
from tensorflow.contrib.autograph.pyct import parser
from tensorflow.contrib.autograph.pyct import qual_names
from tensorflow.contrib.autograph.pyct.static_analysis import activity
from tensorflow.contrib.autograph.pyct.static_analysis import live_values
from tensorflow.contrib.autograph.pyct.static_analysis import type_info
from tensorflow.contrib.autograph.utils import type_hints
from tensorflow.python.util import tf_inspect


# TODO(mdan): Might we not need any renaming at all?


class ConversionMap(object):
  """ConversionMap keeps track of converting function hierarchies.

  This object is mutable, and is updated as functions are converted.

  Attributes:
    recursive: Whether to recursively convert any functions that the decorator
        function may call.
    nocompile_decorators: tuple of decorator functions that toggle compilation
        off.
    dependency_cache: dict[object]: ast; maps original entities to their
        converted AST
    additional_imports: set(object); additional entities which for any reason
        cannot be attached after loading and need to be explicitly imported
        in the generated code
    name_map: dict[string]: string; maps original entities to the name of
        their converted counterparts
    api_module: A reference to the api module. The reference needs to be passed
        to avoid circular dependencies.
  """

  # TODO(mdan): Rename to ConversionContext, and pull in additional flags.

  def __init__(self, recursive, nocompile_decorators, partial_types,
               api_module):
    self.recursive = recursive
    self.nocompile_decorators = nocompile_decorators
    self.partial_types = partial_types if partial_types else ()
    self.dependency_cache = {}
    self.additional_imports = set()
    self.name_map = {}
    self.api_module = api_module

  def new_namer(self, namespace):
    return naming.Namer(namespace, self.recursive, self.name_map,
                        self.partial_types)

  def update_name_map(self, namer):
    for o, name in namer.renamed_calls.items():
      if o in self.name_map:
        if self.name_map[o] != name:
          raise ValueError(
              'Calls to %s were converted using multiple names (%s). This is '
              'possible when an entity with one of these names already '
              'existed. To fix, avoid using any of these names.')
      else:
        self.name_map[o] = name

  def add_to_cache(self, original_entity, converted_ast):
    self.dependency_cache[original_entity] = converted_ast


def is_whitelisted_for_graph(o):
  """Check whether an entity is whitelisted for use in graph mode.

  Examples of whitelisted entities include all members of the tensorflow
  package.

  Args:
    o: A Python entity.
  Returns:
    Boolean
  """
  m = tf_inspect.getmodule(o)
  for prefix, in config.DEFAULT_UNCOMPILED_MODULES:
    if m.__name__.startswith(prefix):
      return True
  return False


def entity_to_graph(o, conversion_map, arg_values, arg_types):
  """Compile a Python entity into equivalent TensorFlow.

  The function will also recursively compile all the entities that `o`
  references, updating `dependency_cache`.

  This function is reentrant, and relies on dependency_cache to avoid
  generating duplicate code.

  Args:
    o: A Python entity.
    conversion_map: A ConversionMap object.
    arg_values: A dict containing value hints for symbols like function
        parameters.
    arg_types: A dict containing type hints for symbols like function
        parameters.

  Returns:
    A tuple (ast, new_name, namespace):
        * ast: An AST representing an entity with interface equivalent to `o`,
            but which when executed it creates TF a graph.
        * new_name: The symbol name under which the new entity can be found.
        * namespace: A dict mapping all symbols visible to the converted entity,
            keyed by their symbol name.

  Raises:
    ValueError: if the entity type is not supported.
  """
  if tf_inspect.isclass(o):
    node, name, ns = class_to_graph(o, conversion_map)
  elif tf_inspect.isfunction(o):
    node, name, ns = function_to_graph(o, conversion_map, arg_values, arg_types)
  elif tf_inspect.ismethod(o):
    node, name, ns = function_to_graph(o, conversion_map, arg_values, arg_types)
  else:
    raise ValueError(
        'Entity "%s" has unsupported type "%s". Only functions and classes are '
        'supported for now.' % (o, type(o)))

  conversion_map.add_to_cache(o, node)
  if conversion_map.recursive:
    while True:
      candidate = None
      for obj in conversion_map.name_map.keys():
        if obj not in conversion_map.dependency_cache:
          candidate = obj
          break
      if candidate is None:
        break
      if (hasattr(candidate, 'im_class') and
          getattr(candidate, 'im_class') not in conversion_map.partial_types):
        # Class members are converted with their objects, unless they're
        # only converted partially.
        continue
      entity_to_graph(candidate, conversion_map, {}, {})

  return node, name, ns


def class_to_graph(c, conversion_map):
  """Specialization of `entity_to_graph` for classes."""
  converted_members = {}
  method_filter = lambda m: tf_inspect.isfunction(m) or tf_inspect.ismethod(m)
  members = tf_inspect.getmembers(c, predicate=method_filter)
  if not members:
    raise ValueError('Cannot convert %s: it has no member methods.' % c)

  class_namespace = {}
  for _, m in members:
    node, _, namespace = function_to_graph(
        m,
        conversion_map=conversion_map,
        arg_values={},
        arg_types={'self': (c.__name__, c)},
        owner_type=c)
    if class_namespace is None:
      class_namespace = namespace
    else:
      class_namespace.update(namespace)
    converted_members[m] = node
  namer = conversion_map.new_namer(class_namespace)
  class_name = namer.compiled_class_name(c.__name__, c)
  node = gast.ClassDef(
      class_name,
      bases=[],
      keywords=[],
      body=list(converted_members.values()),
      decorator_list=[])

  return node, class_name, class_namespace


def _add_reserved_symbol(namespace, name, entity):
  if name not in namespace:
    namespace[name] = entity
  elif namespace[name] != entity:
    raise ValueError('The name "%s" is reserved and may not be used.' % name)


def _add_self_references(namespace, api_module):
  # Manually add the utils namespace which may be used from generated code.
  _add_reserved_symbol(namespace, 'autograph_utils', utils)
  _add_reserved_symbol(namespace, '__ops', operators)
  # We also make reference to the api module for dynamic conversion, but
  # to avoid circular references we don't import it here.
  _add_reserved_symbol(namespace, 'autograph_api', api_module)


def function_to_graph(f, conversion_map, arg_values, arg_types,
                      owner_type=None):
  """Specialization of `entity_to_graph` for callable functions."""
  node, source = parser.parse_entity(f)
  node = node.body[0]

  namespace = inspect_utils.getnamespace(f)
  _add_self_references(namespace, conversion_map.api_module)
  namer = conversion_map.new_namer(namespace)

  ctx = context.EntityContext(
      namer=namer,
      source_code=source,
      source_file='<fragment>',
      namespace=namespace,
      arg_values=arg_values,
      arg_types=arg_types,
      owner_type=owner_type,
      recursive=conversion_map.recursive,
      type_annotation_func=type_hints.set_element_type)
  node, deps = node_to_graph(node, ctx, conversion_map.nocompile_decorators)

  # TODO(mdan): This somewhat duplicates the call rename logic in call_treest.py
  new_name, did_rename = namer.compiled_function_name(f.__name__, f, owner_type)
  if not did_rename:
    new_name = f.__name__
    if node.name != f.__name__:
      raise NotImplementedError('Strange corner case. Send us offending code!')

  node.name = new_name
  conversion_map.update_name_map(namer)
  # TODO(mdan): Use this at compilation.
  conversion_map.additional_imports.update(deps)

  return node, new_name, namespace


def _static_analysis_pass(node, ctx):
  node = qual_names.resolve(node)
  node = activity.resolve(node, ctx, None)
  node = live_values.resolve(node, ctx, config.PYTHON_LITERALS)
  node = type_info.resolve(node, ctx)
  return node


def node_to_graph(node, ctx, nocompile_decorators):
  """Convert Python code to equivalent TF graph mode code.

  Args:
    node: A Python AST node representing the code to convert.
    ctx: An EntityContext object.
    nocompile_decorators: A tuple containing decorators to be stripped from
        functions during conversion.

  Returns:
    A tuple (node, deps):
        * node: A Python ast node, representing the converted code.
        * deps: A set of strings, the fully qualified names of entity
            dependencies that this node has.
  """
  # TODO(mdan): Verify arguments for correctness.

  # TODO(mdan): Factor out common elements.
  # These include:
  #   * code move between blocks
  #   * visiting blocks in transformers

  # Certain steps, especially canonicalization, insert new symbols into the
  # tree, which must be accounted. Although less efficient, it is most robust
  # to re-run the analysis.

  node = _static_analysis_pass(node, ctx)

  # TODO(mdan): Clean this up.
  # Some intermediate analyses are not required, and some comments got orphaned.

  # Past this point, line numbers are no longer accurate so we ignore the
  # source.
  # TODO(mdan): Is it feasible to reconstruct intermediate source code?
  ctx.source_code = None
  node = ifexp.transform(node, ctx)
  node, deps = decorators.transform(node, nocompile_decorators)
  node = break_statements.transform(node, ctx)
  node = asserts.transform(node, ctx)

  # Note: sequencing continue canonicalization before for loop one avoids
  # dealing with the extra loop increment operation that the for
  # canonicalization creates.
  node = continue_statements.transform(node, ctx)
  ctx.namespace['len'] = len

  node = _static_analysis_pass(node, ctx)
  node = single_return.transform(node, ctx)

  node = _static_analysis_pass(node, ctx)
  node = lists.transform(node, ctx)
  node = builtin_functions.transform(node, ctx)

  node = _static_analysis_pass(node, ctx)
  node = call_trees.transform(node, ctx, config.DEFAULT_UNCOMPILED_MODULES,
                              nocompile_decorators)
  node = control_flow.transform(node, ctx)

  # control_flow may create new symbols and change scopes.
  node = _static_analysis_pass(node, ctx)
  node = logical_expressions.transform(node, ctx)
  node = side_effect_guards.transform(node, ctx)
  node = name_scopes.transform(node, ctx)

  return node, deps
