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
"""Utilities for exporting TensorFlow symbols to the API.

Exporting a function or a class:

To export a function or a class use tf_export decorator. For e.g.:
```python
@tf_export('foo', 'bar.foo')
def foo(...):
  ...
```

If a function is assigned to a variable, you can export it by calling
tf_export explicitly. For e.g.:
```python
foo = get_foo(...)
tf_export('foo', 'bar.foo')(foo)
```


Exporting a constant
```python
foo = 1
tf_export("consts.foo").export_constant(__name__, 'foo')
```
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import sys

from tensorflow.python.util import tf_decorator

ESTIMATOR_API_NAME = 'estimator'
TENSORFLOW_API_NAME = 'tensorflow'

_Attributes = collections.namedtuple(
    'ExportedApiAttributes', ['names', 'constants'])

# Attribute values must be unique to each API.
API_ATTRS = {
    TENSORFLOW_API_NAME: _Attributes(
        '_tf_api_names',
        '_tf_api_constants'),
    ESTIMATOR_API_NAME: _Attributes(
        '_estimator_api_names',
        '_estimator_api_constants')
}


class SymbolAlreadyExposedError(Exception):
  """Raised when adding API names to symbol that already has API names."""
  pass


class api_export(object):  # pylint: disable=invalid-name
  """Provides ways to export symbols to the TensorFlow API."""

  def __init__(self, *args, **kwargs):
    """Export under the names *args (first one is considered canonical).

    Args:
      *args: API names in dot delimited format.
      **kwargs: Optional keyed arguments.
          overrides: List of symbols that this is overriding
          (those overrided api exports will be removed). Note: passing overrides
          has no effect on exporting a constant.
          api_name: Name of the API you want to generate (e.g. `tensorflow` or
          `estimator`). Default is `tensorflow`.
    """
    self._names = args
    self._api_name = kwargs.get('api_name', TENSORFLOW_API_NAME)
    self._overrides = kwargs.get('overrides', [])

  def __call__(self, func):
    """Calls this decorator.

    Args:
      func: decorated symbol (function or class).

    Returns:
      The input function with _tf_api_names attribute set.

    Raises:
      SymbolAlreadyExposedError: Raised when a symbol already has API names
        and kwarg `allow_multiple_exports` not set.
    """
    api_names_attr = API_ATTRS[self._api_name].names

    # Undecorate overridden names
    for f in self._overrides:
      _, undecorated_f = tf_decorator.unwrap(f)
      delattr(undecorated_f, api_names_attr)

    _, undecorated_func = tf_decorator.unwrap(func)

    # Check for an existing api. We check if attribute name is in
    # __dict__ instead of using hasattr to verify that subclasses have
    # their own _tf_api_names as opposed to just inheriting it.
    if api_names_attr in undecorated_func.__dict__:
      raise SymbolAlreadyExposedError(
          'Symbol %s is already exposed as %s.' %
          (undecorated_func.__name__, getattr(
              undecorated_func, api_names_attr)))  # pylint: disable=protected-access
    setattr(undecorated_func, api_names_attr, self._names)
    return func

  def export_constant(self, module_name, name):
    """Store export information for constants/string literals.

    Export information is stored in the module where constants/string literals
    are defined.

    e.g.
    ```python
    foo = 1
    bar = 2
    tf_export("consts.foo").export_constant(__name__, 'foo')
    tf_export("consts.bar").export_constant(__name__, 'bar')
    ```

    Args:
      module_name: (string) Name of the module to store constant at.
      name: (string) Current constant name.
    """
    module = sys.modules[module_name]
    if not hasattr(module, API_ATTRS[self._api_name].constants):
      setattr(module, API_ATTRS[self._api_name].constants, [])
    # pylint: disable=protected-access
    getattr(module, API_ATTRS[self._api_name].constants).append(
        (self._names, name))


tf_export = functools.partial(api_export, api_name=TENSORFLOW_API_NAME)
estimator_export = functools.partial(tf_export, api_name=ESTIMATOR_API_NAME)
