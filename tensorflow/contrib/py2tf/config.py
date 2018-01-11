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
"""Global configuration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

PYTHON_LITERALS = {
    'None': None,
}

DEFAULT_UNCOMPILED_MODULES = set((
    ('tensorflow',),
))

NO_SIDE_EFFECT_CONSTRUCTORS = set(('tensorflow',))

# TODO(mdan): Also allow controlling the generated names (for testability).
COMPILED_IMPORT_STATEMENTS = (
    'from contextlib import contextmanager',
    'import tensorflow as tf',
)
