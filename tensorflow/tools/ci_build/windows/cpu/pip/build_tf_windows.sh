#!/bin/bash
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
#
# This script assumes the standard setup on tensorflow Jenkins windows machines.
# It is NOT guaranteed to work on any other machine. Use at your own risk!
#
# REQUIREMENTS:
# * All installed in standard locations:
#   - JDK8, and JAVA_HOME set.
#   - Microsoft Visual Studio 2015 Community Edition
#   - Msys2
#   - Anaconda3
# * Bazel windows executable copied as "bazel.exe" and included in PATH.

# All commands shall pass, and all should be visible.
set -x
set -e

# This script is under <repo_root>/tensorflow/tools/ci_build/windows/cpu/pip/
# Change into repository root.
script_dir=$(dirname $0)
cd ${script_dir%%tensorflow/tools/ci_build/windows/cpu/pip}.

# Setting up the environment variables Bazel and ./configure needs
source "tensorflow/tools/ci_build/windows/bazel/common_env.sh" \
  || { echo "Failed to source common_env.sh" >&2; exit 1; }

# load bazel_test_lib.sh
source "tensorflow/tools/ci_build/windows/bazel/bazel_test_lib.sh" \
  || { echo "Failed to source bazel_test_lib.sh" >&2; exit 1; }

# Recreate an empty bazelrc file under source root
export TMP_BAZELRC=.tmp.bazelrc
rm -f "${TMP_BAZELRC}"
touch "${TMP_BAZELRC}"

function cleanup {
  # Remove all options in .tmp.bazelrc
  echo "" > "${TMP_BAZELRC}"
}
trap cleanup EXIT

skip_test=0
release_build=0

for ARG in "$@"; do
  if [[ "$ARG" == --skip_test ]]; then
    skip_test=1
  elif [[ "$ARG" == --enable_gcs_remote_cache ]]; then
    set_gcs_remote_cache_options
  elif [[ "$ARG" == --release_build ]]; then
    release_build=1
  fi
done

if [[ "$release_build" != 1 ]]; then
  # --define=override_eigen_strong_inline=true speeds up the compiling of conv_grad_ops_3d.cc and conv_ops_3d.cc
  # by 20 minutes. See https://github.com/tensorflow/tensorflow/issues/10521
  # Because this hurts the performance of TF, we don't enable it in release build.
  echo "build --define=override_eigen_strong_inline=true" >> "${TMP_BAZELRC}"
fi

# The host and target platforms are the same in Windows build. So we don't have
# to distinct them. This helps avoid building the same targets twice.
echo "build --distinct_host_configuration=false" >> "${TMP_BAZELRC}"

# Enable short object file path to avoid long path issue on Windows.
echo "startup --output_user_root=${TMPDIR}" >> "${TMP_BAZELRC}"

if ! grep -q "import %workspace%/${TMP_BAZELRC}" .bazelrc; then
  echo "import %workspace%/${TMP_BAZELRC}" >> .bazelrc
fi

run_configure_for_cpu_build

bazel build --announce_rc --config=opt tensorflow/tools/pip_package:build_pip_package || exit $?

if [[ "$skip_test" == 1 ]]; then
  exit 0
fi

# Create a python test directory to avoid package name conflict
PY_TEST_DIR="py_test_dir"
create_python_test_dir "${PY_TEST_DIR}"

./bazel-bin/tensorflow/tools/pip_package/build_pip_package "$PWD/${PY_TEST_DIR}"

# Running python tests on Windows needs pip package installed
PIP_NAME=$(ls ${PY_TEST_DIR}/tensorflow-*.whl)
reinstall_tensorflow_pip ${PIP_NAME}

# NUMBER_OF_PROCESSORS is predefined on Windows
N_JOBS="${NUMBER_OF_PROCESSORS}"

# Define no_tensorflow_py_deps=true so that every py_test has no deps anymore,
# which will result testing system installed tensorflow
bazel test --announce_rc --config=opt -k --test_output=errors \
  --define=no_tensorflow_py_deps=true --test_lang_filters=py \
  --test_tag_filters=-no_pip,-no_windows,-no_oss \
  --build_tag_filters=-no_pip,-no_windows,-no_oss --build_tests_only \
  --jobs="${N_JOBS}" --test_timeout="300,450,1200,3600" \
  --flaky_test_attempts=3 \
  //${PY_TEST_DIR}/tensorflow/python/... \
  //${PY_TEST_DIR}/tensorflow/contrib/...
