# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests l2hmc fit to 2D strongly correlated Gaussian executed eagerly."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy.random as npr
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.contrib.eager.python.examples.l2hmc import l2hmc


def get_default_hparams():
  return tf.contrib.training.HParams(
      x_dim=2,
      n_samples=200,
      n_steps=10,
      eps=.1,
      n_iters=10,
      learning_rate=.0003,
      n_warmup_iters=3)


# Relevant functions for benchmarking
def compute_loss(dynamics, x, scale=.1, eps=1e-4):
  """Compute loss defined in equation (8)."""

  z = tf.random_normal(tf.shape(x))
  x_, _, x_accept_prob, x_out = dynamics.apply_transition(x)
  z_, _, z_accept_prob, _ = dynamics.apply_transition(z)

  # Add eps for numerical stability; following released impl
  x_loss = tf.reduce_sum((x - x_)**2, axis=1) * x_accept_prob + eps
  z_loss = tf.reduce_sum((z - z_)**2, axis=1) * z_accept_prob + eps

  loss = tf.reduce_mean(
      (1. / x_loss + 1. / z_loss) * scale - (x_loss + z_loss) / scale, axis=0)

  return loss, x_out


def loss_and_grads(dynamics, x, loss_fn=compute_loss):
  """Obtain loss value and gradients."""

  with tf.GradientTape() as tape:
    loss_val, x_out = loss_fn(dynamics, x)
  grads = tape.gradient(loss_val, dynamics.variables)

  return loss_val, grads, x_out


def warmup(dynamics, optimizer, n_iters=1, n_samples=200, loss_fn=compute_loss):
  """Warmup optimization to reduce overhead."""

  samples = tf.random_normal(
      shape=[n_samples, dynamics.x_dim], dtype=tf.float32)

  for _ in range(n_iters):
    _, grads, samples = loss_and_grads(dynamics, samples, loss_fn=loss_fn)
    optimizer.apply_gradients(zip(grads, dynamics.variables))


def fit(dynamics,
        samples,
        optimizer,
        loss_fn=compute_loss,
        n_iters=5000,
        verbose=True,
        logdir=None,
        decay_lr=True):
  """Fit L2HMC sampler with given log-likelihood function."""

  if logdir:
    summary_writer = tf.contrib.summary.create_file_writer(logdir)

  for i in range(n_iters):
    loss, grads, samples = loss_and_grads(dynamics, samples, loss_fn=loss_fn)
    # TODO(lxuechen): Proper learning rate decay
    if decay_lr:
      grads = [grad * .96**(i // 1000) for grad in grads]
    optimizer.apply_gradients(zip(grads, dynamics.variables))
    if verbose:
      print("Iteration %d: loss %.4f" % (i, loss))

    if logdir:
      with summary_writer.as_default():
        with tf.contrib.summary.always_record_summaries():
          tf.contrib.summary.scalar("loss", loss)


class L2hmcTest(tf.test.TestCase):
  """Unit tests for l2hmc in both eager and graph mode."""

  def test_apply_transition(self):
    """Testing function `Dynamics.apply_transition` in graph and eager mode."""

    # Eager mode testing
    hparams = get_default_hparams()
    dynamics = l2hmc.Dynamics(
        x_dim=hparams.x_dim,
        loglikelihood_fn=l2hmc.get_scg_energy_fn(),
        n_steps=hparams.n_steps,
        eps=hparams.eps)
    samples = tf.random_normal(shape=[hparams.n_samples, hparams.x_dim])
    x_, v_, x_accept_prob, x_out = dynamics.apply_transition(samples)

    self.assertEqual(x_.shape, v_.shape)
    self.assertEqual(x_out.shape, samples.shape)
    self.assertEqual(x_.shape, x_out.shape)
    self.assertEqual(x_accept_prob.shape, (hparams.n_samples,))

    # Graph mode testing
    with tf.Graph().as_default():
      dynamics = l2hmc.Dynamics(
          x_dim=hparams.x_dim,
          loglikelihood_fn=l2hmc.get_scg_energy_fn(),
          n_steps=hparams.n_steps,
          eps=hparams.eps)
      x = tf.placeholder(tf.float32, shape=[None, hparams.x_dim])
      x_, v_, x_accept_prob, x_out = dynamics.apply_transition(x)
      samples = npr.normal(size=[hparams.n_samples, hparams.x_dim])

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        np_x_, np_v_, np_x_accept_prob, np_x_out = sess.run(
            [x_, v_, x_accept_prob, x_out], feed_dict={x: samples})

        self.assertEqual(np_x_.shape, np_v_.shape)
        self.assertEqual(samples.shape, np_x_out.shape)
        self.assertEqual(np_x_.shape, np_x_out.shape)
        self.assertEqual(np_x_accept_prob.shape, (hparams.n_samples,))


class L2hmcBenchmark(tf.test.Benchmark):
  """Eager and graph benchmarks for l2hmc."""

  def _get_energy_fn(self):
    """Get specific energy function according to FLAGS."""

    if FLAGS.energy_fn == "scg":
      energy_fn = l2hmc.get_scg_energy_fn()
    elif FLAGS.energy_fn == "multivariate_gaussian":
      energy_fn = l2hmc.get_multivariate_gaussian_energy_fn(x_dim=FLAGS.x_dim)
    else:
      raise ValueError("No such energy function %s" % FLAGS.energy_fn)

    return energy_fn

  def benchmark_graph(self):
    """Benchmark Graph performance."""

    hparams = get_default_hparams()
    tf.reset_default_graph()
    with tf.Graph().as_default():
      energy_fn = self._get_energy_fn()
      dynamics = l2hmc.Dynamics(
          x_dim=hparams.x_dim,
          loglikelihood_fn=energy_fn,
          n_steps=hparams.n_steps,
          eps=hparams.eps)
      x = tf.placeholder(tf.float32, shape=[None, hparams.x_dim])
      loss, x_out = compute_loss(dynamics, x)

      global_step = tf.Variable(0., name="global_step", trainable=False)
      learning_rate = tf.train.exponential_decay(
          hparams.learning_rate, global_step, 1000, 0.96, staircase=True)
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
      train_op = optimizer.minimize(loss, global_step=global_step)

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Warmup to reduce initialization effect when timing
        samples = npr.normal(size=[hparams.n_samples, hparams.x_dim])
        for _ in range(hparams.n_warmup_iters):
          _, _, _, _ = sess.run(
              [x_out, loss, train_op, learning_rate], feed_dict={x: samples})

        # Training
        start_time = time.time()
        for i in range(hparams.n_iters):
          samples, loss_np, _, _ = sess.run(
              [x_out, loss, train_op, learning_rate], feed_dict={x: samples})
          print("Iteration %d: loss %.4f" % (i, loss_np))
        wall_time = time.time() - start_time
        examples_per_sec = hparams.n_samples / wall_time

        self.report_benchmark(
            name="graph_train_%s" % ("gpu"
                                     if tf.test.is_gpu_available() else "cpu"),
            iters=hparams.n_iters,
            extras={"examples_per_sec": examples_per_sec},
            wall_time=wall_time)

  def benchmark_eager(self):
    self._benchmark_eager()

  def benchmark_eager_defun(self):
    self._benchmark_eager(defun=True)

  def _benchmark_eager(self, defun=False):
    """Benchmark Eager performance."""

    hparams = get_default_hparams()
    energy_fn = self._get_energy_fn()
    dynamics = l2hmc.Dynamics(
        x_dim=hparams.x_dim,
        loglikelihood_fn=energy_fn,
        n_steps=hparams.n_steps,
        eps=hparams.eps)
    optimizer = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)
    loss_fn = tfe.defun(compute_loss) if defun else compute_loss

    # Warmup to reduce initialization effect when timing
    warmup(dynamics, optimizer, n_iters=hparams.n_warmup_iters, loss_fn=loss_fn)

    # Training
    samples = tf.random_normal(
        shape=[hparams.n_samples, hparams.x_dim], dtype=tf.float32)
    start_time = time.time()
    fit(dynamics,
        samples,
        optimizer,
        loss_fn=loss_fn,
        n_iters=hparams.n_iters,
        decay_lr=True)
    wall_time = time.time() - start_time
    examples_per_sec = hparams.n_samples / wall_time

    self.report_benchmark(
        name="eager_train_%s%s" % ("gpu" if tf.test.is_gpu_available() else
                                   "cpu", "_defun" if defun else ""),
        iters=hparams.n_iters,
        extras={"examples_per_sec": examples_per_sec},
        wall_time=wall_time)

    del dynamics
    del loss_fn


if __name__ == "__main__":
  tf.flags.DEFINE_string("energy_fn", "scg",
                         ("The energy function/unnormalized log-probability. "
                          "Either be `scg` or `multivariate_gaussian`"))
  tf.flags.DEFINE_integer("x_dim", 2, "Dimensionality of observation space.")
  FLAGS = tf.flags.FLAGS
  tf.enable_eager_execution()
  tf.test.main()
