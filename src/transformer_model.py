"""Transformer model for human motion prediction.

This file follows the same external interface as Seq2SeqModel:
- get_batch(...)
- get_batch_srnn(...)
- step(...)

So it can be plugged into translate.py without changing the data pipeline.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import numpy as np
import tensorflow as tf

import seq2seq_model


class TransformerModel(object):
  """Transformer-based forecaster for human motion prediction."""

  def __init__(self,
               architecture,
               source_seq_len,
               target_seq_len,
               rnn_size,
               num_layers,
               max_gradient_norm,
               batch_size,
               learning_rate,
               learning_rate_decay_factor,
               summaries_dir,
               loss_to_use,
               number_of_actions,
               one_hot=True,
               residual_velocities=False,
               dtype=tf.float32):

    # Keep names close to the original Seq2SeqModel.
    self.HUMAN_SIZE = 54
    self.input_size = self.HUMAN_SIZE + number_of_actions if one_hot else self.HUMAN_SIZE
    self.source_seq_len = source_seq_len
    self.target_seq_len = target_seq_len
    self.rnn_size = rnn_size
    self.batch_size = batch_size
    self.one_hot = one_hot
    self.residual_velocities = residual_velocities

    # Transformer hyperparameters.
    # num_layers is reused as the number of Transformer blocks.
    self.num_layers = num_layers
    self.num_heads = 4
    if self.rnn_size % self.num_heads != 0:
      raise ValueError("rnn_size must be divisible by num_heads. "
                       "Current rnn_size=%d, num_heads=%d"
                       % (self.rnn_size, self.num_heads))

    print("Building TransformerModel")
    print("One hot is ", one_hot)
    print("Input size is %d" % self.input_size)
    print("d_model = %d, layers = %d, heads = %d"
          % (self.rnn_size, self.num_layers, self.num_heads))

    # Summary writers.
    self.train_writer = tf.summary.FileWriter(
      os.path.normpath(os.path.join(summaries_dir, 'train')))
    self.test_writer = tf.summary.FileWriter(
      os.path.normpath(os.path.join(summaries_dir, 'test')))

    self.learning_rate = tf.Variable(
      float(learning_rate), trainable=False, dtype=dtype)
    self.learning_rate_decay_op = self.learning_rate.assign(
      self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)

    # === Inputs ===
    with tf.name_scope("inputs"):
      enc_in = tf.placeholder(
        dtype, shape=[None, source_seq_len - 1, self.input_size], name="enc_in")
      dec_in = tf.placeholder(
        dtype, shape=[None, target_seq_len, self.input_size], name="dec_in")
      dec_out = tf.placeholder(
        dtype, shape=[None, target_seq_len, self.input_size], name="dec_out")

      self.encoder_inputs = enc_in
      self.decoder_inputs = dec_in
      self.decoder_outputs = dec_out

    # === Build Transformer forecaster ===
    with tf.variable_scope("transformer_forecaster"):
      pred_seq = self._build_model(enc_in, dec_in)

    # Outputs must be a list of [batch, input_size] tensors,
    # because translate.py and data_utils.revert_output_format expect that format.
    self.outputs = tf.unstack(pred_seq, axis=1)

    with tf.name_scope("loss_angles"):
      pose_loss = tf.reduce_mean(tf.square(dec_out - pred_seq))

      true_vel = dec_out[:, 1:, 0:self.HUMAN_SIZE] - dec_out[:, :-1, 0:self.HUMAN_SIZE]
      pred_vel = pred_seq[:, 1:, 0:self.HUMAN_SIZE] - pred_seq[:, :-1, 0:self.HUMAN_SIZE]
      vel_loss = tf.reduce_mean(tf.square(true_vel - pred_vel))

      self.loss = pose_loss + 0.5 * vel_loss
      # self.loss = pose_loss

      self.loss_summary = tf.summary.scalar("loss/loss", self.loss)
      self.pose_loss_summary = tf.summary.scalar("loss/pose_loss", pose_loss)
      self.vel_loss_summary = tf.summary.scalar("loss/velocity_loss", vel_loss)

    # Optimizer.
    # Adam is usually much more stable for Transformer than plain SGD.
    params = tf.trainable_variables()
    opt = tf.train.AdamOptimizer(self.learning_rate)

    gradients = tf.gradients(self.loss, params)
    clipped_gradients, norm = tf.clip_by_global_norm(
      gradients, max_gradient_norm)

    self.gradient_norms = norm
    self.updates = opt.apply_gradients(
      zip(clipped_gradients, params), global_step=self.global_step)

    self.learning_rate_summary = tf.summary.scalar(
      "learning_rate/learning_rate", self.learning_rate)

    # translate.py expects these per-action summary placeholders to exist.
    self._create_euler_error_summaries()

    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

  def _positional_encoding(self, length, depth):
    """Create sinusoidal positional encoding with shape [1, length, depth]."""
    positions = np.arange(length)[:, np.newaxis]
    dims = np.arange(depth)[np.newaxis, :]

    angle_rates = 1.0 / np.power(10000.0, (2 * (dims // 2)) / float(depth))
    angle_rads = positions * angle_rates

    pe = np.zeros((length, depth), dtype=np.float32)
    pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)

  def _layer_norm(self, x, name):
    with tf.variable_scope(name):
      return tf.contrib.layers.layer_norm(x)

  def _split_heads(self, x, num_heads):
    """Split [B, T, D] into [B, H, T, D/H]."""
    batch_size = tf.shape(x)[0]
    length = tf.shape(x)[1]
    depth = self.rnn_size // num_heads
    x = tf.reshape(x, [batch_size, length, num_heads, depth])
    return tf.transpose(x, [0, 2, 1, 3])

  def _combine_heads(self, x):
    """Combine [B, H, T, D/H] into [B, T, D]."""
    x = tf.transpose(x, [0, 2, 1, 3])
    batch_size = tf.shape(x)[0]
    length = tf.shape(x)[1]
    return tf.reshape(x, [batch_size, length, self.rnn_size])

  def _multi_head_attention(self, q, k, v, name):
    with tf.variable_scope(name):
      q_proj = tf.layers.dense(q, self.rnn_size, use_bias=False, name="q")
      k_proj = tf.layers.dense(k, self.rnn_size, use_bias=False, name="k")
      v_proj = tf.layers.dense(v, self.rnn_size, use_bias=False, name="v")

      qh = self._split_heads(q_proj, self.num_heads)
      kh = self._split_heads(k_proj, self.num_heads)
      vh = self._split_heads(v_proj, self.num_heads)

      depth = self.rnn_size // self.num_heads
      scores = tf.matmul(qh, kh, transpose_b=True)
      scores = scores / math.sqrt(float(depth))

      weights = tf.nn.softmax(scores, axis=-1)
      context = tf.matmul(weights, vh)

      context = self._combine_heads(context)
      out = tf.layers.dense(context, self.rnn_size, name="out")
      return out

  def _feed_forward(self, x, name):
    with tf.variable_scope(name):
      hidden = tf.layers.dense(
        x, self.rnn_size * 4, activation=tf.nn.relu, name="dense_1")
      out = tf.layers.dense(hidden, self.rnn_size, name="dense_2")
      return out

  def _encoder_block(self, x, layer_id):
    with tf.variable_scope("encoder_layer_%d" % layer_id):
      attn_in = self._layer_norm(x, "ln_attn")
      x = x + self._multi_head_attention(attn_in, attn_in, attn_in, "self_attn")

      ffn_in = self._layer_norm(x, "ln_ffn")
      x = x + self._feed_forward(ffn_in, "ffn")
      return x

  def _decoder_block(self, y, memory, layer_id):
    with tf.variable_scope("decoder_layer_%d" % layer_id):
      self_attn_in = self._layer_norm(y, "ln_self_attn")
      y = y + self._multi_head_attention(
        self_attn_in, self_attn_in, self_attn_in, "self_attn")

      cross_q = self._layer_norm(y, "ln_cross_q")
      cross_kv = self._layer_norm(memory, "ln_cross_kv")
      y = y + self._multi_head_attention(cross_q, cross_kv, cross_kv, "cross_attn")

      ffn_in = self._layer_norm(y, "ln_ffn")
      y = y + self._feed_forward(ffn_in, "ffn")
      return y

  def _build_model(self, enc_in, dec_in):
    """Build non-autoregressive Transformer forecaster.

    enc_in: [B, source_seq_len-1, input_size]
    dec_in: [B, target_seq_len, input_size]
            only dec_in[:,0,:] is used as the last observed frame for residuals.
    return: [B, target_seq_len, input_size]
    """

    batch_size = tf.shape(enc_in)[0]

    # Project past pose sequence to d_model.
    x = tf.layers.dense(enc_in, self.rnn_size, name="input_projection")
    x = x + self._positional_encoding(self.source_seq_len - 1, self.rnn_size)

    # Encoder over observed history.
    memory = x
    for i in range(self.num_layers):
      memory = self._encoder_block(memory, i)

    memory = self._layer_norm(memory, "encoder_final_ln")

    # Learnable future query embeddings.
    start_frame = dec_in[:, 0:1, :]  # last observed pose, shape [B, 1, input_size]
    decoder_context = tf.tile(start_frame, [1, self.target_seq_len, 1])

    y = tf.layers.dense(
      decoder_context,
      self.rnn_size,
      name="decoder_input_projection"
    )

    y = y + self._positional_encoding(self.target_seq_len, self.rnn_size)

    # Decoder queries attend to encoded history.
    for i in range(self.num_layers):
      y = self._decoder_block(y, memory, i)

    y = self._layer_norm(y, "decoder_final_ln")

    raw = tf.layers.dense(y, self.input_size, name="output_projection")

    if self.residual_velocities:
      # Treat network output as velocity/residual over pose dimensions.
      base_pose = dec_in[:, 0:1, 0:self.HUMAN_SIZE]
      delta_pose = raw[:, :, 0:self.HUMAN_SIZE]
      pred_pose = base_pose + tf.cumsum(delta_pose, axis=1)

      if self.one_hot:
        action_part = dec_in[:, 0:1, self.HUMAN_SIZE:self.input_size]
        action_part = tf.tile(action_part, [1, self.target_seq_len, 1])
        raw = tf.concat([pred_pose, action_part], axis=2)
      else:
        raw = pred_pose

    return raw

  def _create_euler_error_summaries(self):
    """Create the same attributes that translate.py expects."""

    actions = ["walking", "eating", "smoking", "discussion", "directions",
               "greeting", "phoning", "posing", "purchases", "sitting",
               "sittingdown", "takingphoto", "waiting", "walkingdog",
               "walkingtogether"]

    horizons = [
      ("80", "0080"),
      ("160", "0160"),
      ("320", "0320"),
      ("400", "0400"),
      ("560", "0560"),
      ("1000", "1000")
    ]

    for action in actions:
      with tf.name_scope("euler_error_%s" % action):
        for short_ms, padded_ms in horizons:
          ph_name = "%s_err%s" % (action, short_ms)
          summary_name = "%s_err%s_summary" % (action, short_ms)

          ph = tf.placeholder(
            tf.float32, name="%s_srnn_seeds_%s" % (action, padded_ms))
          summary = tf.summary.scalar(
            "euler_error_%s/srnn_seeds_%s" % (action, padded_ms), ph)

          setattr(self, ph_name, ph)
          setattr(self, summary_name, summary)

  def step(self, session, encoder_inputs, decoder_inputs, decoder_outputs,
           forward_only, srnn_seeds=False):
    """Same interface as Seq2SeqModel.step(...)."""

    input_feed = {
      self.encoder_inputs: encoder_inputs,
      self.decoder_inputs: decoder_inputs,
      self.decoder_outputs: decoder_outputs
    }

    if not srnn_seeds:
      if not forward_only:
        output_feed = [
          self.updates,
          self.gradient_norms,
          self.loss,
          self.loss_summary,
          self.learning_rate_summary
        ]
        outputs = session.run(output_feed, input_feed)
        return outputs[1], outputs[2], outputs[3], outputs[4]
      else:
        output_feed = [self.loss, self.loss_summary]
        outputs = session.run(output_feed, input_feed)
        return outputs[0], outputs[1]
    else:
      output_feed = [self.loss, self.outputs, self.loss_summary]
      outputs = session.run(output_feed, input_feed)
      return outputs[0], outputs[1], outputs[2]

  # Reuse the original batching logic directly.
  get_batch = seq2seq_model.Seq2SeqModel.get_batch
  find_indices_srnn = seq2seq_model.Seq2SeqModel.find_indices_srnn
  get_batch_srnn = seq2seq_model.Seq2SeqModel.get_batch_srnn