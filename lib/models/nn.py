#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from lib import linalg
from lib.etc.tarjan import Tarjan
from lib.models import rnn
from configurable import Configurable
from vocab import Vocab

#***************************************************************
class NN(Configurable):
  """"""
  
  ZERO = tf.convert_to_tensor(0.)
  ONE = tf.convert_to_tensor(1.)
  PUNCT = set(['``', "''", ':', ',', '.', 'PU', 'PUNCT'])
  
  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""
    
    global_step = kwargs.pop('global_step', None)
    super(NN, self).__init__(*args, **kwargs)
    
    if global_step is not None:
      self._global_sigmoid = 1-tf.nn.sigmoid(3*(2*global_step/(self.train_iters-1)-1))
    else:
      self._global_sigmoid = 1
    
    self.tokens_to_keep3D = None
    self.sequence_lengths = None
    self.n_tokens = None
    self.moving_params = None
    return
  
  #=============================================================
  def embed_concat(self, word_inputs, tag_inputs=None, rel_inputs=None):
    """"""
    
    if self.moving_params is None:
      word_keep_prob = self.word_keep_prob
      tag_keep_prob = self.tag_keep_prob
      rel_keep_prob = self.rel_keep_prob
      noise_shape = tf.pack([tf.shape(word_inputs)[0], tf.shape(word_inputs)[1], 1])
      
      word_mask = 1
      tag_mask = 1
      rel_mask = 1
      
      if word_keep_prob < 1:
        word_mask = tf.nn.dropout(tf.ones(noise_shape), word_keep_prob)*word_keep_prob
      if tag_inputs is not None and tag_keep_prob < 1:
        tag_mask = tf.nn.dropout(tf.ones(noise_shape), tag_keep_prob)*tag_keep_prob
      if rel_inputs is not None and rel_keep_prob < 1:
        rel_mask = tf.nn.dropout(tf.ones(noise_shape), rel_keep_prob)*rel_keep_prob
        
      word_inputs *= word_mask #* (word_mask + (1-tag_mask) + (1-rel_mask))
      if tag_inputs is not None:
        tag_inputs *= tag_mask #* ((1-word_mask) + (tag_mask) + (1-rel_mask))
      if rel_inputs is not None:
        rel_inputs *= rel_mask #* ((1-word_mask) + (1-tag_mask) + (rel_mask))
    return tf.concat(2, filter(lambda x: x is not None, [word_inputs, tag_inputs, rel_inputs]))
    
  #=============================================================
  def RNN(self, inputs):
    """"""
    
    input_size = inputs.get_shape().as_list()[-1]
    cell = self.recur_cell(self._config, input_size=input_size, moving_params=self.moving_params)
    lengths = tf.reshape(tf.to_int64(self.sequence_lengths), [-1])
    
    if self.moving_params is None:
      if self.drop_gradually:
        s = self.global_sigmoid
        ff_keep_prob = s + (1-s)*self.ff_keep_prob
        recur_keep_prob = s + (1-s)*self.recur_keep_prob
      else:
        ff_keep_prob = self.ff_keep_prob
        recur_keep_prob = self.recur_keep_prob
    else:
      ff_keep_prob = 1
      recur_keep_prob = 1
    
    if self.recur_bidir:
      top_recur, fw_recur, bw_recur = rnn.dynamic_bidirectional_rnn(cell, cell, inputs,
                                                                    lengths,
                                                                    ff_keep_prob=ff_keep_prob,
                                                                    recur_keep_prob=recur_keep_prob,
                                                                    dtype=tf.float32)
      fw_cell, fw_out = tf.split(1, 2, fw_recur)
      bw_cell, bw_out = tf.split(1, 2, bw_recur)
      end_recur = tf.concat(1, [fw_out, bw_out])
      top_recur.set_shape([tf.Dimension(None), tf.Dimension(None), tf.Dimension(2*self.recur_size)])
      if self.moving_params is None:
        for direction in ('FW', 'BW'):
          if self.recur_cell.__name__ != 'GRUCell':
            with tf.variable_scope("BiRNN_%s/%s/Linear" % (direction,self.recur_cell.__name__), reuse=True):
              matrix = tf.get_variable('Weights')
              n_splits = matrix.get_shape().as_list()[-1] // self.recur_size
              I = tf.diag(tf.ones([self.recur_size]))
              for W in tf.split(1, n_splits, matrix):
                WTWmI = tf.matmul(W, W, transpose_a=True) - I
                tf.add_to_collection('ortho_losses', tf.nn.l2_loss(WTWmI))
          else:
            for name in ['Gates', 'Candidate']:
              with tf.variable_scope("BiRNN_%s/GRUCell/%s/Linear" % (direction,name), reuse=True):
                matrix = tf.get_variable('Weights')
                n_splits = matrix.get_shape().as_list()[-1] // self.recur_size
                I = tf.diag(tf.ones([self.recur_size]))
                for W in tf.split(1, n_splits, matrix):
                  WTWmI = tf.matmul(W, W, transpose_a=True) - I
                  tf.add_to_collection('ortho_losses', tf.nn.l2_loss(WTWmI))
    else:
      top_recur, end_recur = rnn.dynamic_rnn(cell, inputs,
                                             lengths,
                                             ff_keep_prob=ff_keep_prob,
                                             recur_keep_prob=recur_keep_prob,
                                             dtype=tf.float32)
      top_recur.set_shape([tf.Dimension(None), tf.Dimension(None), tf.Dimension(self.recur_size)])
      if self.moving_params is None:
        if self.recur_cell.__name__ != 'GRUCell':
          with tf.variable_scope("%s/Linear" % (self.recur_cell.__name__), reuse=True):
            matrix = tf.get_variable('Weights')
            n_splits = matrix.get_shape().as_list()[-1] // self.recur_size
            I = tf.diag(tf.ones([self.recur_size]))
            for W in tf.split(1, n_splits, matrix):
              WTWmI = tf.matmul(W, W, transpose_a=True) - I
              tf.add_to_collection('ortho_losses', tf.nn.l2_loss(WTWmI))
        else:
          for name in ['Gates', 'Candidate']:
            with tf.variable_scope("GRUCell/%s/Linear" % (name), reuse=True):
              matrix = tf.get_variable('Weights')
              n_splits = matrix.get_shape().as_list()[-1] // self.recur_size
              I = tf.diag(tf.ones([self.recur_size]))
              for W in tf.split(1, n_splits, matrix):
                WTWmI = tf.matmul(W, W, transpose_a=True) - I
                tf.add_to_collection('ortho_losses', tf.nn.l2_loss(WTWmI))
              
    if self.moving_params is None:
      tf.add_to_collection('recur_losses', self.recur_loss(top_recur))
      tf.add_to_collection('covar_losses', self.covar_loss(top_recur))
    return top_recur, end_recur
  
  #=============================================================
  def MLP(self, inputs, n_splits=1):
    """"""
    
    n_dims = len(inputs.get_shape().as_list())
    batch_size = tf.shape(inputs)[0]
    bucket_size = tf.shape(inputs)[1]
    input_size = inputs.get_shape().as_list()[-1]
    output_size = self.mlp_size
    output_shape = tf.pack([batch_size] + [bucket_size]*(n_dims-2) + [output_size])
    shape_to_set = [tf.Dimension(None)]*(n_dims-1) + [tf.Dimension(output_size)]
    
    if self.moving_params is None:
      if self.drop_gradually:
        s = self.global_sigmoid
        keep_prob = s + (1-s)*self.mlp_keep_prob
      else:
        keep_prob = self.mlp_keep_prob
    else:
      keep_prob = 1
    if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
      noise_shape = tf.pack([batch_size] + [1]*(n_dims-2) + [input_size])
      inputs = tf.nn.dropout(inputs, keep_prob, noise_shape=noise_shape)
    
    linear = linalg.linear(inputs,
                        output_size,
                        n_splits=n_splits,
                        add_bias=True,
                        moving_params=self.moving_params)
    if n_splits == 1:
      linear = [linear]
    for i, split in enumerate(linear):
      split = self.mlp_func(split)
      split.set_shape(shape_to_set)
      linear[i] = split
    if self.moving_params is None:
      with tf.variable_scope('Linear', reuse=True):
        matrix = tf.get_variable('Weights')
        I = tf.diag(tf.ones([self.mlp_size]))
        for W in tf.split(1, n_splits, matrix):
          WTWmI = tf.matmul(W, W, transpose_a=True) - I
          tf.add_to_collection('ortho_losses', tf.nn.l2_loss(WTWmI))
      for split in linear:
        tf.add_to_collection('covar_losses', self.covar_loss(split))
    if n_splits == 1:
      return linear[0]
    else:
      return linear
  
  #=============================================================
  def double_MLP(self, inputs, n_splits=1):
    """"""
    
    batch_size = tf.shape(inputs)[0]
    bucket_size = tf.shape(inputs)[1]
    input_size = inputs.get_shape().as_list()[-1]
    output_size = self.mlp_size
    output_shape = tf.pack([batch_size, bucket_size, bucket_size, output_size])
    shape_to_set = [tf.Dimension(None), tf.Dimension(None), tf.Dimension(None), tf.Dimension(output_size)]
    
    if self.moving_params is None:
      if self.drop_gradually:
        s = self.global_sigmoid
        keep_prob = s + (1-s)*self.mlp_keep_prob
      else:
        keep_prob = self.mlp_keep_prob
    else:
      keep_prob = 1
    if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
      noise_shape = tf.pack([batch_size, 1, input_size])
      inputs = tf.nn.dropout(inputs, keep_prob, noise_shape=noise_shape)
    
    lin1, lin2 = linalg.linear(inputs,
                               output_size*n_splits,
                               n_splits=2,
                               add_bias=True,
                               moving_params=self.moving_params)
    lin1 = tf.reshape(tf.transpose(lin1, [0, 2, 1]), tf.pack([-1, bucket_size, 1]))
    lin2 = tf.reshape(tf.transpose(lin2, [0, 2, 1]), tf.pack([-1, 1, bucket_size]))
    lin = lin1 + lin2
    lin = tf.reshape(lin, tf.pack([batch_size, n_splits*output_size, bucket_size, bucket_size]))
    lin = tf.transpose(lin, [0,2,3,1])
    top_mlps = tf.split(3, n_splits, self.mlp_func(lin))
    for top_mlp in top_mlps:
      top_mlp.set_shape(shape_to_set)
    if self.moving_params is None:
      with tf.variable_scope('Linear', reuse=True):
        matrix = tf.get_variable('Weights')
        I = tf.diag(tf.ones([self.mlp_size]))
        for W in tf.split(1, 2*n_splits, matrix):
          WTWmI = tf.matmul(W, W, transpose_a=True) - I
          tf.add_to_collection('ortho_losses', tf.nn.l2_loss(WTWmI))
      for split in top_mlps:
        tf.add_to_collection('covar_losses', self.covar_loss(split))
    if n_splits == 1:
      return top_mlps[0]
    else:
      return top_mlps
  
  #=============================================================
  def linear_classifier(self, inputs, n_classes, add_bias=True):
    """"""
    
    n_dims = len(inputs.get_shape().as_list())
    batch_size = tf.shape(inputs)[0]
    bucket_size = tf.shape(inputs)[1]
    input_size = inputs.get_shape().as_list()[-1]
    output_size = n_classes
    output_shape = tf.pack([batch_size] + [bucket_size]*(n_dims-2) + [output_size])
    
    if self.moving_params is None:
      if self.drop_gradually:
        s = self.global_sigmoid
        keep_prob = s + (1-s)*self.mlp_keep_prob
      else:
        keep_prob = self.mlp_keep_prob
    else:
      keep_prob = 1
    if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
      noise_shape = tf.pack([batch_size] + [1]*(n_dims-2) +[input_size])
      inputs = tf.nn.dropout(inputs, keep_prob, noise_shape=noise_shape)
    
    inputs = tf.reshape(inputs, [-1, input_size])
    output = linalg.linear(inputs,
                    output_size,
                    add_bias=add_bias,
                    initializer=tf.zeros_initializer,
                    moving_params=self.moving_params)
    output = tf.reshape(output, output_shape)
    output.set_shape([tf.Dimension(None)]*(n_dims-1) + [tf.Dimension(output_size)])
    return output
  
  #=============================================================
  def bilinear_classifier(self, inputs1, inputs2, add_bias1=False, add_bias2=True):
    """"""
    
    input_shape = tf.shape(inputs1)
    batch_size = input_shape[0]
    bucket_size = input_shape[1]
    input_size = inputs1.get_shape().as_list()[-1]
    shape_to_set = tf.pack([batch_size, bucket_size, input_size+1])
    
    if self.moving_params is None:
      if self.drop_gradually:
        s = self.global_sigmoid
        keep_prob = s + (1-s)*self.mlp_keep_prob
      else:
        keep_prob = self.mlp_keep_prob
    else:
      keep_prob = 1
    if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
      noise_shape = tf.pack([batch_size, 1, input_size])
      inputs1 = tf.nn.dropout(inputs1, keep_prob, noise_shape=noise_shape)
      inputs2 = tf.nn.dropout(inputs2, keep_prob, noise_shape=noise_shape)
    
    bilin = linalg.bilinear(inputs1, inputs2, 1,
                            add_bias1=add_bias1,
                            add_bias2=add_bias2,
                            initializer=tf.zeros_initializer,
                            moving_params=self.moving_params)
    output = tf.squeeze(bilin)
    return output
  
  #=============================================================
  def diagonal_bilinear_classifier(self, inputs1, inputs2, add_bias1=False, add_bias2=True):
    """"""
    
    input_shape = tf.shape(inputs1)
    batch_size = input_shape[0]
    bucket_size = input_shape[1]
    input_size = inputs1.get_shape().as_list()[-1]
    shape_to_set = tf.pack([batch_size, bucket_size, input_size+1])
    
    if self.moving_params is None:
      if self.drop_gradually:
        s = self.global_sigmoid
        keep_prob = s + (1-s)*self.mlp_keep_prob
      else:
        keep_prob = self.mlp_keep_prob
    else:
      keep_prob = 1
    if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
      noise_shape = tf.pack([batch_size, 1, input_size])
      inputs1 = tf.nn.dropout(inputs1, tf.sqrt(keep_prob), noise_shape=noise_shape)
      inputs2 = tf.nn.dropout(inputs2, tf.sqrt(keep_prob), noise_shape=noise_shape)
    
    bilin = linalg.diagonal_bilinear(inputs1, inputs2, 1,
                                     add_bias1=add_bias1,
                                     add_bias2=add_bias2,
                                     initializer=tf.zeros_initializer,
                                     moving_params=self.moving_params)
    output = tf.squeeze(bilin)
    return output
  
  #=============================================================
  def conditional_linear_classifier(self, inputs, n_classes, probs, add_bias=True):
    """"""
    
    input_shape = tf.shape(inputs)
    batch_size = input_shape[0]
    bucket_size = input_shape[1]
    input_size = inputs.get_shape().as_list()[-1]
    
    if len(probs.get_shape().as_list()) == 2:
      probs = tf.to_float(tf.one_hot(tf.to_int64(probs), bucket_size, 1, 0))
    else:
      probs = tf.stop_gradient(probs)
    
    if self.moving_params is None:
      if self.drop_gradually:
        s = self.global_sigmoid
        keep_prob = s + (1-s)*self.mlp_keep_prob
      else:
        keep_prob = self.mlp_keep_prob
    else:
      keep_prob = 1
    if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
      noise_shape = tf.pack([batch_size, 1, 1, input_size])
      inputs = tf.nn.dropout(inputs, keep_prob, noise_shape=noise_shape)
    
    lin = linalg.linear(inputs,
                        n_classes,
                        add_bias=add_bias,
                        initializer=tf.zeros_initializer,
                        moving_params=self.moving_params)
    weighted_lin = tf.batch_matmul(lin, tf.expand_dims(probs, 3), adj_x=True)
    
    return weighted_lin, lin
  
  #=============================================================
  def conditional_diagonal_bilinear_classifier(self, inputs1, inputs2, n_classes, probs, add_bias1=True, add_bias2=True):
    """"""
    
    input_shape = tf.shape(inputs1)
    batch_size = input_shape[0]
    bucket_size = input_shape[1]
    input_size = inputs1.get_shape().as_list()[-1]
    input_shape_to_set = [tf.Dimension(None), tf.Dimension(None), input_size+1]
    output_shape = tf.pack([batch_size, bucket_size, n_classes, bucket_size])
    if len(probs.get_shape().as_list()) == 2:
      probs = tf.to_float(tf.one_hot(tf.to_int64(probs), bucket_size, 1, 0))
    else:
      probs = tf.stop_gradient(probs)
    
    if self.moving_params is None:
      if self.drop_gradually:
        s = self.global_sigmoid
        keep_prob = s + (1-s)*self.mlp_keep_prob
      else:
        keep_prob = self.mlp_keep_prob
    else:
      keep_prob = 1
    if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
      noise_shape = tf.pack([batch_size, 1, input_size])
      inputs1 = tf.nn.dropout(inputs1, tf.sqrt(keep_prob), noise_shape=noise_shape)
      inputs2 = tf.nn.dropout(inputs2, tf.sqrt(keep_prob), noise_shape=noise_shape)
    
    inputs1 = tf.concat(2, [inputs1, tf.ones(tf.pack([batch_size, bucket_size, 1]))])
    inputs1.set_shape(input_shape_to_set)
    inputs2 = tf.concat(2, [inputs2, tf.ones(tf.pack([batch_size, bucket_size, 1]))])
    inputs2.set_shape(input_shape_to_set)
    
    bilin = linalg.diagonal_bilinear(inputs1, inputs2,
                                     n_classes,
                                     add_bias1=add_bias1,
                                     add_bias2=add_bias2,
                                     initializer=tf.zeros_initializer,
                                     moving_params=self.moving_params)
    weighted_bilin = tf.batch_matmul(bilin, tf.expand_dims(probs, 3))
    
    return weighted_bilin, bilin
  
  #=============================================================
  def conditional_bilinear_classifier(self, inputs1, inputs2, n_classes, probs, add_bias1=True, add_bias2=True):
    """"""
    
    input_shape = tf.shape(inputs1)
    batch_size = input_shape[0]
    bucket_size = input_shape[1]
    input_size = inputs1.get_shape().as_list()[-1]
    input_shape_to_set = [tf.Dimension(None), tf.Dimension(None), input_size+1]
    output_shape = tf.pack([batch_size, bucket_size, n_classes, bucket_size])
    if len(probs.get_shape().as_list()) == 2:
      probs = tf.to_float(tf.one_hot(tf.to_int64(probs), bucket_size, 1, 0))
    else:
      probs = tf.stop_gradient(probs)
    
    if self.moving_params is None:
      if self.drop_gradually:
        s = self.global_sigmoid
        keep_prob = s + (1-s)*self.mlp_keep_prob
      else:
        keep_prob = self.mlp_keep_prob
    else:
      keep_prob = 1
    if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
      noise_shape = tf.pack([batch_size, 1, input_size])
      inputs1 = tf.nn.dropout(inputs1, keep_prob, noise_shape=noise_shape)
      inputs2 = tf.nn.dropout(inputs2, keep_prob, noise_shape=noise_shape)
    
    inputs1 = tf.concat(2, [inputs1, tf.ones(tf.pack([batch_size, bucket_size, 1]))])
    inputs1.set_shape(input_shape_to_set)
    inputs2 = tf.concat(2, [inputs2, tf.ones(tf.pack([batch_size, bucket_size, 1]))])
    inputs2.set_shape(input_shape_to_set)
    
    bilin = linalg.bilinear(inputs1, inputs2,
                     n_classes,
                     add_bias1=add_bias1,
                     add_bias2=add_bias2,
                     initializer=tf.zeros_initializer,
                     moving_params=self.moving_params)
    weighted_bilin = tf.batch_matmul(bilin, tf.expand_dims(probs, 3))
    
    return weighted_bilin, bilin
  
  #=============================================================
  def output(self, logits3D, targets3D):
    """"""
    
    original_shape = tf.shape(logits3D)
    batch_size = original_shape[0]
    bucket_size = original_shape[1]
    flat_shape = tf.pack([batch_size, bucket_size])
    
    logits2D = tf.reshape(logits3D, tf.pack([batch_size*bucket_size, -1]))
    targets1D = tf.reshape(targets3D, [-1])
    tokens_to_keep1D = tf.reshape(self.tokens_to_keep3D, [-1])
    
    predictions1D = tf.to_int32(tf.argmax(logits2D, 1))
    probabilities2D = tf.nn.softmax(logits2D)
    cross_entropy1D = tf.nn.sparse_softmax_cross_entropy_with_logits(logits2D, targets1D)
    
    correct1D = tf.to_float(tf.equal(predictions1D, targets1D))
    n_correct = tf.reduce_sum(correct1D * tokens_to_keep1D)
    accuracy = n_correct / self.n_tokens
    loss = tf.reduce_sum(cross_entropy1D * tokens_to_keep1D) / self.n_tokens
    
    output = {
      'probabilities': tf.reshape(probabilities2D, original_shape),
      'predictions': tf.reshape(predictions1D, flat_shape),
      'tokens': tokens_to_keep1D,
      'correct': correct1D * tokens_to_keep1D,
      'n_correct': n_correct,
      'n_tokens': self.n_tokens,
      'accuracy': accuracy,
      'loss': loss
    }
    
    return output
  
  #=============================================================
  def conditional_probabilities(self, logits4D, transpose=True):
    """"""
    
    if transpose:
      logits4D = tf.transpose(logits4D, [0,1,3,2])
    original_shape = tf.shape(logits4D)
    n_classes = original_shape[3]
    
    logits2D = tf.reshape(logits4D, tf.pack([-1, n_classes]))
    probabilities2D = tf.nn.softmax(logits2D)
    return tf.reshape(probabilities2D, original_shape)
    
  #=============================================================
  def pseudo_predict(self, predictions, targets):
    """"""
    
    random_flip = tf.random_uniform(tf.shape(predictions))
    return tf.select(tf.greater(random_flip, self.global_sigmoid), predictions, targets)
  
  #=============================================================
  def recur_loss(self, top_recur):
    """"""
    
    batch_size = tf.to_float(tf.shape(top_recur)[0])
    lengths = tf.reshape(self.sequence_lengths, [-1,1])
    
    norms = tf.sqrt(tf.reduce_sum(top_recur**2, 2, keep_dims=True) + 1e-12)
    means = tf.reduce_sum(norms * self.tokens_to_keep3D, 1, keep_dims=True) / (self.n_tokens + 1e-12)
    centered_norms = norms - means
    var_norms = tf.reduce_sum(centered_norms**2, 1) / (lengths + 1e-12)
    mean_var_norms = var_norms / batch_size
    return tf.nn.l2_loss(tf.reduce_mean(mean_var_norms, 0))
  
  #=============================================================
  def covar_loss(self, top_states):
    """"""
    
    n_dims = len(top_states.get_shape().as_list())
    hidden_size = top_states.get_shape().as_list()[-1]
    n_tokens = tf.to_float(self.n_tokens)
    I = tf.diag(tf.ones([hidden_size]))
    
    if n_dims == 3:
      top_states = top_states * self.tokens_to_keep3D
      n_tokens = self.n_tokens
    elif n_dims == 4:
      top_states = top_states * tf.expand_dims(self.tokens_to_keep3D, 1) * tf.expand_dims(self.tokens_to_keep3D, 2)
      n_tokens = self.n_tokens**2
    top_states = tf.reshape(top_states * self.tokens_to_keep3D, [-1, hidden_size])
    means = tf.reduce_sum(top_states, 0, keep_dims=True) / n_tokens
    centered_states = top_states - means
    covar_mat = tf.matmul(centered_states, centered_states, transpose_a=True) / n_tokens
    off_diag_covar_mat = covar_mat * (1-I)
    return tf.nn.l2_loss(off_diag_covar_mat)
  
  #=============================================================
  @staticmethod
  def tag_argmax(tag_probs, tokens_to_keep):
    """"""
    
    return np.argmax(tag_probs[:,Vocab.ROOT:], axis=1)+Vocab.ROOT
  
  #=============================================================
  @staticmethod
  def parse_argmax(parse_probs, tokens_to_keep):
    """"""
    
    tokens_to_keep[0] = True
    length = np.sum(tokens_to_keep)
    I = np.eye(len(tokens_to_keep))
    # block loops and pad heads
    parse_probs = parse_probs * tokens_to_keep * (1-I)
    parse_preds = np.argmax(parse_probs, axis=1)
    tokens = np.arange(1, length)
    roots = np.where(parse_preds[tokens] == 0)[0]+1
    # ensure at least one root
    if len(roots) < 1:
      # The current root probabilities
      root_probs = parse_probs[tokens,0]
      # The current head probabilities
      old_head_probs = parse_probs[tokens, parse_preds[tokens]]
      # Get new potential root probabilities
      new_root_probs = root_probs / old_head_probs
      # Select the most probable root
      new_root = tokens[np.argmax(new_root_probs)]
      # Make the change
      parse_preds[new_root] = 0
    # ensure at most one root
    elif len(roots) > 1:
      # The probabilities of the current heads
      root_probs = parse_probs[roots,0]
      # Set the probability of depending on the root zero
      parse_probs[roots,0] = 0
      # Get new potential heads and their probabilities
      new_heads = np.argmax(parse_probs[roots][:,tokens], axis=1)+1
      new_head_probs = parse_probs[roots, new_heads] / root_probs
      # Select the most probable root
      new_root = roots[np.argmin(new_head_probs)]
      # Make the change
      parse_preds[roots] = new_heads
      parse_preds[new_root] = 0
    # remove cycles
    tarjan = Tarjan(parse_preds, tokens)
    cycles = tarjan.SCCs
    for SCC in tarjan.SCCs:
      if len(SCC) > 1:
        dependents = set()
        to_visit = set(SCC)
        while len(to_visit) > 0:
          node = to_visit.pop()
          if not node in dependents:
            dependents.add(node)
            to_visit.update(tarjan.edges[node])
        # The indices of the nodes that participate in the cycle
        cycle = np.array(list(SCC))
        # The probabilities of the current heads
        old_heads = parse_preds[cycle]
        old_head_probs = parse_probs[cycle, old_heads]
        # Set the probability of depending on a non-head to zero
        non_heads = np.array(list(dependents))
        parse_probs[np.repeat(cycle, len(non_heads)), np.repeat([non_heads], len(cycle), axis=0).flatten()] = 0
        # Get new potential heads and their probabilities
        new_heads = np.argmax(parse_probs[cycle][:,tokens], axis=1)+1
        new_head_probs = parse_probs[cycle, new_heads] / old_head_probs
        # Select the most probable change
        change = np.argmax(new_head_probs)
        changed_cycle = cycle[change]
        old_head = old_heads[change]
        new_head = new_heads[change]
        # Make the change
        parse_preds[changed_cycle] = new_head
        tarjan.edges[new_head].add(changed_cycle)
        tarjan.edges[old_head].remove(changed_cycle)
    return parse_preds
  
  #=============================================================
  @staticmethod
  def rel_argmax(rel_probs, tokens_to_keep):
    """"""
    
    tokens_to_keep[0] = True
    rel_probs[:,Vocab.PAD] = 0
    root = Vocab.ROOT
    length = np.sum(tokens_to_keep)
    tokens = np.arange(1, length)
    rel_preds = np.argmax(rel_probs, axis=1)
    roots = np.where(rel_preds[tokens] == root)[0]+1
    if len(roots) < 1:
      rel_preds[1+np.argmax(rel_probs[tokens,root])] = root
    elif len(roots) > 1:
      root_probs = rel_probs[roots, root]
      rel_probs[roots, root] = 0
      new_rel_preds = np.argmax(rel_probs[roots], axis=1)
      new_rel_probs = rel_probs[roots, new_rel_preds] / root_probs
      new_root = roots[np.argmin(new_rel_probs)]
      rel_preds[roots] = new_rel_preds
      rel_preds[new_root] = root
    return rel_preds
  
  #=============================================================
  def __call__(self, inputs, targets, moving_params=None):
    """"""
    
    raise NotImplementedError()
  
  #=============================================================
  @property
  def global_sigmoid(self):
    return self._global_sigmoid
