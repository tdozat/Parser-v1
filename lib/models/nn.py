#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
# Copyright 2016 Timothy Dozat
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
      
      if word_keep_prob < 1:
        word_mask = tf.nn.dropout(tf.ones(noise_shape), word_keep_prob)*word_keep_prob
      else:
        word_mask = 1
      if tag_inputs is not None and tag_keep_prob < 1:
        tag_mask = tf.nn.dropout(tf.ones(noise_shape), tag_keep_prob)*tag_keep_prob
      else:
        tag_mask = 1
      if rel_inputs is not None and rel_keep_prob < 1:
        rel_mask = tf.nn.dropout(tf.ones(noise_shape), rel_keep_prob)*rel_keep_prob
      else:
        rel_mask = 1
      
      word_embed_size = word_inputs.get_shape().as_list()[-1]
      tag_embed_size = 0 if tag_inputs is None else tag_inputs.get_shape().as_list()[-1]
      rel_embed_size = 0 if rel_inputs is None else rel_inputs.get_shape().as_list()[-1]
      total_size = word_embed_size + tag_embed_size + rel_embed_size
      if word_embed_size == tag_embed_size:
        total_size += word_embed_size
      dropped_sizes = word_mask * word_embed_size + tag_mask * tag_embed_size + rel_mask * rel_embed_size
      if word_embed_size == tag_embed_size:
        dropped_sizes += word_mask * tag_mask * word_embed_size
      scale_factor = total_size / (dropped_sizes + self.epsilon)
      
      word_inputs *= word_mask * scale_factor
      if tag_inputs is not None:
        tag_inputs *= tag_mask * scale_factor
      if rel_inputs is not None:
        rel_inputs *= rel_mask * scale_factor
    else:
      word_embed_size = word_inputs.get_shape().as_list()[-1]
      tag_embed_size = 0 if tag_inputs is None else tag_inputs.get_shape().as_list()[-1]
      rel_embed_size = 0 if rel_inputs is None else rel_inputs.get_shape().as_list()[-1]
    
    return tf.concat(2, filter(lambda x: x is not None, [word_inputs, tag_inputs, rel_inputs]))
  
  #=============================================================
  def RNN(self, inputs):
    """"""
    
    input_size = inputs.get_shape().as_list()[-1]
    cell = self.recur_cell(self._config, input_size=input_size, moving_params=self.moving_params)
    lengths = tf.reshape(tf.to_int64(self.sequence_lengths), [-1])
    
    if self.moving_params is None:
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
    else:
      top_recur, end_recur = rnn.dynamic_rnn(cell, inputs,
                                             lengths,
                                             ff_keep_prob=ff_keep_prob,
                                             recur_keep_prob=recur_keep_prob,
                                             dtype=tf.float32)
      top_recur.set_shape([tf.Dimension(None), tf.Dimension(None), tf.Dimension(self.recur_size)])
    return top_recur, end_recur
  
  #=============================================================
  def soft_attn(self, top_recur):
    """"""
    
    reuse = (self.moving_params is not None) or None
    
    input_size = top_recur.get_shape().as_list()[-1]
    with tf.variable_scope('MLP', reuse=reuse):
      head_mlp, dep_mlp = self.MLP(top_recur, self.info_mlp_size,
                                   func=self.info_func,
                                   keep_prob=self.info_keep_prob,
                                   n_splits=2)
    with tf.variable_scope('Arcs', reuse=reuse):
      arc_logits = self.bilinear_classifier(dep_mlp, head_mlp, keep_prob=self.info_keep_prob)
      arc_prob = self.softmax(arc_logits)
      head_lin = tf.batch_matmul(arc_prob, top_recur)
      top_recur = tf.concat(2, [top_recur, head_lin])
    top_recur.set_shape([tf.Dimension(None), tf.Dimension(None), tf.Dimension(4*self.recur_size)])
    return top_recur

  #=============================================================
  def linear(self, inputs, output_size, n_splits=1, add_bias=False):
    """"""
    
    n_dims = len(inputs.get_shape().as_list())
    batch_size = tf.shape(inputs)[0]
    bucket_size = tf.shape(inputs)[1]
    input_size = inputs.get_shape().as_list()[-1]
    output_shape = tf.pack([batch_size] + [bucket_size]*(n_dims-2) + [output_size])
    shape_to_set = [tf.Dimension(None)]*(n_dims-1) + [tf.Dimension(output_size)]
    
    if self.moving_params is None:
      keep_prob = self.info_keep_prob
    else:
      keep_prob = 1
    
    if keep_prob < 1:
      noise_shape = tf.pack([batch_size] + [1]*(n_dims-2) + [input_size])
      inputs = tf.nn.dropout(inputs, keep_prob, noise_shape=noise_shape)

    lin = linalg.linear(inputs,
                        output_size,
                        n_splits=n_splits,
                        add_bias=add_bias,
                        moving_params=self.moving_params)
    if n_splits == 1:
      lin = [lin]
    for i, split in enumerate(lin):
      split.set_shape(shape_to_set)
    if n_splits == 1:
      return lin[0]
    else:
      return lin

  #=============================================================
  def softmax(self, inputs):
    """"""
    
    input_shape = tf.shape(inputs)
    batch_size = input_shape[0]
    bucket_size = input_shape[1]
    input_size = input_shape[2]
    inputs = tf.reshape(inputs, tf.pack([-1, input_size]))
    probs = tf.nn.softmax(inputs)
    probs = tf.reshape(probs, tf.pack([batch_size, bucket_size, input_size]))
    return probs
  
  #=============================================================
  def MLP(self, inputs, output_size, func=None, keep_prob=None, n_splits=1):
    """"""
    
    n_dims = len(inputs.get_shape().as_list())
    batch_size = tf.shape(inputs)[0]
    bucket_size = tf.shape(inputs)[1]
    input_size = inputs.get_shape().as_list()[-1]
    output_shape = tf.pack([batch_size] + [bucket_size]*(n_dims-2) + [output_size])
    shape_to_set = [tf.Dimension(None)]*(n_dims-1) + [tf.Dimension(output_size)]
    if func is None:
      func = self.mlp_func
    
    if self.moving_params is None:
      if keep_prob is None:
        keep_prob = self.mlp_keep_prob
    else:
      keep_prob = 1
    if keep_prob < 1:
      noise_shape = tf.pack([batch_size] + [1]*(n_dims-2) + [input_size])
      inputs = tf.nn.dropout(inputs, keep_prob, noise_shape=noise_shape)
    
    linear = linalg.linear(inputs,
                        output_size,
                        n_splits=n_splits * (1+(func.__name__ in ('gated_tanh', 'gated_identity'))),
                        add_bias=True,
                        moving_params=self.moving_params)
    if func.__name__ in ('gated_tanh', 'gated_identity'):
      linear = [tf.concat(n_dims-1, [lin1, lin2]) for lin1, lin2 in zip(linear[:len(linear)//2], linear[len(linear)//2:])]
    if n_splits == 1:
      linear = [linear]
    for i, split in enumerate(linear):
      split = func(split)
      split.set_shape(shape_to_set)
      linear[i] = split
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
    output_size = self.attn_mlp_size
    output_shape = tf.pack([batch_size, bucket_size, bucket_size, output_size])
    shape_to_set = [tf.Dimension(None), tf.Dimension(None), tf.Dimension(None), tf.Dimension(output_size)]
    
    if self.moving_params is None:
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
    if n_splits == 1:
      return top_mlps[0]
    else:
      return top_mlps
  
  #=============================================================
  def linear_classifier(self, inputs, n_classes, add_bias=True, keep_prob=None):
    """"""
    
    n_dims = len(inputs.get_shape().as_list())
    batch_size = tf.shape(inputs)[0]
    bucket_size = tf.shape(inputs)[1]
    input_size = inputs.get_shape().as_list()[-1]
    output_size = n_classes
    output_shape = tf.pack([batch_size] + [bucket_size]*(n_dims-2) + [output_size])
    
    if self.moving_params is None:
      if keep_prob is None:
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
  def bilinear_classifier(self, inputs1, inputs2, add_bias1=True, add_bias2=False, keep_prob=None):
    """"""
    
    input_shape = tf.shape(inputs1)
    batch_size = input_shape[0]
    bucket_size = input_shape[1]
    input_size = inputs1.get_shape().as_list()[-1]
    
    if self.moving_params is None:
      if keep_prob is None:
        keep_prob = self.mlp_keep_prob
    else:
      keep_prob = 1
    if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
      noise_shape = tf.pack([batch_size, 1, input_size])
      # Experimental
      #inputs1 = tf.nn.dropout(inputs1, keep_prob if add_bias2 else tf.sqrt(keep_prob), noise_shape=noise_shape)
      #inputs2 = tf.nn.dropout(inputs2, keep_prob if add_bias1 else tf.sqrt(keep_prob), noise_shape=noise_shape)
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
  def diagonal_bilinear_classifier(self, inputs1, inputs2, add_bias1=True, add_bias2=False):
    """"""
    
    input_shape = tf.shape(inputs1)
    batch_size = input_shape[0]
    bucket_size = input_shape[1]
    input_size = inputs1.get_shape().as_list()[-1]
    shape_to_set = tf.pack([batch_size, bucket_size, input_size+1])
    
    if self.moving_params is None:
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
  def tag_argmax(self, tag_probs, tokens_to_keep):
    """"""
    
    return np.argmax(tag_probs[:,Vocab.ROOT:], axis=1)+Vocab.ROOT
  
  #=============================================================
  def parse_argmax(self, parse_probs, tokens_to_keep):
    """"""
    
    if self.ensure_tree:
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
    else:
      tokens_to_keep[0] = True
      length = np.sum(tokens_to_keep)
      # block and pad heads
      parse_probs = parse_probs * tokens_to_keep
      parse_preds = np.argmax(parse_probs, axis=1)
      return parse_preds
  
  #=============================================================
  def rel_argmax(self, rel_probs, tokens_to_keep):
    """"""
    
    if self.ensure_tree:
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
    else:
      rel_probs[:,Vocab.PAD] = 0
      rel_preds = np.argmax(rel_probs, axis=1)
      return rel_preds
  
  #=============================================================
  def __call__(self, inputs, targets, moving_params=None):
    """"""
    
    raise NotImplementedError()
  
  #=============================================================
  @property
  def global_sigmoid(self):
    return self._global_sigmoid
