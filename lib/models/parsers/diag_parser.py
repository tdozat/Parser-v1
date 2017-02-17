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

from vocab import Vocab
from lib.linalg import linear
from lib.models.parsers.base_parser import BaseParser

#***************************************************************
class DiagParser(BaseParser):
  """"""
  
  #=============================================================
  def __call__(self, dataset, moving_params=None):
    """"""
    
    vocabs = dataset.vocabs
    inputs = dataset.inputs
    targets = dataset.targets
    
    reuse = (moving_params is not None)
    self.tokens_to_keep3D = tf.expand_dims(tf.to_float(tf.greater(inputs[:,:,0], vocabs[0].ROOT)), 2)
    self.sequence_lengths = tf.reshape(tf.reduce_sum(self.tokens_to_keep3D, [1, 2]), [-1,1])
    self.n_tokens = tf.reduce_sum(self.sequence_lengths)
    self.moving_params = moving_params
    
    word_inputs, pret_inputs = vocabs[0].embedding_lookup(inputs[:,:,0], inputs[:,:,1], moving_params=self.moving_params)
    tag_inputs  = vocabs[1].embedding_lookup(inputs[:,:,2], moving_params=self.moving_params)
    
    embed_inputs = top_recur = self.embed_concat(word_inputs+pret_inputs, tag_inputs)
    if self.moving_params is None:
      if self.drop_gradually:
        s = self.global_sigmoid
        keep_prob = s + (1-s)*self.ff_keep_prob
      else:
        keep_prob = self.ff_keep_prob
    else:
      keep_prob = 1.
    batch_size = tf.shape(inputs)[0]
    for i in xrange(self.n_recur):
      if self.moving_params is None:
        input_size = top_recur.get_shape().as_list()[-1]
        fw_keep_mask = tf.nn.dropout(tf.ones(tf.pack([batch_size, input_size])), keep_prob=keep_prob)
        if self.recur_bidir:
          bw_keep_mask = tf.nn.dropout(tf.ones(tf.pack([batch_size, input_size])), keep_prob=keep_prob)
        else:
          bw_keep_mask = None
      else:
        fw_keep_mask = bw_keep_mask = None
      with tf.variable_scope('RNN%d' % i, reuse=reuse):
        top_recur, _ = self.RNN(top_recur, fw_keep_mask=fw_keep_mask, bw_keep_mask=bw_keep_mask)
    
    top_mlp = top_recur
    if self.n_mlp > 0:
      with tf.variable_scope('MLP0', reuse=reuse):
        dep_mlp, head_dep_mlp, rel_mlp, head_rel_mlp = self.MLP(top_mlp, n_splits=4)
      for i in xrange(1,self.n_mlp):
        with tf.variable_scope('DepMLP%d' % i, reuse=reuse):
          dep_mlp = self.MLP(dep_mlp)
        with tf.variable_scope('HeadDepMLP%d' % i, reuse=reuse):
          head_dep_mlp = self.MLP(head_dep_mlp)
        with tf.variable_scope('RelMLP%d' % i, reuse=reuse):
          rel_mlp = self.MLP(rel_mlp)
        with tf.variable_scope('HeadRelMLP%d' % i, reuse=reuse):
          head_rel_mlp = self.MLP(head_rel_mlp)
    else:
      dep_mlp = head_dep_mlp = rel_mlp = head_rel_mlp = top_mlp
    
    with tf.variable_scope('Parses', reuse=reuse):
      parse_logits = self.diagonal_bilinear_classifier(dep_mlp, head_dep_mlp)
      parse_output = self.output(parse_logits, targets[:,:,1])
      if moving_params is None:
        predictions = targets[:,:,1]
      else:
        predictions = parse_output['predictions']
    with tf.variable_scope('Rels', reuse=reuse):
      rel_logits, rel_logits_cond = self.conditional_diagonal_bilinear_classifier(rel_mlp, head_rel_mlp, len(vocabs[2]), predictions)
      rel_output = self.output(rel_logits, targets[:,:,2])
      rel_output['probabilities'] = self.conditional_probabilities(rel_logits_cond)
    
    output = {}
    output['probabilities'] = tf.tuple([parse_output['probabilities'],
                                        rel_output['probabilities']])
    output['predictions'] = tf.pack([parse_output['predictions'],
                                     rel_output['predictions']])
    output['correct'] = parse_output['correct'] * rel_output['correct']
    output['tokens'] = parse_output['tokens']
    output['n_correct'] = tf.reduce_sum(output['correct'])
    output['n_tokens'] = self.n_tokens
    output['accuracy'] = output['n_correct'] / output['n_tokens']
    output['loss'] = parse_output['loss'] + rel_output['loss'] 
    
    output['embed'] = embed_inputs
    output['recur'] = top_recur
    output['dep'] = dep_mlp
    output['head_dep'] = head_dep_mlp
    output['rel'] = rel_mlp
    output['head_rel'] = head_rel_mlp
    output['parse_logits'] = parse_logits
    output['rel_logits'] = rel_logits
    return output
  
  #=============================================================
  def prob_argmax(self, parse_probs, rel_probs, tokens_to_keep):
    """"""
    
    parse_preds = self.parse_argmax(parse_probs, tokens_to_keep)
    rel_probs = rel_probs[np.arange(len(parse_preds)), parse_preds]
    rel_preds = self.rel_argmax(rel_probs, tokens_to_keep)
    return parse_preds, rel_preds
