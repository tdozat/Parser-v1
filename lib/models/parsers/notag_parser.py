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

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from vocab import Vocab
from lib.models.parsers.base_parser import BaseParser

#***************************************************************
class NoTagParser(BaseParser):
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
    
    top_recur = embed_inputs = self.embed_concat(word_inputs+pret_inputs)
    for i in xrange(self.n_recur):
      with tf.variable_scope('RNN%d' % i, reuse=reuse):
        top_recur, _ = self.RNN(top_recur)
    
    with tf.variable_scope('MLP', reuse=reuse):
      dep_mlp, head_mlp = self.MLP(top_recur, self.class_mlp_size+self.attn_mlp_size, n_splits=2)
      dep_arc_mlp, dep_rel_mlp = dep_mlp[:,:,:self.attn_mlp_size], dep_mlp[:,:,self.attn_mlp_size:]
      head_arc_mlp, head_rel_mlp = head_mlp[:,:,:self.attn_mlp_size], head_mlp[:,:,self.attn_mlp_size:]
    
    with tf.variable_scope('Arcs', reuse=reuse):
      arc_logits = self.bilinear_classifier(dep_arc_mlp, head_arc_mlp)
      arc_output = self.output(arc_logits, targets[:,:,1])
      if moving_params is None:
        predictions = targets[:,:,1]
      else:
        predictions = arc_output['predictions']
    with tf.variable_scope('Rels', reuse=reuse):
      rel_logits, rel_logits_cond = self.conditional_bilinear_classifier(dep_rel_mlp, head_rel_mlp, len(vocabs[2]), predictions)
      rel_output = self.output(rel_logits, targets[:,:,2])
      rel_output['probabilities'] = self.conditional_probabilities(rel_logits_cond)
    
    output = {}
    output['probabilities'] = tf.tuple([arc_output['probabilities'],
                                        rel_output['probabilities']])
    output['predictions'] = tf.pack([arc_output['predictions'],
                                     rel_output['predictions']])
    output['correct'] = arc_output['correct'] * rel_output['correct']
    output['tokens'] = arc_output['tokens']
    output['n_correct'] = tf.reduce_sum(output['correct'])
    output['n_tokens'] = self.n_tokens
    output['accuracy'] = output['n_correct'] / output['n_tokens']
    output['loss'] = arc_output['loss'] + rel_output['loss'] 
    if self.word_l2_reg > 0:
      output['loss'] += word_loss
    
    output['embed'] = embed_inputs
    output['recur'] = top_recur
    output['dep_arc'] = dep_arc_mlp
    output['head_dep'] = head_arc_mlp
    output['dep_rel'] = dep_rel_mlp
    output['head_rel'] = head_rel_mlp
    output['arc_logits'] = arc_logits
    output['rel_logits'] = rel_logits
    return output
  
  #=============================================================
  def prob_argmax(self, parse_probs, rel_probs, tokens_to_keep):
    """"""
    
    parse_preds = self.parse_argmax(parse_probs, tokens_to_keep)
    rel_probs = rel_probs[np.arange(len(parse_preds)), parse_preds]
    rel_preds = self.rel_argmax(rel_probs, tokens_to_keep)
    return parse_preds, rel_preds
