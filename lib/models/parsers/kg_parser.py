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
class KGParser(BaseParser):
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
    
    word_inputs = vocabs[0].embedding_lookup(inputs[:,:,0], inputs[:,:,1], moving_params=self.moving_params)
    tag_inputs  = vocabs[1].embedding_lookup(inputs[:,:,2], moving_params=self.moving_params)
    
    top_recur = self.embed_concat(word_inputs, tag_inputs)
    for i in xrange(self.n_recur):
      with tf.variable_scope('RNN%d' % i, reuse=reuse):
        top_recur, _ = self.RNN(top_recur)
    
    top_mlp = top_recur
    with tf.variable_scope('MLP0', reuse=reuse):
      parse_mlp, rel_mlp = self.double_MLP(top_mlp, n_splits=2)
    for i in xrange(1,self.n_mlp):
      with tf.variable_scope('ParseMLP%d' % i, reuse=reuse):
        parse_mlp = self.MLP(parse_mlp)
      with tf.variable_scope('RelMLP%d' % i, reuse=reuse):
        rel_mlp = self.MLP(rel_mlp)
    
    with tf.variable_scope('Parses', reuse=reuse):
      parse_logits = tf.squeeze(self.linear_classifier(parse_mlp, 1))
      parse_output = self.output(parse_logits, targets[:,:,1])
      if moving_params is None:
        predictions = targets[:,:,1]
      else:
        predictions = parse_output['predictions']
    with tf.variable_scope('Rels', reuse=reuse):
      rel_logits, rel_logits_cond = self.conditional_linear_classifier(rel_mlp, len(vocabs[2]), predictions)
      rel_output = self.output(rel_logits, targets[:,:,2])
      rel_output['probabilities'] = self.conditional_probabilities(rel_logits_cond, transpose=False)
    
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
    
    output['embed'] = tf.pack([word_inputs, tag_inputs])
    output['recur'] = top_recur
    output['parse_mlp'] = parse_mlp
    output['rel_mlp'] = rel_mlp
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
