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
# TODO try applying recur_diag_bilin to last layer only
class ShallowParser(BaseParser):
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
    tag_inputs = vocabs[1].embedding_lookup(inputs[:,:,2], moving_params=self.moving_params)
    if self.add_to_pretrained and not self.char_based:
      word_inputs += pret_inputs
    if self.word_l2_reg > 0:
      unk_mask = tf.expand_dims(tf.to_float(tf.greater(inputs[:,:,1], vocabs[0].UNK)),2)
      word_loss = self.word_l2_reg*tf.nn.l2_loss((word_inputs - pret_inputs) * unk_mask)
    embed_inputs = self.embed_concat(word_inputs, tag_inputs)
    
    top_recur = embed_inputs
    recur_diag_bilin = False#self.recur_diag_bilin and tag_inputs.get_shape().as_list()[-1] == word_inputs.get_shape().as_list()[-1]
    for i in xrange(self.n_recur):
      with tf.variable_scope('RNN%d' % i, reuse=reuse):
        top_recur, _ = self.RNN(top_recur, recur_diag_bilin=recur_diag_bilin)
        recur_diag_bilin = self.recur_diag_bilin
    if self.attn_based:
      top_recur = self.soft_attn(top_recur, recur_diag_bilin=recur_diag_bilin)
      recur_diag_bilin = False
    
    with tf.variable_scope('Arcs', reuse=reuse):
      arc_logits = self.bilinear_classifier(top_recur,top_recur)
      arc_output = self.output(arc_logits, targets[:,:,1])
      if moving_params is None:
        predictions = targets[:,:,1]
      else:
        predictions = arc_output['predictions']
    with tf.variable_scope('Rels', reuse=reuse):
      rel_logits, rel_logits_cond = self.conditional_bilinear_classifier(top_recur, top_recur, len(vocabs[2]), predictions)
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
