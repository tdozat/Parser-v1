#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from vocab import Vocab
from lib.models import NN

#***************************************************************
class BaseParser(NN):
  """"""
  
  #=============================================================
  def __call__(self, dataset, moving_params=None):
    """"""
    
    raise NotImplementedError
  
  #=============================================================
  def prob_argmax(self, parse_probs, rel_probs, tokens_to_keep):
    """"""
    
    raise NotImplementedError
  
  #=============================================================
  def sanity_check(self, inputs, targets, predictions, vocabs, fileobject):
    """"""
    
    for tokens, golds, parse_preds, rel_preds in zip(inputs, targets, predictions[0], predictions[1]):
      for l, (token, gold, parse, rel) in enumerate(zip(tokens, golds, parse_preds, rel_preds)):
        if token[0] > 1:
          word = vocabs[0][token[0]]
          glove = vocabs[0].get_embed(token[1])
          tag = vocabs[1][token[2]]
          gold_tag = gold[0]
          pred_parse = parse
          pred_rel = vocabs[2][rel]
          gold_parse = gold[1]
          gold_rel = vocabs[2][gold[2]]
          fileobject.write('%d\t%s\t%s\t%s\t%s\t_\t%d\t%s\t%d\t%s\n' % (l, word, glove, tag, gold_tag, pred_parse, pred_rel, gold_parse, gold_rel))
      fileobject.write('\n')
    return
  
  #=============================================================
  def validate(self, mb_inputs, mb_targets, mb_probs):
    """"""
    
    sents = []
    mb_parse_probs, mb_rel_probs = mb_probs
    for inputs, targets, parse_probs, rel_probs in zip(mb_inputs, mb_targets, mb_parse_probs, mb_rel_probs):
      tokens_to_keep = np.greater(inputs[:,0], Vocab.ROOT)
      length = np.sum(tokens_to_keep)
      parse_preds, rel_preds = self.prob_argmax(parse_probs, rel_probs, tokens_to_keep)
      
      sent = -np.ones( (length, 9), dtype=int)
      tokens = np.arange(1, length+1)
      sent[:,0] = tokens
      sent[:,1:4] = inputs[tokens]
      sent[:,4] = targets[tokens,0]
      sent[:,5] = parse_preds[tokens]
      sent[:,6] = rel_preds[tokens]
      sent[:,7:] = targets[tokens, 1:]
      sents.append(sent)
    return sents
  
  #=============================================================
  @staticmethod
  def evaluate(filename, punct=NN.PUNCT):
    """"""
    
    correct = {'UAS': [], 'LAS': []}
    with open(filename) as f:
      for line in f:
        line = line.strip().split('\t')
        if len(line) == 10 and line[4] not in punct:
          correct['UAS'].append(0)
          correct['LAS'].append(0)
          if line[6] == line[8]:
            correct['UAS'][-1] = 1
            if line[7] == line[9]:
              correct['LAS'][-1] = 1
    correct = {k:np.array(v) for k, v in correct.iteritems()}
    return 'UAS: %.2f    LAS: %.2f\n' % (np.mean(correct['UAS']) * 100, np.mean(correct['LAS']) * 100), correct
  
  #=============================================================
  @property
  def input_idxs(self):
    return (0, 1, 2)
  @property
  def target_idxs(self):
    return (3, 4, 5)
