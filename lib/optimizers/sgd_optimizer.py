#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from lib.optimizers.base_optimizer import BaseOptimizer

#***************************************************************
class SGDOptimizer(BaseOptimizer):
  """"""
  
  #=============================================================
  def _apply_dense(self, cache):
    """"""
    
    g_t = cache['g_t']
    cache['s_t'] = self.learning_rate * g_t
    return cache
  
  #=============================================================
  def _apply_sparse(self, cache):
    """"""
    
    g_t, idxs = cache['g_t'], cache['idxs']
    idxs, idxs_ = tf.unique(idxs)
    g_t_ = tf.unsorted_segment_sum(g_t, idxs_, tf.size(idxs))
    
    cache['g_t'] = g_t_
    cache['idxs'] = idxs
    cache['s_t'] = self.learning_rate * g_t_
    
    return cache
  
