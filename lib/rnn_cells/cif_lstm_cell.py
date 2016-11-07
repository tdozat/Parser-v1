#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from lib.rnn_cells.base_cell import BaseCell
from lib import linalg

#***************************************************************
class CifLSTMCell(BaseCell):
  """"""
  
  #=============================================================
  def __call__(self, inputs, state, scope=None):
    """"""
    
    with tf.variable_scope(scope or type(self).__name__):
      cell_tm1, hidden_tm1 = tf.split(1, 2, state)
      linear = linalg.linear([inputs, hidden_tm1],
                          self.output_size,
                          add_bias=False,
                          n_splits=3,
                          moving_params=self.moving_params)
      with tf.variable_scope('Linear'):
        biases = tf.get_variable('Biases', [2*self.output_size], initializer=tf.zeros_initializer)
      biases = tf.split(0, 2, biases)
      update_bias, output_bias = biases
      cell_act, update_act, output_act = linear
      
      cell_tilde_t = cell_act
      update_gate = linalg.sigmoid(update_act+update_bias-self.forget_bias)
      output_gate = linalg.sigmoid(output_act+output_bias)
      cell_t = update_gate * cell_tilde_t + (1-update_gate) * cell_tm1
      hidden_tilde_t = self.recur_func(cell_t)
      hidden_t = hidden_tilde_t * output_gate
      
      return hidden_t, tf.concat(1, [cell_t, hidden_t])
  
  #=============================================================
  @property
  def state_size(self):
    return self.output_size * 2
  