#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

import tensorflow as tf

from lib.rnn_cells.base_cell import BaseCell
from lib import linalg

#***************************************************************
class LSTMCell(BaseCell):
  """"""
  
  #=============================================================
  def __call__(self, inputs, state, scope=None):
    """"""
    
    with tf.variable_scope(scope or type(self).__name__):
      cell_tm1, hidden_tm1 = tf.split(1, 2, state)
      if self.recur_diag_bilin:
        inputs1, inputs2 = tf.split(1, 2, inputs)
        input_list = [inputs1*inputs2, inputs1+inputs2, hidden_tm1]
      else:
        input_list = [inputs, hidden_tm1]
      linear = linalg.linear(input_list,
                             self.output_size,
                             add_bias=False,
                             n_splits=4,
                             moving_params=self.moving_params)
      with tf.variable_scope('Linear'):
        biases = tf.get_variable('Biases', [4*self.output_size], initializer=tf.zeros_initializer)
      biases = tf.split(0, 4, biases)
      cell_act, input_act, forget_act, output_act = linear
      cell_bias, input_bias, forget_bias, output_bias = biases
      
      cell_tilde_t = linalg.tanh(cell_act+cell_bias)
      input_gate = linalg.sigmoid(input_act+input_bias)
      forget_gate = linalg.sigmoid(forget_act+forget_bias-self.forget_bias)
      output_gate = linalg.sigmoid(output_act+output_bias)
      cell_t = input_gate * cell_tilde_t + (1-forget_gate) * cell_tm1
      hidden_tilde_t = self.recur_func(cell_t)
      hidden_t = hidden_tilde_t * output_gate

      if self.hidden_include_prob < 1 and self.moving_params is None:
        hidden_mask = tf.nn.dropout(tf.ones_like(hidden_t), self.hidden_include_prob)*self.hidden_include_prob
        hidden_t = hidden_mask * hidden_t + (1-hidden_mask) * hidden_tm1
      if self.cell_include_prob < 1 and self.moving_params is None:
        cell_mask = tf.nn.dropout(tf.ones_like(cell_t), self.cell_include_prob)*self.cell_include_prob
        cell_t = cell_mask * cell_t + (1-cell_mask) * cell_tm1
      
      return hidden_t, tf.concat(1, [cell_t, hidden_t])
  
  #=============================================================
  @property
  def state_size(self):
    return self.output_size * 2
