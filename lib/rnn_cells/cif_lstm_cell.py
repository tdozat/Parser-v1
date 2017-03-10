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
class CifLSTMCell(BaseCell):
  """"""
  
  #=============================================================
  def __call__(self, inputs, state, scope=None):
    """"""
    
    if self.recur_diag_bilin:
      inputs1, inputs2 = tf.split(1, 2, inputs)
      inputs = tf.concat(1, [inputs1*inputs2, inputs1, inputs2])
    with tf.variable_scope(scope or type(self).__name__):
      cell_tm1, hidden_tm1 = tf.split(1, 2, state)
      linear = linalg.linear([inputs, hidden_tm1],
                          self.output_size,
                          add_bias=True,
                          n_splits=3,
                          moving_params=self.moving_params)
      cell_act, update_act, output_act = linear
      
      cell_tilde_t = cell_act
      update_gate = linalg.sigmoid(update_act-self.forget_bias)
      output_gate = linalg.sigmoid(output_act)
      cell_t = update_gate * cell_tilde_t + (1-update_gate) * cell_tm1
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
  
