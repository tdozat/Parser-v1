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
class GRUCell(BaseCell):
  """"""
  
  #=============================================================
  def __call__(self, inputs, state, scope=None):
    """"""
    
    with tf.variable_scope(scope or type(self).__name__):
      cell_tm1, hidden_tm1 = tf.split(1, 2, state)
      with tf.variable_scope('Gates'):
        linear = linalg.linear([inputs, hidden_tm1],
                               self.output_size,
                               add_bias=True,
                               n_splits=2,
                               moving_params=self.moving_params)
        update_act, reset_act = linear
        update_gate = linalg.sigmoid(update_act-self.forget_bias)
        reset_gate = linalg.sigmoid(reset_act)
        reset_state = reset_gate * hidden_tm1
      with tf.variable_scope('Candidate'):
        hidden_act = linalg.linear([inputs, reset_state],
                                   self.output_size,
                                   add_bias=True,
                                   moving_params=self.moving_params)
        hidden_tilde = self.recur_func(hidden_act)
      cell_t = update_gate * cell_tm1 + (1-update_gate) * hidden_tilde
    return cell_t, tf.concat(1, [cell_t, cell_t])
  
  #=============================================================
  @property
  def state_size(self):
    return self.output_size * 2
