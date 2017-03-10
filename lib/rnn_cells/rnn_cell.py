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
class RNNCell(BaseCell):
  """"""
  
  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""
    
    super(RNNCell, self).__init__(*args, **kwargs)
  
  #=============================================================
  def __call__(self, inputs, state, scope=None):
    """"""
    
    if self.recur_diag_bilin:
      inputs1, inputs2 = tf.split(1, 2, inputs)
      inputs = tf.concat(1, [inputs1*inputs2, inputs1, inputs2])
    with tf.variable_scope(scope or type(self).__name__):
      hidden_act = linalg.linear([inputs, state],
                                 self.output_size,
                                 add_bias=False,
                                 moving_params=self.moving_params)
      hidden = self.recur_func(hidden_act)
    return hidden, hidden
  
  #=============================================================
  @property
  def state_size(self):
    return self.output_size
  
