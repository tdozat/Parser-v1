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

from configurable import Configurable

#***************************************************************
class BaseCell(Configurable):
  """"""
  
  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""
    
    input_size = kwargs.pop('input_size', None)
    output_size = kwargs.pop('output_size', None)
    recur_diag_bilin = kwargs.pop('recur_diag_bilin', False)
    self.moving_params = kwargs.pop('moving_params', None)
    super(BaseCell, self).__init__(*args, **kwargs)
    self._output_size = output_size if output_size is not None else self.recur_size
    self._input_size = input_size if input_size is not None else self.output_size
    self._recur_diag_bilin = recur_diag_bilin
  
  #=============================================================
  def __call__(self, inputs, state, scope=None):
    """"""
    
    raise NotImplementedError()
  
  #=============================================================
  def zero_state(self, batch_size, dtype):
    """"""
    
    zero_state = tf.get_variable('Zero_state',
                                 shape=self.state_size,
                                 dtype=dtype,
                                 initializer=tf.zeros_initializer)
    state = tf.reshape(tf.tile(zero_state, tf.pack([batch_size])), tf.pack([batch_size, self.state_size]))
    state.set_shape([None, self.state_size])
    return state
  
  #=============================================================
  @property
  def input_size(self):
    return self._input_size
  @property
  def output_size(self):
    return self._output_size
  @property
  def recur_diag_bilin(self):
    return self._recur_diag_bilin
  @property
  def state_size(self):
    raise NotImplementedError()
