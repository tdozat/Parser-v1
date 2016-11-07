#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    self.moving_params = kwargs.pop('moving_params', None)
    super(BaseCell, self).__init__(*args, **kwargs)
    self._output_size = self.recur_size
    self._input_size = input_size if input_size is not None else self.output_size
  
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
  def state_size(self):
    raise NotImplementedError()