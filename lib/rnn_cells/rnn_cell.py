#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
  