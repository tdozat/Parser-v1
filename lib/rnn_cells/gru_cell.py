#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
      with tf.variable_scope('Gates'):
        linear = linalg.linear([inputs, state],
                               self.output_size,
                               add_bias=True,
                               n_splits=2,
                               moving_params=self.moving_params)
        update_act, reset_act = linear
        update_gate = linalg.sigmoid(update_act-self.forget_bias)
        reset_gate = linalg.sigmoid(reset_act)
        reset_state = reset_gate * state
      with tf.variable_scope('Candidate'):
        hidden_act = linalg.linear([inputs, reset_state],
                                   self.output_size,
                                   add_bias=False,
                                   moving_params=self.moving_params)
        hidden_tilde = self.recur_func(hidden_act)
      hidden = update_gate * state + (1-update_gate) * hidden_tilde
    return hidden, hidden
  
  #=============================================================
  @property
  def state_size(self):
    return self.output_size
