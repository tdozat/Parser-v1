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

import numpy as np
import tensorflow as tf

#***************************************************************
sig_const = np.arctanh(1/3)
tanh_const = np.arctanh(np.sqrt(1/3))

def tanh(x):
  return tf.tanh(x)
def sigmoid(x):
  return (tf.tanh(x)+1)/2

#===============================================================
def orthonormal_initializer(input_size, output_size):
  """"""
  
  print(tf.get_variable_scope().name)
  I = np.eye(output_size)
  lr = .1
  eps = .05/(output_size + input_size)
  success = False
  tries = 0
  while not success and tries < 10:
    Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    for i in xrange(100):
      QTQmI = Q.T.dot(Q) - I
      loss = np.sum(QTQmI**2 / 2)
      Q2 = Q**2
      Q -= lr*Q.dot(QTQmI) / (np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
      if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
        tries += 1
        lr /= 2
        break
    success = True
  if success:
    print('Orthogonal pretrainer loss: %.2e' % loss)
  else:
    print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
    Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
  return Q.astype(np.float32)

#===============================================================
def linear(inputs, output_size, add_bias=True, n_splits=1, initializer=None, scope=None, moving_params=None):
  """"""
  
  if not isinstance(inputs, (list, tuple)):
    inputs = [inputs]
  output_size *= n_splits
  
  with tf.variable_scope(scope or 'Linear'):
    # Reformat the input
    total_input_size = 0
    shapes = [a.get_shape().as_list() for a in inputs]
    for shape in shapes:
      total_input_size += shape[-1]
    input_shape = tf.shape(inputs[0])
    output_shape = []
    for i in xrange(len(shapes[0])):
      output_shape.append(input_shape[i])
    output_shape[-1] = output_size
    output_shape = tf.pack(output_shape)
    for i, (input_, shape) in enumerate(zip(inputs, shapes)):
      inputs[i] = tf.reshape(input_, [-1, shape[-1]])
    concatenation = tf.concat(1, inputs)
    
    # Get the matrix
    if initializer is None and moving_params is None:
      mat = orthonormal_initializer(total_input_size, output_size//n_splits)
      mat = np.concatenate([mat]*n_splits, axis=1)
      initializer = tf.constant_initializer(mat)
    matrix = tf.get_variable('Weights', [total_input_size, output_size], initializer=initializer)
    if moving_params is not None:
      matrix = moving_params.average(matrix)
    else:
      tf.add_to_collection('Weights', matrix)
    
    # Get the bias
    if add_bias:
      bias = tf.get_variable('Biases', [output_size], initializer=tf.zeros_initializer)
      if moving_params is not None:
        bias = moving_params.average(bias)
    else:
      bias = 0
    
    # Do the multiplication
    new = tf.matmul(concatenation, matrix) + bias
    new = tf.reshape(new, output_shape)
    new.set_shape([tf.Dimension(None) for _ in xrange(len(shapes[0])-1)] + [tf.Dimension(output_size)])
    if n_splits > 1:
      return tf.split(len(new.get_shape().as_list())-1, n_splits, new)
    else:
      return new

#===============================================================
def bilinear(inputs1, inputs2, output_size, add_bias2=True, add_bias1=True, add_bias=False, initializer=None, scope=None, moving_params=None):
  """"""
  
  with tf.variable_scope(scope or 'Bilinear'):
    # Reformat the inputs
    ndims = len(inputs1.get_shape().as_list())
    inputs1_shape = tf.shape(inputs1)
    inputs1_bucket_size = inputs1_shape[ndims-2]
    inputs1_size = inputs1.get_shape().as_list()[-1]
    
    inputs2_shape = tf.shape(inputs2)
    inputs2_bucket_size = inputs2_shape[ndims-2]
    inputs2_size = inputs2.get_shape().as_list()[-1]
    output_shape = []
    batch_size = 1
    for i in xrange(ndims-2):
      batch_size *= inputs1_shape[i]
      output_shape.append(inputs1_shape[i])
    output_shape.append(inputs1_bucket_size)
    output_shape.append(output_size)
    output_shape.append(inputs2_bucket_size)
    output_shape = tf.pack(output_shape)
    inputs1 = tf.reshape(inputs1, tf.pack([batch_size, inputs1_bucket_size, inputs1_size]))
    inputs2 = tf.reshape(inputs2, tf.pack([batch_size, inputs2_bucket_size, inputs2_size]))
    if add_bias1:
      inputs1 = tf.concat(2, [inputs1, tf.ones(tf.pack([batch_size, inputs1_bucket_size, 1]))])
    if add_bias2:
      inputs2 = tf.concat(2, [inputs2, tf.ones(tf.pack([batch_size, inputs2_bucket_size, 1]))])
    
    # Get the matrix
    if initializer is None and moving_params is None:
      mat = orthonormal_initializer(inputs1_size+add_bias1, inputs2_size+add_bias2)[:,None,:]
      mat = np.concatenate([mat]*output_size, axis=1)
      initializer = tf.constant_initializer(mat)
    weights = tf.get_variable('Weights', [inputs1_size+add_bias1, output_size, inputs2_size+add_bias2], initializer=initializer)
    if moving_params is not None:
      weights = moving_params.average(weights)
    else:
      tf.add_to_collection('Weights', weights)
    
    # Do the multiplications
    # (bn x d) (d x rd) -> (bn x rd)
    lin = tf.matmul(tf.reshape(inputs1, [-1, inputs1_size+add_bias1]),
                        tf.reshape(weights, [inputs1_size+add_bias1, -1]))
    # (b x nr x d) (b x n x d)T -> (b x nr x n)
    bilin = tf.batch_matmul(tf.reshape(lin, tf.pack([batch_size, inputs1_bucket_size*output_size, inputs2_size+add_bias2])),
                                   inputs2, adj_y=True)
    # (bn x r x n)
    bilin = tf.reshape(bilin, tf.pack([-1, output_size, inputs2_bucket_size]))
    # (b x n x r x n)
    bilin = tf.reshape(bilin, output_shape)
    
    # Get the bias
    if add_bias:
      bias = tf.get_variable('Biases', [output_size], initializer=tf.zeros_initializer)
      if moving_params is not None:
        bias = moving_params.average(bias)
      bilin += tf.expand_dims(bias, 1)
    
    return bilin

#===============================================================
def diagonal_bilinear(inputs1, inputs2, output_size, add_bias2=True, add_bias1=True, add_bias=False, initializer=None, scope=None, moving_params=None):
  """"""
  
  with tf.variable_scope(scope or 'Bilinear'):
    # Reformat the inputs
    ndims = len(inputs1.get_shape().as_list())
    inputs1_shape = tf.shape(inputs1)
    inputs2_shape = tf.shape(inputs2)
    inputs1_bucket_size = inputs1_shape[ndims-2]
    inputs2_bucket_size = inputs2_shape[ndims-2]

    inputs1_size = inputs1.get_shape().as_list()[-1]
    inputs2_size = inputs2.get_shape().as_list()[-1]
    assert inputs1_size == inputs2_size
    
    output_shape = []
    batch_size = 1
    for i in xrange(ndims-2):
      batch_size *= inputs1_shape[i]
      output_shape.append(inputs1_shape[i])
    output_shape.append(inputs1_bucket_size)
    output_shape.append(output_size)
    output_shape.append(inputs2_bucket_size)
    output_shape = tf.pack(output_shape)
    inputs1 = tf.reshape(inputs1, tf.pack([batch_size, inputs1_bucket_size, inputs1_size]))
    inputs2 = tf.reshape(inputs2, tf.pack([batch_size, inputs2_bucket_size, inputs2_size]))
    inputs1.set_shape([tf.Dimension(None)]*2 + [tf.Dimension(inputs1_size)])
    inputs2.set_shape([tf.Dimension(None)]*2 + [tf.Dimension(inputs2_size)])
    
    inputs = broadcast_mult(inputs1, inputs2)
    with tf.variable_scope('Bilinear'):
      bilin = linear(inputs, output_size, add_bias=add_bias, initializer=initializer, scope=scope, moving_params=moving_params)
    with tf.variable_scope('Linear1'):
      lin1 = linear(inputs1, output_size, add_bias=False, initializer=initializer, scope=scope, moving_params=moving_params)
      lin1 = tf.expand_dims(lin1, 2)
    with tf.variable_scope('Linear2'):
      lin2 = linear(inputs2, output_size, add_bias=False, initializer=initializer, scope=scope, moving_params=moving_params)
      lin2 = tf.expand_dims(lin2, 1)

    bilin = tf.transpose(bilin+lin1+lin2, [0,1,3,2])
    
    return bilin
  
#===============================================================
def layer_norm(inputs, beta_start=0, gamma_start=1, scope=None, moving_params=None):
  """"""
  
  with tf.variable_scope(scope or "Layer_norm"):
    gamma = tf.get_variable('Gamma', shape=[],
                            initializer=tf.constant_initializer(gamma_start))
    beta = tf.get_variable('Beta', shape=[],
                            initializer=tf.constant_initializer(beta_start))
    if moving_params is not None:
      gamma = moving_params.average(gamma)
      beta = moving_params.average(beta)
    mean, var = tf.nn.moments(inputs, 1, keep_dims=True)
    inputs = gamma * (inputs-mean) / tf.sqrt(var+self.epsilon) + beta
    return inputs
  
#===============================================================
def broadcast_add(inputs1, inputs2):
  """"""
  
  inputs1_shape = tf.shape(inputs1)
  inputs_size = inputs1.get_shape().as_list()[-1]
  inputs2_shape = tf.shape(inputs2)
  inputs1 = tf.transpose(inputs1, [0,2,1])
  inputs2 = tf.transpose(inputs2, [0,2,1])
  inputs1 = tf.reshape(inputs1, tf.pack([-1,inputs1_shape[1],1]))
  inputs2 = tf.reshape(inputs2, tf.pack([-1,1,inputs2_shape[1]]))
  inputs = inputs1 + inputs2
  inputs = tf.reshape(inputs, [inputs1_shape[0], inputs1_shape[2],  inputs1_shape[1], inputs2_shape[1]])
  inputs = tf.transpose(inputs, [0,2,3,1])
  inputs.set_shape([tf.Dimension(None)]*3 + [tf.Dimension(inputs_size)])
  return inputs
  
#===============================================================
def broadcast_sub(inputs1, inputs2):
  """"""
  
  inputs1_shape = tf.shape(inputs1)
  inputs_size = inputs1.get_shape().as_list()[-1]
  inputs2_shape = tf.shape(inputs2)
  inputs1 = tf.transpose(inputs1, [0,2,1])
  inputs2 = tf.transpose(inputs2, [0,2,1])
  inputs1 = tf.reshape(inputs1, tf.pack([-1,inputs1_shape[1],1]))
  inputs2 = tf.reshape(inputs2, tf.pack([-1,1,inputs2_shape[1]]))
  inputs = inputs1 - inputs2
  inputs = tf.reshape(inputs, [inputs1_shape[0], inputs1_shape[2], inputs1_shape[1], inputs2_shape[1]])
  inputs = tf.transpose(inputs, [0,2,3,1])
  inputs.set_shape([tf.Dimension(None)]*3 + [tf.Dimension(inputs_size)])
  return inputs

#===============================================================
def broadcast_mult(inputs1, inputs2):
  """"""
  
  inputs1_shape = tf.shape(inputs1)
  inputs_size = inputs1.get_shape().as_list()[-1]
  inputs2_shape = tf.shape(inputs2)
  inputs1 = tf.transpose(inputs1, [0,2,1])
  inputs2 = tf.transpose(inputs2, [0,2,1])
  inputs1 = tf.reshape(inputs1, tf.pack([-1,inputs1_shape[1],1]))
  inputs2 = tf.reshape(inputs2, tf.pack([-1,1,inputs2_shape[1]]))
  inputs = inputs1 * inputs2
  inputs = tf.reshape(inputs, tf.pack([inputs1_shape[0], inputs1_shape[2],  inputs1_shape[1], inputs2_shape[1]]))
  inputs = tf.transpose(inputs, [0,2,3,1])
  inputs.set_shape([tf.Dimension(None)]*3 + [tf.Dimension(inputs_size)])
  return inputs

#***************************************************************
if __name__ == '__main__':
  """"""
  
  x1 = tf.Variable(np.random.randn(5,5).astype(np.float32))
  x2 = tf.Variable(np.random.randn(5,2).astype(np.float32))
  z = linear([x1, x2], 10)
  zz = bilinear(x1, x2, 10)
  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    sess.run(z)
    sess.run(zz)
