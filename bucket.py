#!/usr/bin/env python
# -*- coding: UTF-8 -*-

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

from configurable import Configurable
from lib.linalg import linear
from lib.models.nn import NN

#***************************************************************
class Bucket(Configurable):
  """"""
  
  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""
    
    super(Bucket, self).__init__(*args, **kwargs)
    self._size = None
    self._data = None
    self._sents = None
    return
  
  #=============================================================
  def reset(self, size, pad=False):
    """"""
    
    self._size = size
    if pad:
      self._data = [(0,)]
      self._sents = [('',)]
    else:
      self._data = []
      self._sents = []
    return
  
  #=============================================================
  def add(self, sent):
    """"""
    
    if isinstance(self._data, np.ndarray):
      raise TypeError("The buckets have already been finalized, you can't add more")
    if len(sent) > self.size and self.size != -1:
      raise ValueError('Bucket of size %d received sequence of len %d' % (self.size, len(sent)))
    
    words = [word[0] for word in sent][1:] # remove root
    idxs = [word[1:] for word in sent]
    self._sents.append(words)
    self._data.append(idxs)
    return len(self._data)-1
  
  #=============================================================
  def _finalize(self):
    """"""
    
    if self._data is None:
      raise ValueError('You need to reset the Buckets before finalizing them')
    
    if len(self._data) > 0:
      shape = (len(self._data), self.size, len(self._data[-1][-1]))
      data = np.zeros(shape, dtype=np.int32)
      for i, datum in enumerate(self._data):
        datum = np.array(datum)
        data[i, 0:len(datum)] = datum
      self._data = data
      self._sents = np.array(self._sents)
    else:
      self._data = np.zeros((0,1), dtype=np.float32)
      self._sents = np.zeros((0,1), dtype=str)
    print('Bucket %s is %d x %d' % ((self._name,) + self._data.shape[0:2]))
    return
  
  #=============================================================
  def __len__(self):
    return len(self._data)
  
  #=============================================================
  @property
  def size(self):
    return self._size
  @property
  def data(self):
    return self._data
  @property
  def sents(self):
    return self._sents
