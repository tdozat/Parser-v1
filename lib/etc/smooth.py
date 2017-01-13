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

import numpy as np

def smooth(a, beta=.9):
  b = np.empty_like(a)
  b[0] = a[0]
  for i in xrange(1, len(a)):
    beta_i = beta * (1-beta**i) / (1-beta**(i+1))
    b[i] = beta_i * b[i-1] + (1-beta_i) * a[i]
  return b
