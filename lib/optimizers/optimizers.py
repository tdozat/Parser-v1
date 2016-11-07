#!/usr/bin/env python
 
# Copyright 2015 Google Inc and 2016 Timothy Dozat. All Rights Reserved.
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

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import math_ops

#***************************************************************
class Optimizer(object):
  """ Slightly modified version of the original Optimizer class """
  
  GATE_NONE = 0
  GATE_OP = 1
  GATE_GRAPH = 2
  
  #=============================================================
  def __init__(self, use_locking, name):
    """"""
    
    if not name:
      raise ValueError("Must specify the optimizer name")
    self._use_locking = use_locking
    self._name = name
    self._slots = {}
  
  #=============================================================
  def minimize(self, loss, global_step=None, var_list=None, gate_gradients=GATE_OP,
               aggregation_method=None, colocate_gradients_with_ops=False, name=None):
    """"""
    
    grads_and_vars = self.compute_gradients(
      loss, var_list=var_list,
      gate_gradients=gate_gradients,
      aggregation_method=aggregation_method,
      colocate_gradients_with_ops=colocate_gradients_with_ops)
    return self.apply_gradients(grads_and_vars, global_step=global_step, name=name)
  
  #=============================================================
  def compute_gradients(self, loss, var_list=None, gate_gradients=GATE_OP,
                        aggregation_method=None, colocate_gradients_with_ops=False):
    """"""
    
    # Error checking
    if gate_gradients not in [Optimizer.GATE_NONE, Optimizer.GATE_OP,
                              Optimizer.GATE_GRAPH]:
      raise ValueError("gate_gradients must be one of: Optimizer.GATE_NONE, " +
        "Optimizer.GATE_OP, Optimizer.GATE_GRAPH. Not %s" % gate_gradients)
    self._assert_valid_dtypes([loss])
    if var_list is None:
      var_list = variables.trainable_variables()
    for x_tm1 in var_list:
      if not isinstance(x_tm1, variables.Variable):
        raise TypeError("Argument is not a tf.Variable: %s" % x_tm1)
    if not var_list:
      raise ValueError("No variables to optimize")
    
    # The actual stuff
    var_refs = [x_tm1.ref() for x_tm1 in var_list]
    grads = gradients.gradients(loss, var_refs,
                                gate_gradients=(gate_gradients == Optimizer.GATE_OP),
                                aggregation_method=aggregation_method,
                                colocate_gradients_with_ops=colocate_gradients_with_ops)
    if gate_gradients == Optimizer.GATE_GRAPH:
      grads = control_flow_ops.tuple(grads)
    grads_and_vars = list(zip(grads, var_list))
    self._assert_valid_dtypes([x_tm1 for g_t, x_tm1 in grads_and_vars if g_t is not None])
    return grads_and_vars
  
  #=============================================================
  def approximate_hessian(self, grads_and_vars, name=None):
    """
    I haven't tested this yet so I have no idea if it works, but even if it
    does it's probably super slow, and either way nothing else has been modified
    to deal with it.
    """
    
    gv = 0
    var_refs = []
    for g_t, x_tm1 in grads_and_vars:
      var_refs.append(x_tm1.ref())
      if g_t is None:
        continue
      with ops.name_scope('update_' + x_tm1.op.name), ops.device(x_tm1.device):
        if isinstance(g_t, ops.Tensor):
          gv += math_ops.reduce_sum(g_t * random_ops.random_normal(g_t.get_shape()))
        else:
          idxs, idxs_ = array_ops.unique(g_t.indices)
          g_t_ = math_ops.unsorted_segment_sum(g_t.values, idxs_, array_ops.size(idxs))
          gv += math_ops.reduce_sum(g_t_ * random_ops.random_normal(g_t_.get_shape()))
    hesses = gradients.gradients(gv, var_refs,
                                 gate_gradients=(gate_gradients == Optimizer.GATE_OP),
                                 aggregation_method=aggregation_method,
                                 colocate_gradients_with_ops=colocate_gradients_with_ops)
    return zip([g_t for g_t, _ in grads_and_vars], [x_tm1 for _, x_tm1 in grads_and_vars], hesses)
  
  #=============================================================
  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """"""
    
    # Error checking
    grads_and_vars = tuple(grads_and_vars)
    for g_t, x_tm1 in grads_and_vars:
      if not isinstance(g_t, (ops.Tensor, ops.IndexedSlices, type(None))):
        raise TypeError(
            "Gradient must be a Tensor, IndexedSlices, or None: %s" % g_t)
      if not isinstance(x_tm1, variables.Variable):
        raise TypeError(
            "Variable must be a tf.Variable: %s" % x_tm1)
      if g_t is not None:
        self._assert_valid_dtypes([g_t, x_tm1])
    var_list = [x_tm1 for g_t, x_tm1 in grads_and_vars if g_t is not None]
    if not var_list:
      raise ValueError("No gradients provided for any variable: %s" %
                       (grads_and_vars,))
    
    # The actual stuff
    with ops.control_dependencies(None):
      self._create_slots(grads_and_vars)
    update_ops = []
    with ops.op_scope([], name, self._name) as name:
      prepare = self._prepare(grads_and_vars)
      for g_t, x_tm1 in grads_and_vars:
        if g_t is None:
          continue
        with ops.name_scope("update_" + x_tm1.op.name), ops.device(x_tm1.device):
          if isinstance(g_t, ops.Tensor):
            update_ops.append(self._apply_dense(g_t, x_tm1, prepare))
          else:
            update_ops.append(self._apply_sparse(g_t, x_tm1, prepare))
      if global_step is None:
        return self._finish(update_ops, name)
      else:
        with ops.control_dependencies([self._finish(update_ops, "update")]):
          with ops.device(global_step.device):
            return state_ops.assign_add(global_step, 1, name=name).op
  
  #=============================================================
  def get_slot(self, x_tm1, name):
    """"""
    
    named_slots = self._slots.get(name, None)
    if not named_slots:
      return None
    return named_slots.get(x_tm1, None)

  #=============================================================
  def get_slot_names(self):
    """"""
    
    return sorted(self._slots.keys())

  #=============================================================
  def _assert_valid_dtypes(self, tensors):
    """"""
    valid_dtypes = self._valid_dtypes()
    for t in tensors:
      dtype = t.dtype.base_dtype
      if dtype not in valid_dtypes:
        raise ValueError(
            "Invalid type %r for %s, expected: %s." % (
                dtype, t.name, [v for v in valid_dtypes]))

  #=============================================================
  def _valid_dtypes(self):
    """"""
    
    return set([dtypes.float32])

  #=============================================================
  def _create_slots(self, grads_and_vars):
    """"""
    
    pass

  #=============================================================
  def _prepare(self, grads_and_vars):
    """"""
    pass

  #=============================================================
  def _apply_dense(self, g_t, x_tm1, prepare):
    """"""
    
    raise NotImplementedError()

  #=============================================================
  def _apply_sparse(self, g_t, x_tm1, prepare):
    """"""
    
    raise NotImplementedError()

  #=============================================================
  def _dense_moving_average(self, x_tm1, b_t, name, beta=.9):
    """
    Creates a moving average for a dense variable.
    
    Inputs:
      x_tm1: the associated parameter (e.g. a weight matrix)
      b_t: the value to accumulate (e.g. the gradient)
      name: a string to use to retrieve it later (e.g. 'm')
      beta: the decay factor (defaults to .9)
    Outputs:
      a_t: the average after moving
      t: the internal timestep (used to correct initialization bias)
    """
    
    a_tm1 = self.get_slot(x_tm1, '%s' % name)
    tm1 = self.get_slot(x_tm1, '%s/tm1' % name)
    t = state_ops.assign_add(tm1, 1, use_locking = self._use_locking)
    if beta < 1:
      beta_t = ops.convert_to_tensor(beta, name='%s/decay' % name)
      beta_t = beta_t * (1-beta**tm1) / (1-beta**t)
    else:
      beta_t = tm1 / t
    a_t = state_ops.assign(a_tm1, beta_t*a_tm1, use_locking=self._use_locking)
    a_t = state_ops.assign_add(a_t, (1-beta_t)*b_t, use_locking=self._use_locking)
    return a_t, t
    
  #=============================================================
  def _sparse_moving_average(self, x_tm1, idxs, b_t_, name, beta=.9):
    """
    Creates a moving average for a sparse variable.
    Inputs:
      x_tm1: the associated parameter (e.g. a weight matrix)
      idxs: the tensor representing the indices used
      b_t_: the value to accumulate (e.g. slices of the gradient)
      name: a string to use to retrieve it later (e.g. 'm')
      beta: the decay factor (defaults to .9)
    Outputs:
      a_t: the average after moving (same shape as x_tm1, not b_t_)
      t: the internal timestep (used to correct initialization bias)
    """
    
    a_tm1 = self._zeros_slot(x_tm1, '%s' % name, self._name)
    a_tm1_ = array_ops.gather(a_tm1, idxs)
    tm1 = self._zeros_idx_slot(x_tm1, '%s/tm1' % name, self._name)
    tm1_ = array_ops.gather(tm1, idxs)
    t = state_ops.scatter_add(tm1, idxs, tm1_*0+1, use_locking=self._use_locking)
    t_ = array_ops.gather(t, idxs)
    if beta < 1:
      beta_t = ops.convert_to_tensor(beta, name='%s/decay' % name)
      beta_t_ = beta_t * (1-beta_t**tm1_) / (1-beta_t**t_)
    else:
      beta_t_ = tm1_/t_
    a_t = state_ops.scatter_update(a_tm1, idxs, beta_t_*a_tm1_, use_locking=self._use_locking)
    a_t = state_ops.scatter_add(a_t, idxs, (1-beta_t)*b_t_, use_locking=self._use_locking)
    return a_t, t
    
  #=============================================================
  def _finish(self, update_ops, steps_and_params, name_scope):
    """"""
    
    return control_flow_ops.group(*update_ops, name=name_scope)

  #=============================================================
  def _slot_dict(self, slot_name):
    """"""
    
    named_slots = self._slots.get(slot_name, None)
    if named_slots is None:
      named_slots = {}
      self._slots[slot_name] = named_slots
    return named_slots

  #=============================================================
  def _get_or_make_slot(self, x_tm1, val, slot_name, op_name):
    """"""
    
    named_slots = self._slot_dict(slot_name)
    if x_tm1 not in named_slots:
      named_slots[x_tm1] = Optimizer.create_slot(x_tm1, val, op_name+'/'+slot_name)
    return named_slots[x_tm1]

  #=============================================================
  def _zeros_slot(self, x_tm1, slot_name, op_name):
    """"""
    
    named_slots = self._slot_dict(slot_name)
    if x_tm1 not in named_slots:
      val = array_ops.zeros_like(x_tm1.initialized_value())
      named_slots[x_tm1] = Optimizer.create_slot(x_tm1, val, op_name+'/'+slot_name)
    return named_slots[x_tm1]

  #=============================================================
  def _ones_slot(self, x_tm1, slot_name, op_name):
    """"""
    
    named_slots = self._slot_dict(slot_name)
    if x_tm1 not in named_slots:
      val = array_ops.ones_like(x_tm1.initialized_value())
      named_slots[x_tm1] = Optimizer.create_slot(x_tm1, val, op_name+'/'+slot_name)
    return named_slots[x_tm1]

  #=============================================================
  def _zeros_idx_slot(self, x_tm1, slot_name, op_name):
    """"""
    
    named_slots = self._slot_dict(slot_name)
    if x_tm1 not in named_slots:
      original_shape = x_tm1.initialized_value().get_shape().as_list()
      shape = [1] * len(original_shape)
      shape[0] = original_shape[0]
      val = array_ops.zeros(shape, dtype=x_tm1.dtype)
      named_slots[x_tm1] = Optimizer.create_slot(x_tm1, val, op_name+'/'+slot_name)
    return named_slots[x_tm1]

  #=============================================================
  def _ones_idx_slot(self, x_tm1, slot_name, op_name):
    """"""
    
    named_slots = self._slot_dict(slot_name)
    if x_tm1 not in named_slots:
      original_shape = x_tm1.initialized_value().get_shape().as_list()
      shape = [1] * len(original_shape)
      shape[0] = original_shape[0]
      val = array_ops.ones(shape, dtype=x_tm1.dtype)
      named_slots[x_tm1] = Optimizer.create_slot(x_tm1, val, op_name+'/'+slot_name)
    return named_slots[x_tm1]

  #=============================================================
  def _zero_slot(self, x_tm1, slot_name, op_name):
    """"""
    
    named_slots = self._slot_dict(slot_name)
    if x_tm1 not in named_slots:
      val = array_ops.zeros([], dtype=x_tm1.dtype)
      named_slots[x_tm1] = Optimizer.create_slot(x_tm1, val, op_name+'/'+slot_name)
    return named_slots[x_tm1]

  #=============================================================
  def _one_slot(self, x_tm1, slot_name, op_name):
    """"""
    
    named_slots = self._slot_dict(slot_name)
    if x_tm1 not in named_slots:
      val = array_ops.ones([], dtype=x_tm1.dtype)
      named_slots[x_tm1] = Optimizer.create_slot(x_tm1, val, op_name+'/'+slot_name)
    return named_slots[x_tm1]
  
  #===============================================================
  @staticmethod
  def _create_slot_var(primary, val, scope):
    """"""
    
    slot = variables.Variable(val, name=scope, trainable=False)
    # pylint: disable=protected-access
    if isinstance(primary, variables.Variable) and primary._save_slice_info:
      # Primary is a partitioned variable, so we need to also indicate that
      # the slot is a partitioned variable.  Slots have the same partitioning
      # as their primaries.
      real_slot_name = scope[len(primary.op.name + "/"):-1]
      slice_info = primary._save_slice_info
      slot._set_save_slice_info(x_tm1iables.Variable.SaveSliceInfo(
          slice_info.full_name + "/" + real_slot_name,
          slice_info.full_shape[:],
          slice_info.var_offset[:],
          slice_info.var_shape[:]))
    # pylint: enable=protected-access
    return slot
  
  #===============================================================
  @staticmethod
  def create_slot(primary, val, name, colocate_with_primary=True):
    """"""
    
    # Scope the slot name in the namespace of the primary variable.
    with ops.name_scope(primary.op.name + "/" + name) as scope:
      if colocate_with_primary:
        with ops.device(primary.device):
          return Optimizer._create_slot_var(primary, val, scope)
      else:
        return Optimizer._create_slot_var(primary, val, scope)

#***************************************************************
class BaseOptimizer(Optimizer):
  """
  The base optimizer that everything else here builds off of
  
  This class supports update clipping, update noising, and temporal averaging
  If you set the learning rate to None, it uses the Oren-Luerenburg scalar
  Hessian approximation as the learning rate (only seems to work for SGD, not 
  more complicated algorithms like Adam)
  """
  
  #=============================================================
  def __init__(self, lr=1., eps=1e-16, chi=0., clip=0., noise=None, save_step=False, save_grad=False, use_locking=False, name='Base'):
    """
    Inputs:
      lr: the global learning rate (default is 1; set to None for nifty second-
          order stuff
      eps: the stability constant sometimes needed (default is 1e-16)
      chi: the decay constant for temporal averaging (default is 0 for no
           averaging)
      clip: the maximum global norm for the updates (default is 0 for no
            clipping)
      noise: how much noise to add to the updates (default is None for no
             noise)
      save_step: whether to save the steps to a slot (default is False)
      save_grad: whether to save the grads to a slot (default is False) 
      use_locking: whether to use locking (default is False)
      name: name for the operator (default is 'Base')
    """
    
    super(BaseOptimizer, self).__init__(use_locking, name)
    self._lr = lr
    self._save_step = save_step
    self._save_grad = save_grad
    if lr is None:
      self._save_step = True
      self._save_grad = True
    self._eps = float(eps)
    self._chi = float(chi)
    self._clip = float(clip)
    self._noise = noise
  
  #=============================================================
  @property
  def learning_rate(self):
    """"""
    
    return self._lr
  
  #=============================================================
  @property
  def epsilon(self):
    """"""
    
    return self._eps
  
  #=============================================================
  @property
  def chi(self):
    """"""
    
    return self._chi
  
  #=============================================================
  @property
  def clip(self):
    """"""
    
    return self._clip
  
  #=============================================================
  @property
  def noise(self):
    """"""
    
    return self._noise
  
  #=============================================================
  def _create_slots(self, grads_and_vars):
    """"""
    
    for g_t, x_tm1 in grads_and_vars:
      if self._save_step:
        self._ones_slot(x_tm1, 's', self._name)
      if self._save_grad:
        self._ones_slot(x_tm1, 'g', self._name)
      if self._chi > 0:
        ops.add_to_collection(self._zeros_slot(x_tm1, 'x', self._name),
                              ops.GraphKeys.MOVING_AVERAGE_VARIABLES)
        if isinstance(g_t, ops.Tensor):
          self._zero_slot(x_tm1, 'x/tm1', self._name)
        else:
          self._zeros_idx_slot(x_tm1, 'x/tm1', self._name)
  
  #=============================================================
  def _prepare(self, grads_and_vars):
    """"""
    
    if self._lr is None:
      sTy = 0
      sTs = 0
      yTy = 0
      for g_t, x_tm1 in grads_and_vars:
        if g_t is None:
          continue
        with ops.name_scope('update_' + x_tm1.op.name), ops.device(x_tm1.device):
          if isinstance(g_t, ops.Tensor):
            g_tm1 = self.get_slot(x_tm1, 'g')
            s_tm1 = self.get_slot(x_tm1, 's')
            y_t = (g_t-g_tm1)
            sTy += math_ops.reduce_sum(s_tm1*y_t)
            sTs += math_ops.reduce_sum(s_tm1**2)
            yTy += math_ops.reduce_sum(y_t**2)
          else:
            idxs, idxs_ = array_ops.unique(g_t.indices)
            g_t_ = math_ops.unsorted_segment_sum(g_t.values, idxs_, array_ops.size(idxs))
            g_tm1 = self.get_slot(x_tm1, 'g')
            g_tm1_ = array_ops.gather(g_tm1, idxs)
            s_tm1 = self.get_slot(x_tm1, 's')
            s_tm1_ = array_ops.gather(s_tm1, idxs)
            y_t_ = (g_t_-g_tm1_)
            sTy += math_ops.reduce_sum(s_tm1_*y_t_)
            sTs += math_ops.reduce_sum(s_tm1_**2)
            yTy += math_ops.reduce_sum(y_t_**2)
      sTy = math_ops.abs(sTy)
      self._lr = sTs / (sTy + self._eps)
    
  #=============================================================
  def _apply_dense(self, g_t, x_tm1, prepare):
    """"""
    
    s_t = self._lr * g_t
    return [[s_t, x_tm1, g_t]]
  
  #=============================================================
  def _apply_sparse(self, g_t, x_tm1, prepare):
    """"""
    
    idxs, idxs_ = array_ops.unique(g_t.indices)
    g_t_ = math_ops.unsorted_segment_sum(g_t.values, idxs_, array_ops.size(idxs))
    
    s_t_ = self._lr * g_t_
    return [[s_t_, x_tm1, idxs, g_t_]]
    
  #=============================================================
  def _finish(self, update_ops, name_scope):
    """"""
    
    caches = [update_op[0] for update_op in update_ops]
    update_ops = [update_op[1:] for update_op in update_ops]
    if self._noise is not None:
      for cache in caches:
        s_t, x_tm1 = cache[:2]
        s_t += random_ops.random_normal(x_tm1.initialized_value().get_shape(), stddev=self._noise)
        cache[0] = s_t
    
    if self._clip > 0:
      S_t = [cache[0] for cache in caches]
      S_t, _ = clip_ops.clip_by_global_norm(S_t, self._clip)
      for cache, s_t in zip(caches, S_t):
        cache[0] = s_t
    
    new_update_ops = []
    for cache, update_op in zip(caches, update_ops):
      if len(cache) == 3:
        s_t, x_tm1 = cache[:2]
        with ops.name_scope('update_' + x_tm1.op.name), ops.device(x_tm1.device):
          x_t = state_ops.assign_sub(x_tm1, s_t, use_locking=self._use_locking)
          cache.append(x_t)
      else:
        s_t_, x_tm1, idxs = cache[:3]
        with ops.name_scope('update_' + x_tm1.op.name), ops.device(x_tm1.device):
          x_t = state_ops.scatter_sub(x_tm1, idxs, s_t_, use_locking=self._use_locking)
          cache.append(x_t)
      new_update_ops.append(control_flow_ops.group(*([x_t] + update_op)))
    
    with ops.control_dependencies(new_update_ops):
      more_update_ops = []
      if self._save_step:
        for cache in caches:
          if len(cache) == 4:
            s_t, x_tm1 = cache[:2]
            s_tm1 = self.get_slot(x_tm1, 's')
            with ops.name_scope('update_' + x_tm1.op.name), ops.device(x_tm1.device):
              new_step_and_grads = []
              s_t = state_ops.assign(s_tm1, -s_t, use_locking=self._use_locking)
          else:
            s_t_, x_tm1, idxs = cache[:3]
            s_tm1 = self.get_slot(x_tm1, 's')
            with ops.name_scope('update_' + x_tm1.op.name), ops.device(x_tm1.device):
              s_t = state_ops.scatter_update(s_tm1, idxs, -s_t_, use_locking=self._use_locking)
          more_update_ops.append(s_t)
      if self._save_grad:
        for cache in caches:
          if len(cache) == 4:
            x_tm1, g_t = cache[1:3]
            g_tm1 = self.get_slot(x_tm1, 'g')
            with ops.name_scope('update_' + x_tm1.op.name), ops.device(x_tm1.device):
              new_step_and_grads = []
              g_t = state_ops.assign(g_tm1, g_t, use_locking=self._use_locking)
          else:
            x_tm1, idxs, g_t_ = cache[1:4]
            g_tm1 = self.get_slot(x_tm1, 'g')
            with ops.name_scope('update_' + x_tm1.op.name), ops.device(x_tm1.device):
              g_t = state_ops.scatter_update(g_tm1, idxs, g_t_, use_locking=self._use_locking)
          more_update_ops.append(g_t)
      
      if self._chi > 0:
        for cache in caches:
          if len(cache) == 4:
            _, x_tm1, _, x_t = cache
            with ops.name_scope('update_' + x_tm1.op.name), ops.device(x_tm1.device):
              x_and_t = self._dense_moving_average(x_tm1, x_t, 'x', self._chi)
              more_update_ops.append(control_flow_ops.group(*x_and_t))
          else:
            _, x_tm1, idxs, _, x_t = cache
            with ops.name_scope('update_' + x_tm1.op.name), ops.device(x_tm1.device):
              x_t_ = array_ops.gather(x_t, idxs)
              x_and_t = self._sparse_moving_average(x_tm1, idxs, x_t_, 'x', self._chi)
              more_update_ops.append(control_flow_ops.group(*x_and_t))
    
    return control_flow_ops.group(*(new_update_ops + more_update_ops), name=name_scope)
  
  #==============================================================
  def average(self, var):
    """"""
    
    return self._slot_dict('x').get(var, var)
  
  #==============================================================
  def average_name(self, var):
    """"""
    
    return var.op.name + '/' + self._name + '/' + 'x'
  
  #==============================================================
  def variables_to_restore(self, moving_avg_variables=None):
    """"""
    
    name_map = {}
    if moving_avg_variables is None:
      moving_avg_variables = variables.trainable_variables()
      moving_avg_variables += variables.moving_average_variables()
    # Remove duplicates
    moving_avg_variables = set(moving_avg_variables)
    # Collect all the variables with moving average,
    for v in moving_avg_variables:
      name_map[self.average_name(v)] = v
    # Make sure we restore variables without moving average as well.
    for v in list(set(variables.all_variables()) - moving_avg_variables):
      if v.op.name not in name_map:
        name_map[v.op.name] = v
    return name_map

#***************************************************************
class AdamOptimizer(BaseOptimizer):
  """"""
  
  #=============================================================
  def __init__(self, lr=0.002, mu=.9, ups=.9, eps=1e-16,
               chi=0., clip=0., noise=None, use_locking=False, name='Adam'):
    """
    Implements Adam
    Inputs:
      lr: the learning rate (default is .002)
      mu: the decay constant for the first moment (originally beta1; default is
          .9)
      ups: the decay constant for the uncentered second moment (originall beta2;
           default is .9)
    """
    
    super(AdamOptimizer, self).__init__(lr=lr, eps=eps, chi=chi, clip=clip, noise=noise,
                                        use_locking=use_locking, name=name)
    self._mu = float(mu)
    self._ups = float(ups)
    
  #=============================================================
  @property
  def mu(self):
    """"""
    
    return self._mu
  
  #=============================================================
  @property
  def upsilon(self):
    """"""
    
    return self._ups
  
  #=============================================================
  def _create_slots(self, grads_and_vars):
    """"""
    
    super(AdamOptimizer, self)._create_slots(grads_and_vars)
    for g_t, x_tm1 in grads_and_vars:
      if self._mu > 0:
        self._zeros_slot(x_tm1, "m", self._name)
        if isinstance(g_t, ops.Tensor):
          self._zero_slot(x_tm1, "m/tm1", self._name)
        else:
          self._zeros_idx_slot(x_tm1, "m/tm1", self._name)
      if self._ups > 0:
        self._ones_slot(x_tm1, "v", self._name)
        if isinstance(g_t, ops.Tensor):
          self._zero_slot(x_tm1, "v/tm1", self._name)
        else:
          self._zeros_idx_slot(x_tm1, "v/tm1", self._name)
  
  #=============================================================
  def _apply_dense(self, g_t, x_tm1, prepare):
    """"""
    
    updates = []
    
    if self._mu > 0:
      m_and_t = self._dense_moving_average(x_tm1, g_t, 'm', self._mu)
      m_bar_t = m_and_t[0]
      updates.extend(m_and_t)
    else:
      m_bar_t = g_t
    
    if self._ups > 0:
      v_and_t = self._dense_moving_average(x_tm1, g_t**2, 'v', self._ups)
      eps_t = ops.convert_to_tensor(self._eps)
      v_bar_t = math_ops.sqrt(v_and_t[0]) + eps_t
      updates.extend(v_and_t)
    else:
      v_bar_t = 1.
    
    s_t = self._lr * m_bar_t / v_bar_t
    return [[s_t, x_tm1, g_t]] + updates
  
  #=============================================================
  def _apply_sparse(self, g_t, x_tm1, prepare):
    """"""
    
    idxs, idxs_ = array_ops.unique(g_t.indices)
    g_t_ = math_ops.unsorted_segment_sum(g_t.values, idxs_, array_ops.size(idxs))
    updates = []
    
    if self._mu > 0:
      m_and_t = self._sparse_moving_average(x_tm1, idxs, g_t_, 'm', self._mu)
      m_t_ = array_ops.gather(m_and_t[0], idxs)
      m_bar_t_ = m_t_
      updates.extend(m_and_t)
    else:
      m_bar_t_ = g_t_
    
    if self._ups > 0:
      v_and_t = self._sparse_moving_average(x_tm1, idxs, g_t_**2, 'v', self._ups)
      v_t_ = array_ops.gather(v_and_t[0], idxs)
      eps_t = ops.convert_to_tensor(self._eps)
      v_bar_t_ = math_ops.sqrt(v_t_) + eps_t
      updates.extend(v_and_t)
    else:
      v_bar_t_ = 1.
    
    s_t_ = self._lr * m_bar_t_ / v_bar_t_
    return [[s_t_, x_tm1, idxs, g_t]] + updates
  
  #==============================================================
  def clear_slots(self, var_list=None):
    """"""
    
    updates = []
    if var_list is None:
      var_list = variables.trainable_variables()
    for var in var_list:
      if self._mu > 0:
        m = self.get_slot(var, 'm')
        updates.append(state_ops.assign(m, m*0, use_locking=self._use_locking))
        tm1_m = self.get_slot(var, 'm')
        updates.append(state_ops.assign(tm1_m, tm1_m*0, use_locking=self._use_locking))
      if self._ups > 0:
        v = self.get_slot(var, 'v')
        updates.append(state_ops.assign(v, v*0, use_locking=self._use_locking))
        tm1_v = self.get_slot(var, 'v/tm1')
        updates.append(state_ops.assign(tm1_v, tm1_v*0, use_locking=self._use_locking))
    return control_flow_ops.group(*updates)
  
#***************************************************************
class NadamOptimizer(AdamOptimizer):
  """
  Implements Nesterov Momentum in Adam:
    mu_t <- mu * (1-mu^(t-1)) / (1-mu^t)
    mu_tp1 <- mu * (1-mu^t) / (1-mu^(t+1)) 
    ups_t <- ups * (1-ups^(t-1)) / (1-ups^t)
    m_t <- mu_t*m_tm1 + (1-mu_t)*g_t
    mbar_t <- mu_tp1*m_t + (1-mu_t)*g_t (Nesterov)
    v_t <- ups_t*v_tm1 + (1-ups_t)*g_t**2
    vbar_t <- sqrt(v_t) + eps
    s_t <- lr * mbar_t / vbar_t
    x_t <- x_t - s_t
  """
  
  #=============================================================
  def __init__(self, lr=0.002, mu=.9, ups=.9, eps=1e-16, chi=0., clip=0., noise=None, use_locking=False, name='Nadam'):
    """"""
    
    super(NadamOptimizer, self).__init__(lr=lr, mu=mu, ups=ups, eps=eps,
                                         chi=chi, clip=clip, noise=noise,
                                         use_locking=use_locking, name=name)
  
  #=============================================================
  def _apply_dense(self, g_t, x_tm1, prepare):
    """"""
    
    updates = []
    
    if self._mu > 0:
      m_and_t = self._dense_moving_average(x_tm1, g_t, 'm', self._mu)
      m_t, t = m_and_t
      mu = ops.convert_to_tensor(self._mu)
      mu_t = mu * (1-mu**(t-1)) / (1-mu**t)
      mu_tp1 = mu * (1-mu**t) / (1-mu**(t+1))
      m_bar_t = mu_tp1*m_t + (1-mu_t)*g_t
      updates.extend(m_and_t)
    else:
      m_bar_t = g_t
    
    if self._ups > 0:
      v_and_t = self._dense_moving_average(x_tm1, g_t**2, 'v', self._ups)
      eps_t = ops.convert_to_tensor(self._eps)
      v_bar_t = math_ops.sqrt(v_and_t[0]) + eps_t
      updates.extend(v_and_t)
    else:
      v_bar_t = 1.
    
    lr_t = ops.convert_to_tensor(self._lr)
    s_t = lr_t * m_bar_t / v_bar_t
    return [[s_t, x_tm1, g_t]] + updates
  
  #=============================================================
  def _apply_sparse(self, g_t, x_tm1, prepare):
    """"""
    
    idxs, idxs_ = array_ops.unique(g_t.indices)
    g_t_ = math_ops.unsorted_segment_sum(g_t.values, idxs_, array_ops.size(idxs))
    updates = []
    
    if self._mu > 0:
      m_and_t = self._sparse_moving_average(x_tm1, idxs, g_t_, 'm', self._mu)
      m_t, t = m_and_t
      m_t_ = array_ops.gather(m_t, idxs)
      t_ = array_ops.gather(t, idxs)
      mu = ops.convert_to_tensor(self._mu)
      mu_t_ = mu * (1-mu**(t_-1)) / (1-mu**t_)
      mu_tp1_ = mu * (1-mu**t_) / (1-mu**(t_+1))
      m_bar_t_ = mu_tp1_*m_t_ + (1-mu_t_)*g_t_
      updates.extend(m_and_t)
    else:
      m_bar_t_ = g_t_
    
    if self._ups > 0:
      v_and_t = self._sparse_moving_average(x_tm1, idxs, g_t_**2, 'v', self._ups)
      v_t_ = array_ops.gather(v_and_t[0], idxs)
      eps_t = ops.convert_to_tensor(self._eps)
      v_bar_t_ = math_ops.sqrt(v_t_) + eps_t
      updates.extend(v_and_t)
    else:
      v_bar_t_ = 1.
    
    lr_t = ops.convert_to_tensor(self._lr)
    s_t_ = lr_t * m_bar_t_ / v_bar_t_
    return [[s_t_, x_tm1, idxs, g_t]] + updates
  
#***************************************************************
class RadamOptimizer(AdamOptimizer):
  """
  Implements Reweighted Adam
      mu_t <- mu * (1-mu^(t-1)) / (1-mu^t)
      ups_t <- ups * (1-ups^(t-1)) / (1-ups^t)
      m_t <- mu_t*m_tm1 + (1-mu_t)*g_t
      mbar_t <- (1-gamma)*m_t + gamma*g_t (Reweighting)
      v_t <- ups_t*v_tm1 + (1-ups_t)*g_t**2
      vbar_t <- sqrt(v_t) + eps
      s_t <- lr * mbar_t / vbar_t
      x_t <- x_t - s_t
  """
  
  #=============================================================
  def __init__(self, lr=0.002, mu=.9, gamma=.05, ups=.9, eps=1e-7, chi=0., clip=0., noise=None, use_locking=False, name='Radam'):
    """"""
    
    super(RadamOptimizer, self).__init__(lr=lr, mu=mu, ups=ups, eps=eps,
                                         chi=chi, clip=clip, noise=noise,
                                         use_locking=use_locking, name=name)
    self._gamma = float(gamma)
  
  #=============================================================
  @property
  def gamma(self):
    """"""
    
    return self._gamma
  
  #=============================================================
  def _apply_dense(self, g_t, x_tm1, prepare):
    """"""
    
    updates = []
    
    if self._mu > 0:
      m_and_t = self._dense_moving_average(x_tm1, g_t, 'm', self._mu)
      gamma_t = ops.convert_to_tensor(self._gamma)
      m_bar_t = (1-gamma_t)*m_and_t[0] + gamma_t*g_t
      updates.extend(m_and_t)
    else:
      m_bar_t = g_t
    
    if self._ups > 0:
      v_and_t = self._dense_moving_average(x_tm1, g_t**2, 'v', self._ups)
      eps_t = ops.convert_to_tensor(self._eps)
      v_bar_t = math_ops.sqrt(v_and_t[0]) + eps_t
      updates.extend(v_and_t)
    else:
      v_bar_t = 1.
    
    lr_t = ops.convert_to_tensor(self._lr)
    s_t = lr_t * m_bar_t / v_bar_t
    return [[s_t, x_tm1, g_t]] + updates
  
  #=============================================================
  def _apply_sparse(self, g_t, x_tm1, prepare):
    """"""
    
    idxs, idxs_ = array_ops.unique(g_t.indices)
    g_t_ = math_ops.unsorted_segment_sum(g_t.values, idxs_, array_ops.size(idxs))
    updates = []
    
    if self._mu > 0:
      m_and_t = self._sparse_moving_average(x_tm1, idxs, g_t_, 'm', self._mu)
      m_t_ = array_ops.gather(m_and_t[0], idxs)
      gamma_t = ops.convert_to_tensor(self._gamma)
      m_bar_t_ = (1-gamma_t)*m_t_ + gamma_t*g_t_
      updates.extend(m_and_t)
    else:
      m_bar_t_ = g_t_
    
    if self._ups > 0:
      v_and_t = self._sparse_moving_average(x_tm1, idxs, g_t_**2, 'v', self._ups)
      v_t_ = array_ops.gather(v_and_t[0], idxs)
      eps_t = ops.convert_to_tensor(self._eps)
      v_bar_t_ = math_ops.sqrt(v_t_) + eps_t
      updates.extend(v_and_t)
    else:
      v_bar_t_ = 1.
    
    lr_t = ops.convert_to_tensor(self._lr)
    s_t_ = lr_t * m_bar_t_ / v_bar_t_
    return [[s_t_, x_tm1, idxs, g_t]] + updates

#***************************************************************
if __name__ == '__main__':
  """
  Little sandbox for testing things out--trains a symmetric orthonormal matrix
  """
  
  import sys
  import numpy as np
  import tensorflow as tf
  # TODO create a scalar_moving_average function
  # TODO compute cosine similarity
  # TODO try using moving averages for the OL update--may help Adam
  
  A = np.random.randn(100,100).astype('float32')
  I = tf.Variable(np.eye(100).astype('float32'), trainable=False)
  Q = tf.Variable(A)
  loss_op = tf.reduce_mean((tf.matmul(Q, Q, transpose_a=True) - I)**2 / 2 + (Q - tf.transpose(Q))**2 / 2)
  Q2 = Q**2
  true_hess_op = tf.sqrt(tf.reduce_sum((2*(Q2 + tf.reduce_sum(Q2, 0, keep_dims=True) +
                                           tf.reduce_sum(Q2, 1, keep_dims=True) - I)/100**2)**2))
  gs = tf.Variable(0., trainable=False)
  sig = .005/(1+tf.exp((2*gs-5000)/1000))
  for optimizer in (RadamOptimizer,):
    for lr in (.002,):
      opt = optimizer(lr=lr, noise=sig, clip=5, chi=.99)
      train_op = opt.minimize(loss_op, global_step=gs)
      #hess_op = tf.sqrt(tf.reduce_sum(optimizer.get_slot(Q, 'n')**2))
      #step_op = optimizer.get_slot(Q, 's')
      with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in xrange(5000):
          loss, _ = sess.run([loss_op, train_op])
          print('Loss: %.1e' % (loss,), end='\r')
          sys.stdout.flush()
          
        print(opt.average(Q))
      print()  
  