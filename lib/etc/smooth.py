import numpy as np

def smooth(a, beta=.9):
  b = np.empty_like(a)
  b[0] = a[0]
  for i in xrange(1, len(a)):
    beta_i = beta * (1-beta**i) / (1-beta**(i+1))
    b[i] = beta_i * b[i-1] + (1-beta_i) * a[i]
  return b
