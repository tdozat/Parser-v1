# Optimizers
This directory contains optimizers usable by the network. All moving averages are done "correctly", as described by [the Adam paper](https://arxiv.org/abs/1412.6980), and accumulators for sparce embedding matrices are only updated for embeddings that have a gradient (that is, if an embedding isn't used in the minibatch, its accumulators are not decayed).

* `SGD`: Stochastic gradient descent with no momentum or L2 norm accumulators.
* `Radam`: Reweighted Adam. Nesterov momentum simplifies to interpolating between the momentum vector and the true gradient according to the momentum constant mu; Radam incorporates this reweighting into Adam, parameterizing the reweighting by another free hyperparameter, gamma. Setting gamma to zero makes the algorithm equivalent to vanilla Adam.
