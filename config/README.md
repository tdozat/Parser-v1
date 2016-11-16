# Config
The model allows for the following options:

## OS
This section of the config file contains information about where files are located and where they should be placed.

* `save_dir`: The directory where the model will save information during training and testing.
* `word_file`/`tag_file`/`rel_file`: The name of the text file that will store unique tokens/tags/labels and their counts.
* `embed_dir`: The directory where the embeddings are located.
* `embed_file`: The name of the embedding file.
* `data_dir`: The directory where the data is stored.
* `train_file`/`valid_file`/`test_file`: The names of the training/validation/testing files

## Dataset
This section details how to process the dataset.

* `cased`: Whether to keep the original case of the trainable word embeddings or convert them all to lower case
* `min_occur_count`: How many times a word must occur in the training set to get its own embedding. Any tokens that occur fewer times will be replaced with an `<UNK>` token.
* `n_bkts`: How many buckets to sort the training sentences into.
* `n_valid_bkts`: How many buckets to sort the validation/testing sentences into.
* `lines_per_buffer`: How many sentences of the files to read in and train on at one time. Setting to 0 makes the program read in all lines at once. Currently only 0 is supported.

## Layers
This section details how deep the model should be.

* `n_recur`: How many recurrent layers.
* `n_mlp`: How many MLP layers after the recurrent layers.
* `recur_cell`: The type of recurrent cell to use. Must be the name of a class imported by `lib.recur_cells`.
* `recur_bidir`: Whether or not to make the recurrent layers bidirectional.
* `forget_bias`: How much to bias the forget gate/update gate of the recurrent cell if applicable.

## Sizes
This section details how wide the model should be.

* `embed_size`: Size of the word embeddings.
* `recur_size`: Size of the recurrent layers. If using bidirectional networks, the size of the recurrent layers of the RNN going in each direction.
* `mlp_size`: Size of the MLP layers.

## Functions
This section details the nonlinear functions the model should use. They must be imported by tf.nn or be 'identity'.

* `recur_func`: The nonlinearity in the recurrent cells.
* `mlp_func`: The nonlinearity in the mlp cells.

## Regularization
This section details regularization hyperparameters. Some are experimental and may not work as intended.

* `l2_reg`: L2 regularization penalty on weight matrices.
* `recur_reg`: [Recurrent L2 penalty](https://arxiv.org/abs/1511.08400)
* `covar_reg`: [Covariance L2 penalty](https://arxiv.org/abs/1511.06068)
* `ortho_reg`: L2 penalty to encourage orthogonal weight matrices.

## Dropout
This section details dropout hyperparameters.

* `drop_gradually`: Whether to gradually increase the dropout probability or keep it constant throughout training.
* `word_keep_prob`: Probability of not dropping words.
* `tag_keep_prob`: Probability of not dropping tags.
* `rel_keep_prob`: Probability of not dropping labels. (Not used by any current models)
* `recur_keep_prob`: Probabilty of not dropping connections between recurrent steps.
* `ff_keep_prob`: Probability of not dropping connections feeding into recurrent layers.
* `mlp_keep_prob`: Probability of not dropping connections feeding into MLP layers.

## Learning rate
This section details properties of the learning rate.

* `learning_rate`: The starting learning rate.
* `decay`: How much to decay the learning rate.
* `decay_steps`: How many steps to decay the learning rate by a factor of `decay`.
* `clip`: Gradient clipping. Gradient information is hard-clipped to this value before being accumulated into Adam's accumulators, and parameter updates are soft-clipped to have at most this L2 norm.

## Radam
This section details hyperparameters of [my own variant](https://github.com/tdozat/Optimization) of the Adam learning algorithm.

* `mu`: The momentum hyperparameter. (Adam's beta1)
* `nu`: The L2 norm hyperparameter. (Adam's beta2)
* `gamma`: The momentum-gradient interpolation hyperparameter. (Setting to zero makes the algorithm equivalent to Adam)
* `chi`: The temporal averaging hyperparameter. (Setting to zero makes the algorithm equivalent to Adam)
* `epsilon`: The stability constant.

## Training
This section details training settings.

* `pretrain_iters`: If pretraining the model (not fully supported), how many iterations to pretrain it for. (rounded up to the nearest epoch)
* `train_iters`: How many iterations to train the model for. (rounded up to the nearest epoch)
* `train_batch_size`: Approximately how many tokens (not sentences!) to include in each training minibatch.
* `test_batch_size`: Approximately how many tokens to include in each testing minibatch.
* `validate_every`: How often to test the model on the validation set during training.
* `print_every`: How often to print the current model performance.
* `save_every`: How often to save the model in case of a crash.
* `per_process_gpu_memory_fraction`: How much GPU memory to reserve for training/running the model.
