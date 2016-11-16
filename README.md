# Parser
This repository contains the code used to train the parsers described in the paper [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734). Here we describe how the source code is structured and how to train/validate/test models.

## Where are the files you care about?
* `lib/linalg.py`: This file contains general-purpose functions that don't require any knowledge of hyperparameters. For example, the `linear` and `bilinear` functions, which simply return the result of applying an affine or biaffine transformation to the input.
* `configurable.py`: This file contains the Configurable class, which wraps a `SafeConfigParser` that stores model hyperparameter options (such as dropout keep probability and recurrent size). Most or all classes in this repository inherit from it.
* `lib/models/nn.py`: This file contains the `NN` class, which inherits from `Configurable`. It contains functions such as `MLP` and `RNN` that are general-purpose but require knowledge of model hyperparameters.
* `lib/models/rnn.py`: This file contains functions for building tensorflow recurrent neural networks. It is largely copied and pasted tensorflow source code with a few modifications to include a dynamic *bidirectional* recurrent neural network (rather than just a dynamic *unidirectional* one, which was all that was available when this project was started) and same-mask recurrent dropout.
* `lib/models/parsers`: This directory contains different parser architectures. All parsers inherit from `BaseParser`, which in turn inherits from `NN`. The README in that directory details the differences between architectures.
* `lib/rnn_cells`: This directory contains a number of different recurrent cells (including LSTMs and GRUs). All recurrent cells inherit from `BaseCell` which inherits from `Configurable` (but not `NN`). The README in that directory details the different cell types.
* `lib/optimizers`: This directory contains the optimizer used to optimize the network. All optimizers inherit from `BaseOptimizer` which inherits from `Configurable` (again not `NN`). See the README in that directory for further explanation.
* `vocab.py`: This file contains the `Vocab` class, which manages a vocabulary of discrete strings (tokens, POS tags, dependency labels).
* `bucket.py`: This file contains the `Bucket` class, which manages all sequences of data up to a certain length, and pads everything shorter than that length with special `<PAD>` tokens.
* `metabucket.py`: This file contains the `Metabucket` class, which manages a group of multiple buckets, efficiently determining which bucket a new sentence goes in.
* `dataset.py`: This file contains the `Dataset` class, which manages an entire dataset (e.g. the training set or the test set), reading in a conll file and grabbing minibatches.
* `network.py`: This file contains the `Network` class, which manages the training and testing of a neural network. It contains three `Dataset` objects--one for the training set, one for the validation set, and one for the test set--three `Vocab` objects--one for the words, one for the POS tags, and one for the dependency labels--one `NN` object--a parser architecture or other user-defined architecutre--and a `BaseOptimizer` object (stored in the `self._ops` dictionary). This is also the file you call to run the network.

## How do you run the model?
### Data
After downloading the repository, you will need a few more things:

* **pretrained word embeddings**: We used 100-dimensional [GloVe](http://nlp.stanford.edu/projects/glove/) embeddings
* **data**: We used the Penn TreeBank dataset automatically converted to Stanford Dependencies, but since this dataset is proprietary, you can instead use the freely available English Web Treebank in [Universal Dependencies](http://universaldependencies.org) format.

We will assume that the dataset has been downloaded and exists in the directory `data/EWT` and the word embeddings exist in `data/glove`. 

### Config files
All configuration options can be specified on the command line, but it's much easier to instead store them in a configuration file. This includes the location of the data files. We recommend creating a new configuration file `config/myconfig.cfg` in the config directory:
```
[OS]
embed_dir = data/glove
embed_file = %(embed_dir)s/en.100d.txt
data_dir = data/EWT
train_file = %(data_dir)s/train.conllu
valid_file = %(data_dir)s/dev.conllu
test_file = %(data_dir)s/test.conllu
```
This is also where other options can be specified; for example, to use the same configuration options used in the paper, one would also add
```
[Layers]
n_recur = 4

[Dropout]
mlp_keep_prob = .67
ff_keep_prob = .67

[Regularization]
l2_reg = 0

[Radam]
chi = 0

[Learning rate]
learning_rate = 2e-3
decay_steps = 2500
```

### Training
The model can be trained with
```bash
python network.py --config_file config/myconfig.cfg --save_dir saves/mymodel
```
The `saves` directory must already exist. It will attempt to create a `mymodel` directory in `saves`; if `saves/mymodel` already exists, it will warn the user and ask if they want to continue. This is to prevent accidentally overwriting trained models. The model then reads in the training files and prints out the shapes of each bucket. By default, all matrices are initialized orthonormally; in order to generate orthonormal matrices, it starts with a random normal matrix and optimizes it to be orthonormal (on the CPU, using numpy). The final loss of this is printed, so that if the optimizer diverges (which is very rare but does occasionally happen) the researcher can restart.

Durint training, the model prints out training and validation loss, labeled attachment accuracy, and runtime (in sentences/second). During validation, the model also generates a `sanitycheck.txt` file in the save directory that prints out the model's predictions on sentences in the validation file. It also saves `history.pkl` to the save directory, which records the model's training and validation loss and accuracy. At this stage the model makes no attempt to ensure that the trees are well-formed and it makes no attempt to ignore punctuation.

The model will periodically save its tensorflow state so that it can be reloaded in the event of a crash or accidental termination. If the researcher wishes to terminate the model prematurely, they can do so with `<ctrl-C>`; in this event, they will be prompted to save the model with `<enter>` or discard it with another `<ctrl-C>`.

### Testing
The model can be validated with
```bash
python network.py --save_dir saves/mymodel --validate
python network.py --save_dir saves/mymodel --test
```
This creates a parsed copy of the validation and test files in the save directory. The model also reports unlabeled and labeled attachment accuracy in `saves/mymodel/scores.txt`, but these calculate punctuation differently from what is standard. One should instead use the perl script in `bin` to compute accuracies:
```bash
perl bin/eval.pl -q -b -g data/EWT/dev.conllu \
                       -s saves/mymodel/dev.conllu \
                       -o saves/mymodel/dev.scores.txt
```
Statistical significance between two models can similarly be computed using a perl script:
```bash
perl bin/compare.pl saves/mymodel/dev.scores.txt saves/defaults/dev.scores.txt
```

The current build is designed for research purposes, so explicit functionality for parsing texts is not currently supported.

## What does the model put in the save directory?
* `config.cfg`: A configuration file containing the model hyperparameters. Since hyperparameters can come from a variety of different sources (including multiple config files and command line arguments), this is necessary for restoring it later and remembering what hyperparameters were used.
* `HEAD`: The github repository head--keeps track of the current github build, so that if the current github version is incompatible with the trained model, the researcher knows which commit they need to restore to run it.
* `history.pkl`: A python pickle file containing a dictionary of training and validation history.
* `<model name>`: tensorflow checkpoint file indicating which model to restore.
* `<model name>-trained-<number>(.txt)`: tensorflow model after training for `<number>` iterations.
* `words.txt`/`tags.txt`/`rels.txt`: Vocabulary files containing all words/tags/labels in the training set and their frequency, sorted by frequency.
* `sanitycheck.txt`: The model's validation output. The sentences are grouped by bucket, not in the original order they were observed in the file, and the parses are chosen greedily rather than using any MST parsing algorithm to ensure well-formedness. Predicted heads/relations are put in second-to-last two columns, and gold heads/relations are put in the last two columns.
* `scores.txt`: The model's self-reported unlabeled/labeled accuracy scores. As previously stated, don't trust these numbers too much--use the perl script.
* `dev.conllu`/`test.conllu`: The parsed validation and test datasets.
