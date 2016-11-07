# Scripts

## Rescore
Sometimes there are bugs in the evaluation function that aren't caught until after the model has been fully trained. `rescore.py` rescores files in `save_dir` according to the current version of model `model`'s file evaluation function and saves the results in `save_dir/scores.txt`.

From the root directory:
```bash
python -m scripts.rescore [-h] [-f [FILENAMES [FILENAMES ...]]] [-m MODEL] save_dir
```

## Significance test
We want to know if the differences between models are significant. `sig_test.py` uses approximate randomization testing to determine if the models produced predictions of significantly different quality. In addition to specifying the predicted files to be compared, the user must also specify the types of the two models, since different models can predict the same variables.

From the root directory:
```bash
python -m scripts.sig_test [-h] [-m1 MODEL1] [-m2 MODEL2] [-n N_TRIALS] filename1 filename2
```