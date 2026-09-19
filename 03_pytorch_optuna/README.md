# PyTorch with Optuna

This section extends the PyTorch digit classifier with automated hyperparameter optimization using Optuna.

## Notebooks

* `digits_pytorch_optuna_cpu.ipynb`
* `digits_pytorch_optuna_gpu.ipynb`

## Hyperparameters

The Optuna study searches over:

* learning rate
* batch size
* first hidden-layer size
* second hidden-layer size

## Workflow

```text
training data
↓
Optuna trial
↓
select hyperparameters
↓
train PyTorch model
↓
validation accuracy
↓
repeat across trials
↓
best hyperparameters
↓
retrain final model
↓
evaluate on untouched test set
```

A separate validation set is used during tuning so that the final test set remains independent of hyperparameter selection.

The notebooks include:

* Optuna study creation
* configurable neural-network architecture
* multiple optimization trials
* validation-based model selection
* best-parameter reporting
* trial-performance visualization
* final model retraining
* test-set evaluation
* CPU and GPU implementations

In the GPU version, PyTorch performs model training on CUDA while Optuna manages the hyperparameter search.
