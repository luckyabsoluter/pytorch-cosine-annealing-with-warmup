# Cosine Annealing with Warmup for PyTorch

## News
- 2020/12/22 : update is coming soon...
- 2020/12/24 : Merry Christmas! Release new version, 2.0. Previous version is [here (branch: 1.0)](https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/tree/1.0).
- 2021/06/04 : This package can now be installed with pip.

## Installation
To install this package, use the following command:
```bash
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
```
This command installs the latest version of the package directly from the GitHub repository.

## Args
### CosineAnnealingWarmupRestarts
- optimizer (Optimizer): The optimizer to be wrapped.
- first_cycle_steps (int): Number of steps in the first cycle.
- cycle_mult (float): Multiplier for cycle steps. Default: 1.
- max_lr (float): Maximum learning rate for the first cycle. Default: 0.1.
- min_lr (float): Minimum learning rate. Default: 0.001.
- **warmup_steps (int)**: Number of steps for linear warmup. Default: 0.
- gamma (float): Decay rate for the maximum learning rate per cycle. Default: 1.
- last_epoch (int): Index of the last epoch. Default: -1.

### CosineAnnealingRateWarmupRestarts
- optimizer (Optimizer): The optimizer to be wrapped.
- first_cycle_steps (int): Number of steps in the first cycle.
- cycle_mult (float): Multiplier for cycle steps. Default: -1.
- max_lr (float): Maximum learning rate for the first cycle. Default: 0.1.
- min_lr (float): Minimum learning rate. Default: 0.001.
- **warmup_rate (float)**: Linear warmup rate for the current cycle's step size. Default: 0.
- gamma (float): Decay rate for the maximum learning rate per cycle. Default: 1.
- last_epoch (int): Index of the last epoch. Default: -1.

## Example
### CosineAnnealingWarmupRestarts
```python
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

model = ...  # Define your model
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)  # lr is min lr
scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=200,
                                          cycle_mult=1.0,
                                          max_lr=0.1,
                                          min_lr=0.001,
                                          warmup_steps=50,
                                          gamma=1.0)
for epoch in range(n_epoch):
    train()  # Training step
    valid()  # Validation step
    scheduler.step()  # Update learning rate
```

- case1 : `CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=500, cycle_mult=1.0, max_lr=0.1, min_lr=0.001, warmup_steps=100, gamma=1.0)`
![example1](./src/plot001.png "example1")
- case2 : `CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=200, cycle_mult=1.0, max_lr=0.1, min_lr=0.001, warmup_steps=50, gamma=0.5)`
![example2](./src/plot002.png "example2")

### CosineAnnealingRateWarmupRestarts
```python
from cosine_annealing_warmup import CosineAnnealingRateWarmupRestarts

model = ...  # Define your model
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)  # lr is min lr
scheduler = CosineAnnealingRateWarmupRestarts(optimizer,
                                              first_cycle_steps=100,
                                              cycle_mult=2.0,
                                              max_lr=0.1,
                                              min_lr=0.001,
                                              warmup_rate=0.1,
                                              gamma=1.0)
for epoch in range(n_epoch):
    train()  # Training step
    valid()  # Validation step
    scheduler.step()  # Update learning rate
```
- case1 : `CosineAnnealingRateWarmupRestarts(optimizer, first_cycle_steps=100, cycle_mult=2.0, max_lr=0.1, min_lr=0.001, warmup_rate=0.1, gamma=1.0)`
