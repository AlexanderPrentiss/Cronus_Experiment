# Cronus Aggregation Rule Experiment

Evaluates robust aggregation rules for **Cronus**, a black-box collaborative learning system where clients share predictions instead of model parameters.

The experiment trains 20 MNIST clients and compares RobustFilter, trimmed mean, geometric median, and hybrid aggregators under no attack, LIE attacks, and label-flip attacks.

## Result

RobustFilter + geometric median achieved the lowest average honest-client error in every recorded setting: **6.75%** with no attack, **6.98%** under LIE, and **6.97%** under label flipping.

## Run

```bash
pip install torch torchvision numpy scipy matplotlib jupyter
jupyter notebook AGR_test.ipynb
```

Set `NUM_LIE` or `NUM_LABEL_FLIP` in the data-initialization cell, then run all cells. See [`Cronus_AGR_Improvement_Paper.pdf`](Cronus_AGR_Improvement_Paper.pdf) for the full methodology and results.
