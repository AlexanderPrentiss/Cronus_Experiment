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

## References

- [A Little Is Enough: Circumventing Defenses for Distributed Learning](https://arxiv.org/abs/1902.06156)
- [Cronus: Robust and Heterogeneous Collaborative Learning with Black-Box Knowledge Transfer](https://arxiv.org/abs/1912.11279)
- [On the Strategyproofness of the Geometric Median](https://proceedings.mlr.press/v206/el-mhamdi23a.html)
- [Online Label Aggregation: A Variational Bayesian Approach](https://arxiv.org/abs/1807.07291)
- [Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates](https://arxiv.org/abs/1803.01498)

