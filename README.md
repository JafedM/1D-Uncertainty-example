# 1D-Uncertainty-example
An example of uncertainty in 1D with deep-ensembles, masksembles and dropout

This repository is an example of how to apply the Bayesian Deep learning to estimate uncertainty in a 1D problem using deep-ensembles, masksembles and dropout.

## Examples

![epistemic](https://github.com/JafedM/1D-Uncertainty-example/blob/main/Images/Mask_epistemic.png)

![aleatoric](https://github.com/JafedM/1D-Uncertainty-example/blob/main/Images/Ensemble_aleatoric.png)

## References
```
@inproceedings{Ensambles,
 author = {Lakshminarayanan, Balaji and Pritzel, Alexander and Blundell, Charles},
 title = {Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles},
 url = {https://proceedings.neurips.cc/paper_files/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf},
 year = {2017}
}

@misc{kendall2017uncertainties,
      title={What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?}, 
      author={Alex Kendall and Yarin Gal},
      year={2017},
      eprint={1703.04977},
      archivePrefix={arXiv},
}

@misc{masksembles,
      title={Masksembles for Uncertainty Estimation}, 
      author={Nikita Durasov and Timur Bagautdinov and Pierre Baque and Pascal Fua},
      year={2021},
      eprint={2012.08334},
      archivePrefix={arXiv},
}