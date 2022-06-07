# Adversarial Attacks and Defense for Non-Parametric TSTs

This repository provides codes for ICML 2022 paper: **Adversarial Attacks and Defense for Non-Parametric Two-Sample Tests** (https://arxiv.org/abs/2202.03077) *Xilie Xu\* (NUS), Jingfeng Zhang\* (RIKEN-AIP), Feng Liu (The University of Melbourne), Masashi Sugiyama (RIKEN-AIP/The University of Toyko), Mohan Kankanhalli (NUS)*

## Requirements
This implementation is based on the following packages:
+ Python 3.8
+ PyTorch 1.11
+ CUDA 11.3
+ [Freqopttest](https://github.com/wittawatj/interpretable-test) (This is a package for implementing ME and SCF tests. You can install it via ```pip install git+https://github.com/wittawatj/interpretable-test```)
+ Numpy

## Datasets
Following the setting of [MMD-D](https://github.com/fengliu90/DK-for-TST), before running the code on Higgs and MNIST, you need to download higgs and fake_mnist datasets from the google drive links:
+ fake_mnist dataset: https://drive.google.com/open?id=13JpGbp7PEm4PfZ6VeqpFiy0lHfVpy5Z5

+ higgs data: https://drive.google.com/open?id=1sHIIFCoHbauk6Mkb6e8a_tp1qnvuUOCc


## How to attack non-parametric TSTs?
Import and initialize TST and adversary agaisnt TST with

```
from TST_tools import MMD_D, MMD_G, C2ST_L, C2ST_S, ME, SCF, MMD_RoD
from TST_attack import two_sample_test_attack

s1, s2 = sample_from_P_Q()
MMD_D_test = MMD_D().train(s1, s2)
MMD_


TST_adversary = two_sample_test_attack(num_steps=num_steps, epsilon=epsilon,dynamic_eta=1, max_scale=max_scale, min_scale=min_scale, test)
```

## How to defend non-parametric TSTs?


## Reference
```
@article{xu2022adversarial,
  title={Adversarial Attacks and Defense for Non-Parametric Two-Sample Tests},
  author={Xu, Xilie and Zhang, Jingfeng and Liu, Feng and Sugiyama, Masashi and Kankanhalli, Mohan},
  journal={arXiv preprint arXiv:2202.03077},
  year={2022}
}
```

## Contact
Please contact xuxilie@comp.nus.edu.sg and jingfeng.zhang@riken.jp if you have any question on the codes.
