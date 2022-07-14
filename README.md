# Adversarial Attacks and Defense for Non-Parametric TSTs

This repository provides codes for ICML 2022 paper: **Adversarial Attack and Defense for Non-Parametric Two-Sample Tests** (https://proceedings.mlr.press/v162/xu22m.html) 
 *Xilie Xu\* (NUS), Jingfeng Zhang\* (RIKEN-AIP), Feng Liu (The University of Melbourne), Masashi Sugiyama (RIKEN-AIP/The University of Toyko), Mohan Kankanhalli (NUS)*

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

The first step is to obtain powerful non-parametric TSTs.

```
from TST_tools import MMD_D, MMD_G, C2ST_L, C2ST_S, ME, SCF

# Initialize TSTs
MMD_D_test, MMD_G_test, C2ST_L_test, C2ST_S_test, ME_test, SCF_test
                    = MMD_D(), MMD_G(), C2ST_L(), C2ST_S(), ME(), SCF()

# Train TSTs
P_train, Q_train = sample_from_P_Q()
MMD_D_test.train(P_train, Q_train)
MMD_G_test.train(P_train, Q_train)
C2ST_L_test.train(P_train, Q_train)
C2ST_S_test.train(P_train, Q_train)
ME_test.train(P_train, Q_train)
SCF_test.train(P_train, Q_train)
```

Then we can generate adversarial data against these non-parametric TSTs.

```
from TST_attack import two_sample_test_attack

# Attack TSTs
TST_adversary = two_sample_test_attack(num_steps=num_steps, epsilon=epsilon,dynamic_eta=True, 
                                        max_scale=max_scale, min_scale=min_scale, 
                                        test_rags=[(MMD_D_test, MMD_D_weight),
                                                  (MMD_G_test, MMD_G_weight),
                                                  (C2ST_S_test, C2ST_S_weight),
                                                  (C2ST_L_test, C2ST_L_weight),
                                                  (ME_test, ME_weight),
                                                  (SCF_test, SCF_weight)])

# Generate adversarial pairs
P_test, Q_test = sample_from_P_Q()
Adv_Q_test = TST_adversary.attack(P_test, Q_test)
```

## How to defend non-parametric TSTs?
You can easily obtain our proposed robust non-parametric TSTs (MMD-RoD) with the following code.

```
from TST_tools import MMD_RoD

# Initialize robust TSTs
MMD_RoD_test = MMD_RoD()

# Train robust TSTs
P_train, Q_train = sample_from_P_Q()
MMD_RoD_test.train(P_train, Q_train)
```

## Evaluate test power of non-parametric TSTs

We provide scripts (**run.sh**) for obtaining **all** experimental results in our paper.
Here is an example that evaluates test power on Blob dataest.
```
python TST_Blob.py
```

## Reference
```
@InProceedings{xu2022adversarial,
  title = 	 {Adversarial Attack and Defense for Non-Parametric Two-Sample Tests},
  author =       {Xu, Xilie and Zhang, Jingfeng and Liu, Feng and Sugiyama, Masashi and Kankanhalli, Mohan},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {24743--24769},
  year = 	 {2022},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR}
}
```

## Contact
Please contact xuxilie@comp.nus.edu.sg and jingfeng.zhang@riken.jp if you have any question on the codes.
