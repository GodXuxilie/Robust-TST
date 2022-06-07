# test power in the benign settings and under white-box attacks
nohup python TST_Blob.py --gpu=0 > log/Blob_100.log &

nohup python TST_HDGM.py --gpu=0 > log/HDGM_3000.log &

nohup python TST_HIGGS.py --gpu=0 > log/HIGGS_5000.log &

nohup python TST_MNIST.py --gpu=3 > log/MNIST_500.log &

nohup python TST_CIFAR10.py --gpu=0 > log/CIFAR10_500.log &

# robust kernel
nohup python TST_Blob.py --weight=5,1,1,20,1,1,5 --robust_kernel=1 --gpu=0 > log/Blob_100_robust.log &

nohup python TST_HDGM.py --weight=25,1,1,50,1,1,25 --robust_kernel=1 --gpu=0 > log/HDGM_3000_robust.log &

nohup python TST_HIGGS.py --weight=3,45,4,3,40,3,3 --robust_kernel=1 --gpu=0 > log/HIGGS_5000_robust.log &

nohup python TST_MNIST.py --weight=1,45,1,1,60,1,1 --robust_kernel=1 --gpu=1 > log/MNIST_500_robust.log &

nohup python TST_CIFAR10.py --weight=1,50,4,4,20,1,1 --robust_kernel=1 --gpu=2 > log/CIFAR10_500_robust.log &

# different test sample numbers
nohup python TST_MNIST.py --n=100 --gpu=1 > log/MNIST_100.log &

nohup python TST_MNIST.py --n=200 --gpu=1 > log/MNIST_200.log &

nohup python TST_MNIST.py --n=300 --gpu=1 > log/MNIST_300.log &

nohup python TST_MNIST.py --n=400 --gpu=1 > log/MNIST_400.log &

# different epsilon
nohup python TST_MNIST.py --epsilon=0.025 --step_size=0.025  --gpu=1 > log/MNIST_500_eps0025.log &

nohup python TST_MNIST.py --epsilon=0.075 --step_size=0.075  --gpu=2 > log/MNIST_500_eps0075.log &

nohup python TST_MNIST.py --epsilon=0.1 --step_size=0.1  --gpu=3 > log/MNIST_500_eps01.log &

nohup python TST_MNIST.py --epsilon=0.15 --step_size=0.15  --gpu=1 > log/MNIST_500_eps015.log &

nohup python TST_MNIST.py --epsilon=0.2 --step_size=0.2  --gpu=2 > log/MNIST_500_eps02.log &

# different dimension
nohup python TST_HDGM.py --gpu=0 --d=5 > log/HDGM_3000_d5.log &

nohup python TST_HDGM.py --gpu=1 --d=15 > log/HDGM_3000_d15.log &

nohup python TST_HDGM.py --gpu=2 --d=20 > log/HDGM_3000_d20.log &

nohup python TST_HDGM.py --gpu=3 --d=25 > log/HDGM_3000_d25.log &

# different weight strategy
nohup python TST_Blob.py --weight=1,1,1,1,1,1  --gpu=2 > log/Blob_100_1_1_1_1_1_1.log &

nohup python TST_HDGM.py --weight=1,1,1,1,1,1  --gpu=2 > log/HDGM_3000_1_1_1_1_1_1.log &

nohup python TST_HIGGS.py --weight=1,1,1,1,1,1  --gpu=1 > log/HIGGS_5000_1_1_1_1_1_1.log &

nohup python TST_MNIST.py --weight=1,1,1,1,1,1  --gpu=1 > log/MNIST_500_1_1_1_1_1_1.log &

nohup python TST_CIFAR10.py --weight=1,1,1,1,1,1  --gpu=2 > log/CIFAR10_500_1_1_1_1_1_1.log &

nohup python TST_Blob.py --weight=1,1,1,1,1,1 --adaptive_weight=1  --gpu=2 > log/Blob_100_adaweight.log &

nohup python TST_HDGM.py --weight=1,1,1,1,1,1 --adaptive_weight=1  --gpu=2 > log/HDGM_3000_adaweight.log &

nohup python TST_HIGGS.py --weight=1,1,1,1,1,1 --adaptive_weight=1  --gpu=2 > log/HIGGS_5000_adaweight.log &

nohup python TST_MNIST.py --weight=1,1,1,1,1,1 --adaptive_weight=1  --gpu=1 > log/MNIST_500_adaweight.log &

nohup python TST_CIFAR10.py --weight=1,1,1,1,1,1 --adaptive_weight=1  --gpu=2 > log/CIFAR10_500_adaweight.log &

# One For All
nohup python TST_MNIST.py --weight=1,0,0,0,0,0  --gpu=1 > log/MNIST_500_1_0_0_0_0_0.log &

nohup python TST_MNIST.py --weight=0,1,0,0,0,0  --gpu=1 > log/MNIST_500_0_1_0_0_0_0.log &

nohup python TST_MNIST.py --weight=0,0,1,0,0,0  --gpu=1 > log/MNIST_500_0_0_1_0_0_0.log &

nohup python TST_MNIST.py --weight=0,0,0,1,0,0  --gpu=1 > log/MNIST_500_0_0_0_1_0_0.log &

nohup python TST_MNIST.py --weight=0,0,0,0,1,0  --gpu=1 > log/MNIST_500_0_0_0_0_1_0.log &

nohup python TST_MNIST.py --weight=0,0,0,0,0,1  --gpu=1 > log/MNIST_500_0_0_0_0_0_1.log &


# Leave One Out
nohup python TST_MNIST.py --weight=1,45,1,1,60,0  --gpu=1 > log/MNIST_500_1_45_1_1_60_0.log &

nohup python TST_MNIST.py --weight=1,45,1,1,0,1  --gpu=2 > log/MNIST_500_1_45_1_1_0_1.log &

nohup python TST_MNIST.py --weight=1,45,1,0,60,1  --gpu=3 > log/MNIST_500_1_45_1_0_60_1.log &

nohup python TST_MNIST.py --weight=1,45,0,1,60,1  --gpu=1 > log/MNIST_500_1_45_0_1_60_1.log &

nohup python TST_MNIST.py --weight=1,0,1,1,60,1  --gpu=2 > log/MNIST_500_1_0_1_1_60_1.log &

nohup python TST_MNIST.py --weight=0,45,1,1,60,1  --gpu=3 > log/MNIST_500_0_45_1_1_60_1.log &

# transferability between surrogate and taregt TSTs
nohup python TST_Blob.py --surrogate=1  --gpu=3 > log/Blob_100_surrogate.log &

nohup python TST_HDGM.py --surrogate=1  --gpu=0 > log/HDGM_3000_surrogate.log &

nohup python TST_HIGGS.py --surrogate=1  --gpu=0 > log/HIGGS_5000_surrogate.log &

nohup python TST_MNIST.py --surrogate=1  --gpu=2 > log/MNIST_500_surrogate.log &

nohup python TST_CIFAR10.py --surrogate=1  --gpu=2 > log/CIFAR10_500_surrogate.log &

# replace_P
nohup python TST_Blob.py --replace_P=1  --gpu=2 > log/Blob_100_replace_P.log &

nohup python TST_HDGM.py --replace_P=1  --gpu=2 > log/HDGM_3000_replace_P.log &

nohup python TST_HIGGS.py --replace_P=1  --gpu=2 > log/HIGGS_5000_replace_P.log &

nohup python TST_MNIST.py --replace_P=1  --gpu=1 > log/MNIST_500_replace_P.log &

nohup python TST_CIFAR10.py --replace_P=1  --gpu=2 > log/CIFAR10_500_replace_P.log &

# robust kernel benign pair + adv pair

nohup python TST_Blob.py --weight=5,1,1,20,1,1,5 --robust_kernel=1 --BA=1 --gpu=0 > log/Blob_100_5_1_1_20_1_1_5_robust_BA.log &

nohup python TST_HDGM.py --weight=25,1,1,50,1,1,25 --robust_kernel=1 --BA=1 --gpu=3 > log/HDGM_3000_25_1_1_50_1_1_25_robust_BA.log &

nohup python TST_HIGGS.py --weight=3,45,4,3,40,3,3 --robust_kernel=1 --BA=1 --gpu=0 > log/HIGGS_5000_3_45_4_3_40_3_3_robust_BA.log &

nohup python TST_MNIST.py --weight=1,45,1,1,60,1,1 --robust_kernel=1 --BA=1 --gpu=0 > log/MNIST_500_1_45_1_1_60_1_1_robust_BA.log &

nohup python TST_CIFAR10.py --weight=1,50,4,4,20,1,1 --robust_kernel=1 --BA=1 --gpu=0 > log/CIFAR10_500_1_50_4_4_20_1_1_robust_BA.log &

# type1 in the benign settings
nohup python TST_Blob.py --type1=1 --gpu=2 > log/Blob_100_type1.log &

nohup python TST_HDGM.py --type1=1 --gpu=0 > log/HDGM_3000_type1.log &

nohup python TST_HIGGS.py --type1=1  --gpu=2 > log/HIGGS_5000_type1.log &

nohup python TST_MNIST.py --type1=1  --gpu=1 > log/MNIST_500_type1.log &

nohup python TST_CIFAR10.py --type1=1  --gpu=3 > log/CIFAR10_500_type1.log &