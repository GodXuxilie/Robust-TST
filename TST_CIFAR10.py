import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
import argparse
from torch.autograd import Variable
from TST_tools import MMD_D, MMD_G, C2ST_S, C2ST_L, ME, SCF, MMD_RoD
from TST_attack import two_sample_test_attack
from TST_utils import MatConvert,Pdist2

parser = argparse.ArgumentParser()
### experimental configuration ###
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed1', type=int, default=1102)
parser.add_argument('--seed2', type=int, default=819)
parser.add_argument('--trails', type=int, default=10, help='repeating times')
### data set configuration ###
parser.add_argument('--n', type=int, default=500, help='number of data in each set')
parser.add_argument('--class1', type=int, default=3, help='distribution P')
parser.add_argument('--class2', type=int, default=5, help='distribution Q')
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument('--type1', type=int, default=0, help='whether to test Type-1 error')
### train and test procedure configuration ###
parser.add_argument('--n_epochs', type=int, default=1000, help='number of training epochs')
parser.add_argument('--WB', type=int, default=1, help='whether to use wild bootstrap')
parser.add_argument('--ln', type=float, default=0.5, help='hyper parameters in wild bootstrap')
### TST attack configuration ###
parser.add_argument('--num_steps', type=int, default=50, help='maximum perturbation step K')
parser.add_argument('--epsilon', type=float, default=0.0314, help='perturbation bound')
parser.add_argument('--step_size', type=float, default=0.0314, help='step size')
parser.add_argument('--dynamic_eta', type=int, default=1, help='whether to use dynamic stepsize scheduling')
parser.add_argument('--ball', type=str, default='l_inf', choices=['l_inf', 'l_2'])
parser.add_argument('--verbose', type=int, default=0, help='whether to print logs')
parser.add_argument('--weight', type=str, default='1,50,4,4,20,1', help='attack weight')
parser.add_argument('--adaptive_weight', type=int, default=0, help='whether to use adaptive reweighting')
parser.add_argument('--surrogate', type=int, default=0, help='whether to use surrogate non-parametric TSTs to attack target TSTs')
parser.add_argument('--replace_P', type=int, default=0, help='whether to replace P with P_prime')
### MMD-RoD configuration ###
parser.add_argument('--robust_kernel', type=int, default=0, help='whether to adversarially train deep kernels')
parser.add_argument('--lr_RoD', type=float, default=0.0002, help='learning rate for MMD-RoD')
parser.add_argument('--num_steps_RoD', type=int, default=1, help='number of steps during adversarial training')
parser.add_argument('--BA', type=int, default=0, help='whether to use benign and adversarial data together during adversarially training deep kernels')
args = parser.parse_args()
print(args)

# Setup for experiments
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
dtype = torch.float
device = torch.device("cuda")
N_per = 100 # permutation times
alpha = 0.05 # test threshold
N1 = args.n # number of samples in one set
K = args.trails # number of trails
N = 100 # number of test sets
N_f = 100.0 # number of test sets (float)
n = args.n
learning_rate_MMD_D = 0.0002
learning_rate_C2ST = 0.0001
learning_rate_MMD_G = 0.0005
Tensor = torch.cuda.FloatTensor
category = ['airplane','automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
seed1 = args.seed1 
seed2 = args.seed2
np.random.seed(seed1)
torch.manual_seed(seed1)
torch.cuda.manual_seed(seed1)
torch.backends.cudnn.deterministic = True

Results = np.zeros([8,K])
ATTACKResults = np.zeros([8,K])

weight_args = args.weight.split(",")
weight = [int(x) for x in weight_args]
weight = [x / sum(weight) for x in weight]

adversarial_loss = torch.nn.CrossEntropyLoss()


# Setup save directory
out_dir = 'results/CIFAR10_{}_{}_'.format(category[args.class1], category[args.class2])
if args.robust_kernel:
    out_dir += 'robust_'
out_dir += str(args.num_steps) + '_' + str(args.n) + '_' + str(args.weight)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Configure data loader
dataset_train = datasets.CIFAR10(root=args.data_dir, download=True,train=True,
                           transform=transforms.Compose([
                               transforms.Resize(args.img_size),
                               transforms.ToTensor(),
                            #    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataset_test = datasets.CIFAR10(root=args.data_dir, download=True,train=False,
                           transform=transforms.Compose([
                               transforms.Resize(args.img_size),
                               transforms.ToTensor(),
                            #    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
        
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=50000,
                                             shuffle=True, num_workers=1)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=10000,
                                             shuffle=True, num_workers=1)
# Obtain CIFAR10 images
data_class1_train = []
data_class2_train = []
for i in range(len(dataset_train)):
    if dataset_train[i][1] == args.class1:
        data_class1_train.append(dataset_train[i][0].numpy())
    if dataset_train[i][1] == args.class2:
        data_class2_train.append(dataset_train[i][0].numpy())

if not args.type1:
    data_class1_train = torch.Tensor(data_class1_train)
    data_class2_train = torch.Tensor(data_class2_train)
else:
    data_class1_train = torch.Tensor(data_class1_train)
    data_class2_train = torch.Tensor(data_class1_train)

data_class1_test = []
ori_data_class2_test = []
for i in range(len(dataset_test)):
    if dataset_test[i][1] == args.class1:
        data_class1_test.append(dataset_test[i][0].numpy())
    if dataset_test[i][1] == args.class2:
        ori_data_class2_test.append(dataset_test[i][0].numpy())


if not args.type1:
    data_class1_test = torch.Tensor(data_class1_test)
    data_class2_test = torch.Tensor(ori_data_class2_test)
else:
    data_class1_test = torch.Tensor(data_class1_test)
    data_class2_test = torch.Tensor(data_class1_test)

# Define the deep network for C2ST-S and C2ST-L
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        
        self.width = 2
        self.model = nn.Sequential(
            *discriminator_block(args.channels, 16*self.width, bn=False),
            *discriminator_block(16*self.width, 32*self.width),
            *discriminator_block(32*self.width, 64*self.width),
            *discriminator_block(64*self.width, 128*self.width),
        )

        # The height and width of downsampled image
        ds_size = args.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * self.width * ds_size ** 2, 300),
            nn.ReLU(),
            nn.Linear(300, 2),
            nn.Softmax())
    
    def normalize(self, x):
        return (x - 0.5) / 0.5

    def forward(self, img):
        # img = self.normalize(img)
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

# Define the deep network for MMD-D
class Featurizer(nn.Module):
    def __init__(self):
        super(Featurizer, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0)] #0.25
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(args.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = args.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 300))

    def normalize(self, x):
        return (x - 0.5) / 0.5

    def forward(self, img):
        # img = self.normalize(img)
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        feature = self.adv_layer(out)
        return feature


def train_MMD_D(dataloader_class1, dataloader_class2):
    epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10),  device, dtype))
    sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * 32 * 32), device, dtype)
    sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.005), device, dtype)
    MMD_D_test = MMD_D(HD=True, model=Featurizer(), 
                        parameters=(epsilonOPT, sigmaOPT, sigma0OPT), hyperparameters=(learning_rate_MMD_D, args.n_epochs))
    MMD_D_test.train(dataloader_class1, dataloader_class2)
    return MMD_D_test

def train_MMD_G(s1, s2):
    S = torch.cat([s1.cpu(),s2.cpu()],0).cuda()
    S = S.view(2 * n, -1)
    Dxy = Pdist2(S[:n, :], S[n:, :])
    sigma0 = Dxy.median()
    MMD_G_test = MMD_G(HD=True, parameters=sigma0, hyperparameters=(learning_rate_MMD_G, args.n_epochs))
    MMD_G_test.train(s1, s2)
    return MMD_G_test

def train_C2ST_S(dataloader_class1, dataloader_class2):
    C2ST_S_test = C2ST_S(HD=True, hyperparameters=(learning_rate_C2ST, args.n_epochs, adversarial_loss), discriminator=Discriminator())
    C2ST_S_test.train(dataloader_class1, dataloader_class2)
    return C2ST_S_test

def train_C2ST_L(dataloader_class1, dataloader_class2):
    C2ST_L_test = C2ST_L(HD=True, hyperparameters=(learning_rate_C2ST, args.n_epochs, adversarial_loss), discriminator=Discriminator())
    C2ST_L_test.train(dataloader_class1, dataloader_class2)
    return C2ST_L_test

def train_ME(s1, s2):
    ME_test = ME(HD=True, hyperparameters=(alpha, 1,1,5,15))
    ME_test.train(s1, s2)
    return ME_test

def train_SCF(s1, s2):
    SCF_test = SCF(HD=True, hyperparameters=(alpha, 1,1,5,15))
    SCF_test.train(s1, s2)
    return SCF_test

def train_MMD_RoD(dataloader_class1, dataloader_class2):
    epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), device, dtype))
    sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2*32*32), device, dtype)
    sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.005), device, dtype)
    MMD_RoD_test = MMD_RoD(HD=True, model=Featurizer(), 
                        parameters=(epsilonOPT, sigmaOPT, sigma0OPT), hyperparameters=(args.lr_RoD, args.n_epochs, args.num_steps_RoD, 
                                        args.epsilon, args.step_size, args.dynamic_eta, args.verbose))
    MMD_RoD_test.train(dataloader_class1, dataloader_class2)
    return MMD_RoD_test
    
def test_procedure(s1, s2, MMD_D_test, MMD_G_test, C2ST_S_test, C2ST_L_test, ME_test, SCF_test, MMD_RoD_test=None):
    h_D = MMD_D_test.test(s1, s2, N_per=N_per, alpha=alpha, WB=args.WB, ln=args.ln)
    h_G = MMD_G_test.test(s1, s2, N_per=N_per, alpha=alpha, WB=args.WB, ln=args.ln)
    h_C2ST_S = C2ST_S_test.test(s1, s2, N_per=N_per, alpha=alpha, WB=args.WB, ln=args.ln)
    h_C2ST_L = C2ST_L_test.test(s1, s2, N_per=N_per, alpha=alpha, WB=args.WB, ln=args.ln)
    h_ME = ME_test.test(s1, s2, N_per=N_per, alpha=alpha, WB=args.WB, ln=args.ln)
    h_SCF = SCF_test.test(s1, s2, N_per=N_per, alpha=alpha, WB=args.WB, ln=args.ln)
    if args.robust_kernel:
        h_RoD = MMD_RoD_test.test(s1, s2, N_per=N_per, alpha=alpha, WB=args.WB, ln=args.ln)
    else:
        h_RoD = 0

    if h_D == 0 and h_G ==0 and h_ME == 0 and h_SCF ==0 and h_C2ST_S == 0 and h_C2ST_L == 0 and h_RoD == 0:
        h_Ensemble = 0
    else:
        h_Ensemble = 1
    return h_D, h_G, h_ME, h_SCF, h_C2ST_S, h_C2ST_L, h_RoD, h_Ensemble


# Repeat experiments K times (K = 10) and report average test power (rejection rate)
for kk in range(K):
    # Setup random seed for sampling
    torch.manual_seed(seed1 * (kk + 19) + N1)
    torch.cuda.manual_seed(seed1 * (kk + 19) + N1)
    np.random.seed(seed1 * (kk + 19) + N1)
    # Collect CIFAR10 images
    Ind_tr_class1 = np.random.choice(len(data_class1_train), args.n , replace=False)
    Ind_tr_class2 = np.random.choice(len(data_class2_train), args.n , replace=False)
    train_data_class1 = []
    for i in Ind_tr_class1:
       train_data_class1.append([data_class1_train[i], args.class1])
    dataloader_class1 = torch.utils.data.DataLoader(
        train_data_class1,
        batch_size=args.batch_size,
        shuffle=True,
    )
    train_data_class2 = []
    for i in Ind_tr_class2:
       train_data_class2.append([data_class2_train[i], args.class2])
    dataloader_class2 = torch.utils.data.DataLoader(
        train_data_class2,
        batch_size=args.batch_size,
        shuffle=True,
    )
    Ind_te_class1 = np.random.choice(len(data_class1_test), len(data_class1_test), replace=False)
    Ind_te_class2 = np.random.choice(len(data_class2_test), len(data_class2_test), replace=False)
    # Fetch training data
    s1 = data_class1_train[Ind_tr_class1]
    s2 = data_class2_train[Ind_tr_class2]
    S = torch.cat([s1.cpu(), s2.cpu()], 0).cuda()
    Sv = S.view(2 * args.n, -1)
    

    # Setup random seed for training
    np.random.seed(seed1)
    torch.manual_seed(seed1)
    torch.cuda.manual_seed(seed1)
    # Train MMD-D
    MMD_D_test = train_MMD_D(dataloader_class1, dataloader_class2)
    # Train MMD-G
    MMD_G_test = train_MMD_G(s1, s2)
    # Train C2ST_S
    C2ST_S_test = train_C2ST_S(dataloader_class1, dataloader_class2)
    # Train C2ST_L
    C2ST_L_test = train_C2ST_L(dataloader_class1, dataloader_class2)
    # Train ME
    ME_test =  train_ME(s1, s2)
    # Train SCF
    SCF_test  = train_SCF(s1, s2)
    # Train MMD-RoD
    if args.robust_kernel:
         MMD_RoD_test = train_MMD_RoD(dataloader_class1, dataloader_class2)


    print('===> Test power in the benign setting')

    # Compute test power of MMD_D
    H_D = np.zeros(N)
    # Compute test power of MMD_G
    H_G = np.zeros(N)
    # Compute test power of C2ST_S
    H_C2ST_S = np.zeros(N)
    # Compute test power of C2ST_L
    H_C2ST_L = np.zeros(N)
    # Compute test power of ME
    H_ME = np.zeros(N)
    # Compute test power of SCF
    H_SCF = np.zeros(N)
    # Compute test power of MMD_RoD
    H_RoD = np.zeros(N)
    # Compute test power of Ensemble
    H_Ensemble = np.zeros(N)

    for k in range(N):
        # Fetch test data
        np.random.seed(seed=seed1 * (k + 1) + N1)
        n = args.n
        ind_real = np.random.choice(len(Ind_te_class1), n, replace=False)
        s1 = data_class1_test[Ind_te_class1[ind_real]]

        np.random.seed(seed=seed2 * (k + 3) + N1)
        ind_Fake = np.random.choice(len(Ind_te_class2), n, replace=False)
        s2 = Variable(data_class2_test[Ind_te_class2[ind_Fake]].type(Tensor))
        
        if args.robust_kernel:
            H_D[k], H_G[k], H_ME[k], H_SCF[k], H_C2ST_S[k], H_C2ST_L[k], H_RoD[k], H_Ensemble[k] = test_procedure(s1.cpu(), s2.cpu(),
                    MMD_D_test, MMD_G_test, C2ST_S_test, C2ST_L_test, ME_test, SCF_test, MMD_RoD_test)
        else:
            H_D[k], H_G[k], H_ME[k], H_SCF[k], H_C2ST_S[k], H_C2ST_L[k], H_RoD[k], H_Ensemble[k] = test_procedure(s1.cpu(), s2.cpu(),
                    MMD_D_test, MMD_G_test, C2ST_S_test, C2ST_L_test, ME_test, SCF_test)

        print("Round:", k+1, "MMD-D:", H_D.sum(),  "MMD-G:", H_G.sum(), "C2ST_S: ", H_C2ST_S.sum(), "C2ST_L: ", H_C2ST_L.sum(), "ME:", H_ME.sum(), "SCF:", 
        H_SCF.sum(), "MMD-RoD:", H_RoD.sum(), 'Emsemble: ', H_Ensemble.sum())

   
    Results[0, kk] = H_D.sum() / N_f
    Results[1, kk] = H_G.sum() / N_f
    Results[2, kk] = H_C2ST_S.sum() / N_f
    Results[3, kk] = H_C2ST_L.sum() / N_f
    Results[4, kk] = H_ME.sum() / N_f
    Results[5, kk] = H_SCF.sum() / N_f
    Results[6, kk] = H_RoD.sum() / N_f
    Results[7, kk] = H_Ensemble.sum() / N_f

    print("n =",str(n),"--- Test Power ({} times) in the benign setitngs: ".format(kk))
    print("n =", str(n), "--- Average Test Power of Baselines ({} times): ".format(kk))
    print("MMD-D: ", (Results.sum(1) / (kk+1))[0], "MMD-G: ", (Results.sum(1) / (kk+1))[1], 
            "C2ST-S: ", (Results.sum(1) / (kk+1))[2], "C2ST-L: ", (Results.sum(1) / (kk+1))[3],
            "ME:", (Results.sum(1) / (kk+1))[4], "SCF: ", (Results.sum(1) / (kk+1))[5], 
            "MMD-RoD: ", (Results.sum(1) / (kk+1))[6], "Ensemble: ", (Results.sum(1) / (kk+1))[7])
    print("n =", str(n), "--- Average Test Power of Baselines ({} times) Variance: ".format(kk))
    print("MMD-D: ", np.var(Results[0]), "MMD-G: ", np.var(Results[1]), 
            "C2ST-S: ", np.var(Results[2]),"C2ST-L: ", np.var(Results[3]), 
            "ME:", np.var(Results[4]), "SCF: ", np.var(Results[5]), 
            "MMD-RoD: ", np.var(Results[6]), "Ensemble: ", np.var(Results[7]))


    print('===> Test power under attacks')

    if args.surrogate:
        print('===> Using surrogate non-parametric TSTs to attack target TSTs')
        # Setup random seed for sampling
        torch.manual_seed(seed2 * (kk + 19) + N1)
        torch.cuda.manual_seed(seed2 * (kk + 19) + N1)
        np.random.seed(seed2 * (kk + 19) + N1)
        # Collect CIFAR10 images
        Ind_tr_class1_surrogate = np.random.choice(len(data_class1_train), args.n , replace=False)
        Ind_tr_class2_surrogate = np.random.choice(len(data_class2_train), args.n , replace=False)

        dataloader_class1_surrogate = torch.utils.data.DataLoader(
            data_class2_train[Ind_tr_class2_surrogate],
            batch_size=args.batch_size,
            shuffle=True,
        )   
        dataloader_class2_surrogate = torch.utils.data.DataLoader(
            data_class1_train[Ind_tr_class1_surrogate],
            batch_size=args.batch_size,
            shuffle=True,
        )   

        # Fetch training data
        s1_surrogate = data_class1_train[Ind_tr_class1_surrogate]
        s2_surrogate = data_class2_train[Ind_tr_class2_surrogate]
        S_surrogate = torch.cat([s1_surrogate.cpu(), s2_surrogate.cpu()], 0).cuda()
        Sv_surrogate = S.view(2 * args.n, -1)
        
        # Setup random seed for training
        np.random.seed(seed2)
        torch.manual_seed(seed2)
        torch.cuda.manual_seed(seed2)
        # Train MMD-D
        MMD_D_test_surrogate = train_MMD_D(dataloader_class1_surrogate, dataloader_class2_surrogate)
        # Train MMD-G
        MMD_G_test_surrogate = train_MMD_G(s1_surrogate, s2_surrogate)
        # Train C2ST_S
        C2ST_S_test_surrogate = train_C2ST_S(dataloader_class1_surrogate, dataloader_class2_surrogate)
        # Train C2ST_L
        C2ST_L_test_surrogate = train_C2ST_L(dataloader_class1_surrogate, dataloader_class2_surrogate)
        # Train ME
        ME_test_surrogate =  train_ME(s1_surrogate, s2_surrogate)
        # Train SCF
        SCF_test_surrogate  = train_SCF(s1_surrogate, s2_surrogate)
        
        TSTAttack = two_sample_test_attack(num_steps=args.num_steps, epsilon=args.epsilon,step_size=args.step_size, dynamic_eta=args.dynamic_eta,
                        verbose=args.verbose, max_scale=s2_surrogate.max(), min_scale=s2_surrogate.min(), 
                        test_args=[(MMD_D_test_surrogate, weight[0]), (MMD_G_test, weight[1]), (C2ST_S_test_surrogate, weight[2]), (C2ST_L_test_surrogate, weight[3]),
                                    (ME_test_surrogate, weight[4]), (SCF_test_surrogate, weight[5])])
    else:
        print('===> White-box attack')
        if args.robust_kernel:
            TSTAttack = two_sample_test_attack(num_steps=args.num_steps, epsilon=args.epsilon,step_size=args.step_size, dynamic_eta=args.dynamic_eta,
                        verbose=args.verbose, max_scale=s2.max(), min_scale=s2.min(), 
                        test_args=[(MMD_D_test, weight[0]), (MMD_G_test, weight[1]), (C2ST_S_test, weight[2]), (C2ST_L_test, weight[3]),
                                    (ME_test, weight[4]), (SCF_test, weight[5]), (MMD_RoD_test, weight[6])])
        else:
            TSTAttack = two_sample_test_attack(num_steps=args.num_steps, epsilon=args.epsilon,step_size=args.step_size, dynamic_eta=args.dynamic_eta,
                        verbose=args.verbose, max_scale=s2.max(), min_scale=s2.min(), 
                        test_args=[(MMD_D_test, weight[0]), (MMD_G_test, weight[1]), (C2ST_S_test, weight[2]), (C2ST_L_test, weight[3]),
                                    (ME_test, weight[4]), (SCF_test, weight[5])])

    # Compute test power of MMD_D
    H_D_adv = np.zeros(N)
    # Compute test power of MMD_RoD
    H_RoD_adv = np.zeros(N)
    # Compute test power of MMD_G
    H_G_adv = np.zeros(N)
    # Compute test power of ME
    H_ME_adv = np.zeros(N)
    # Compute test power of SCF
    H_SCF_adv = np.zeros(N)
    # Compute test power of C2ST_S
    H_C2ST_S_adv = np.zeros(N)
    # Compute test power of C2ST_L
    H_C2ST_L_adv = np.zeros(N)
    # Compute test power of Ensemble
    H_Ensemble_adv = np.zeros(N)

    save_index = 0
    
    for k in range(N):
        np.random.seed(seed=seed1 * (k + 1) + N1)
        ind_real = np.random.choice(len(Ind_te_class1), n, replace=False)
        s1 = data_class1_test[Ind_te_class1[ind_real]]

        np.random.seed(seed=seed2 * (k + 3) + N1)
        ind_Fake = np.random.choice(len(Ind_te_class2), n, replace=False)
        s2 = Variable(data_class2_test[Ind_te_class2[ind_Fake]].type(Tensor))

        adv_s2 = TSTAttack.attack(s1, s2)
        
        if args.replace_P:
            np.random.seed(seed=seed2 * (k + 666) + N1)
            ind_real = np.random.choice(len(Ind_te_class1), n, replace=False)
            s1 = data_class1_test[Ind_te_class1[ind_real]]

        if args.robust_kernel:
            H_D[k], H_G[k], H_ME[k], H_SCF[k], H_C2ST_S[k], H_C2ST_L[k], H_RoD[k], H_Ensemble[k] = test_procedure(s1.cpu(), s2.cpu(),
                    MMD_D_test, MMD_G_test, C2ST_S_test, C2ST_L_test, ME_test, SCF_test, MMD_RoD_test)
        else:
            H_D[k], H_G[k], H_ME[k], H_SCF[k], H_C2ST_S[k], H_C2ST_L[k], H_RoD[k], H_Ensemble[k] = test_procedure(s1.cpu(), s2.cpu(),
                    MMD_D_test, MMD_G_test, C2ST_S_test, C2ST_L_test, ME_test, SCF_test)


        if H_Ensemble_adv[k] == 0:
            np.save('{}/FAKE_ORI_{}'.format(out_dir, save_index), s2.cpu().numpy())
            np.save('{}/FAKE_ADV_{}'.format(out_dir, save_index), adv_s2.cpu().numpy())
            np.save('{}/REAL_{}'.format(out_dir, save_index), s1)
            save_index += 1

        print("Round:", k+1, "MMD-D:", H_D_adv.sum(),  "MMD-G:", H_G_adv.sum(), "C2ST_S: ", H_C2ST_S_adv.sum(), "C2ST_L: ", H_C2ST_L_adv.sum(), "ME:", H_ME_adv.sum(), "SCF:", 
        H_SCF_adv.sum(), "MMD-RoD:", H_RoD_adv.sum(), 'Emsemble: ', H_Ensemble_adv.sum())


    ATTACKResults[0, kk] = H_D_adv.sum() / N_f
    ATTACKResults[1, kk] = H_G_adv.sum() / N_f
    ATTACKResults[2, kk] = H_C2ST_S_adv.sum() / N_f
    ATTACKResults[3, kk] = H_C2ST_L_adv.sum() / N_f
    ATTACKResults[4, kk] = H_ME_adv.sum() / N_f
    ATTACKResults[5, kk] = H_SCF_adv.sum() / N_f
    ATTACKResults[6, kk] = H_RoD_adv.sum() / N_f
    ATTACKResults[7, kk] = H_Ensemble_adv.sum() / N_f

    print("n =",str(n),"--- Test Power ({} times) Under Attack: ".format(kk))
    print("n =", str(n), "--- Average Test Power ({} times) Under Attack: ".format(kk))
    print("MMD-D: ", (ATTACKResults.sum(1) / (kk+1))[0], "MMD-G: ", (ATTACKResults.sum(1) / (kk+1))[1], 
            "C2ST-S: ", (ATTACKResults.sum(1) / (kk+1))[2], "C2ST-L: ", (ATTACKResults.sum(1) / (kk+1))[3],
            "ME:", (ATTACKResults.sum(1) / (kk+1))[4], "SCF: ", (ATTACKResults.sum(1) / (kk+1))[5], 
            'MMD-RoD: ', (ATTACKResults.sum(1) / (kk+1))[6], "Ensemble: ", (ATTACKResults.sum(1) / (kk+1))[7])
    print("n =", str(n), "--- Average Test Power ({} times) Variance Under Attack: ".format(kk))
    print("MMD-D: ", np.var(ATTACKResults[0]), "MMD-G: ", np.var(ATTACKResults[1]), "C2ST-S: ", np.var(ATTACKResults[2]),
            "C2ST-L: ", np.var(ATTACKResults[3]), "ME:", np.var(ATTACKResults[4]),
            "SCF: ", np.var(ATTACKResults[5]), 'MMD-RoD: ', np.var(ATTACKResults[6]), "Ensemble: ", np.var(ATTACKResults[7]))

np.save('{}/Benign_Results'.format(out_dir), Results)
np.save('{}/Adversarial_Results'.format(out_dir), ATTACKResults)