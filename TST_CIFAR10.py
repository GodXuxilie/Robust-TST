import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch
from TST_utils import MatConvert, Pdist2, MMDu, TST_MMD_adaptive_bandwidth, TST_MMD_u, TST_ME, TST_SCF, TST_C2ST_D, TST_LCE_D, TST_WBMMD_u, TST_MMD_adaptive_WB
from TST_attack import two_sample_test_attack

# parameters setting
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument('--data_dir', type=str, default='../data', help='path to save raw data')
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate for C2STs")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--epsilon', type=float, default=0.0314, help='perturbation bound')
parser.add_argument('--num_steps', type=int, default=50, help='maximum perturbation step K')
parser.add_argument('--step_size', type=float, default=0.0314, help='step size')
parser.add_argument('--class1', type=int, default=3, help='distribution P')
parser.add_argument('--class2', type=int, default=5, help='distribution Q')
parser.add_argument('--n', type=int, default=500, help="number of samples in one set")
parser.add_argument('--float', type=float, default=0.5)
parser.add_argument('--ball', type=str, default='l_inf')
parser.add_argument('--lr_u', type=float, default=0.0002)
parser.add_argument('--type1', type=int, default=0, help='whether to test Type-1 error')
parser.add_argument('--dynamic_eta', type=int, default=1, help='whether to use dynamic stepsize scheduling')
parser.add_argument('--trails', type=int, default=10, help='repeating times')
parser.add_argument('--robust_kernel', type=int, default=0, help='whether to adversarially train deep kernels')
parser.add_argument('--attack_num', type=int, default=1, help='number of steps during adversarial training')
parser.add_argument('--verbose', type=int, default=0, help='whether to print logs')
parser.add_argument('--weight', type=str, default='1,50,4,4,20,1', help='attack weight')
parser.add_argument('--adaptive_weight', type=int, default=0, help='whether to use adaptive reweighting')
parser.add_argument('--surrogate', type=int, default=0, help='whether to use surrogate non-parametric TSTs to attack target TSTs')
parser.add_argument('--WB', type=int, default=1, help='whether to use wild bootstrap')
parser.add_argument('--ln', type=float, default=0.5, help='hyper parameters in wild bootstrap')
parser.add_argument('--replace_P', type=int, default=0, help='whether to replace P with P_prime')
parser.add_argument('--seed1', type=int, default=1102)
parser.add_argument('--seed2', type=int, default=819)
parser.add_argument('--BA', type=int, default=0, help='whether to use benign and adversarial data together during adversarially training deep kernels')
args = parser.parse_args()
print(args)

# Setup for experiments
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
dtype = torch.float
device = torch.device("cuda")
cuda = True if torch.cuda.is_available() else False
is_cuda = True
N_per = 100 # permutation times
alpha = 0.05 # test threshold
N1 = args.n # number of samples in one set
K = args.trails # number of trails
N = 100 # number of test sets
N_f = 100.0 # number of test sets (float)
seed1 = args.seed1 
seed2 = args.seed2
steps = args.num_steps
step_size = args.step_size
epsilon = args.epsilon
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
category = ['airplane','automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
Results = np.zeros([8,K])
ATTACKResults = np.zeros([8,K])
weight_tmp = args.weight.split(",")
weight = [int(x) for x in weight_tmp]
weight = [x / sum(weight) for x in weight]
adversarial_loss = torch.nn.CrossEntropyLoss()
np.random.seed(seed1)
torch.manual_seed(seed1)
torch.cuda.manual_seed(seed1)
torch.backends.cudnn.deterministic = True

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
ori_nat_data_class2_test = []
for i in range(len(dataset_test)):
    if dataset_test[i][1] == args.class1:
        data_class1_test.append(dataset_test[i][0].numpy())
    if dataset_test[i][1] == args.class2:
        ori_data_class2_test.append(dataset_test[i][0].numpy())
        ori_nat_data_class2_test.append(dataset_test[i][0].numpy())


if not args.type1:
    data_class1_test = torch.Tensor(data_class1_test)
    data_class2_test = torch.Tensor(ori_data_class2_test)
    nat_data_class2_test = torch.Tensor(ori_nat_data_class2_test)
else:
    data_class1_test = torch.Tensor(data_class1_test)
    data_class2_test = torch.Tensor(data_class1_test)
    nat_data_class2_test = torch.Tensor(data_class1_test)

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
    featurizer = Featurizer()
    featurizer = torch.nn.DataParallel(featurizer).cuda()
    epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10),  device, dtype))
    epsilonOPT.requires_grad = True
    sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * 32 * 32), device, dtype)
    sigmaOPT.requires_grad = True
    sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.005), device, dtype)
    sigma0OPT.requires_grad = True
    optimizer_F = torch.optim.Adam(list(featurizer.parameters()) + [epsilonOPT] + [sigmaOPT] + [sigma0OPT], lr=0.0002)
    for epoch in range(args.n_epochs):
        for i, data in enumerate(zip(dataloader_class1, dataloader_class2)):
            real_imgs = data[0][0]
            Fake_imgs = data[1][0]
            # Adversarial ground truths
            valid = Variable(Tensor(real_imgs.shape[0], 1).fill_(0.0), requires_grad=False)
            fake = Variable(Tensor(Fake_imgs.shape[0], 1).fill_(1.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(real_imgs.type(Tensor))
            Fake_imgs = Variable(Fake_imgs.type(Tensor))
            X = torch.cat([real_imgs, Fake_imgs], 0).cuda()
            Y = torch.cat([valid, fake], 0).squeeze().long().cuda()

            ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))
            sigma = sigmaOPT ** 2
            sigma0_u = sigma0OPT ** 2
            optimizer_F.zero_grad()
            modelu_output = featurizer(X)
            TEMP = MMDu(modelu_output, real_imgs.shape[0], X.view(X.shape[0],-1), sigma, sigma0_u, ep)
            mmd_value_temp = -1 * (TEMP[0])
            mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
            STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
            STAT_u.backward()
            optimizer_F.step()
            if (epoch+1) % 100 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [Stat: %f] "
                    % (epoch, args.n_epochs, i, len(dataloader_class1), -STAT_u.item())
                )
    return featurizer,  sigma, sigma0_u, ep

def train_MMD_RoD(dataloader_class1, dataloader_class2):
    featurizer_RoD = Featurizer()
    featurizer_RoD = torch.nn.DataParallel(featurizer_RoD).cuda()
    epsilonOPT_RoD = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10),  device, dtype))
    epsilonOPT_RoD.requires_grad = True
    sigmaOPT_RoD = MatConvert(np.ones(1) * np.sqrt(2 * 32 * 32), device, dtype)
    sigmaOPT_RoD.requires_grad = True
    sigma0OPT_RoD = MatConvert(np.ones(1) * np.sqrt(0.005), device, dtype)
    sigma0OPT_RoD.requires_grad = True
    optimizer_F_RoD = torch.optim.Adam(list(featurizer_RoD.parameters()) + [epsilonOPT_RoD] + [sigmaOPT_RoD] + [sigma0OPT_RoD], lr=args.lr_u)
    
    for epoch in range(args.n_epochs):
        for i, data in enumerate(zip(dataloader_class1, dataloader_class2)):
            real_imgs = data[0][0]
            Fake_imgs = data[1][0]
            valid = Variable(Tensor(real_imgs.shape[0], 1).fill_(0.0), requires_grad=False)
            fake = Variable(Tensor(Fake_imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            real_imgs = Variable(real_imgs.type(Tensor))
            Fake_imgs = Variable(Fake_imgs.type(Tensor))
            X = torch.cat([real_imgs, Fake_imgs], 0).cuda()
            Y = torch.cat([valid, fake], 0).squeeze().long().cuda()
            ep_RoD = torch.exp(epsilonOPT_RoD) / (1 + torch.exp(epsilonOPT_RoD))
            sigma_RoD = sigmaOPT_RoD ** 2
            sigma0_u_RoD = sigma0OPT_RoD ** 2

            TSTAttack = two_sample_test_attack(num_steps=args.attack_num, epsilon=args.epsilon, step_size=args.step_size, dynamic_eta=args.dynamic_eta, 
                                            verbose=args.verbose, MMD_RoD_args=(featurizer_RoD, sigma_RoD.detach(), sigma0_u_RoD.detach(), ep_RoD.detach()), 
                                            max_scale=Fake_imgs.max(), min_scale=Fake_imgs.min())
            adv_Fake_imgs = TSTAttack.attack(real_imgs.cuda(), Fake_imgs.cuda(), Fake_imgs.cuda(), weight=[0,0,0,0,0,0,1])
            X = torch.cat([real_imgs, adv_Fake_imgs], 0).cuda()

            optimizer_F_RoD.zero_grad()
            modelu_output = featurizer_RoD(X)
            TEMP = MMDu(modelu_output, real_imgs.shape[0], X.view(X.shape[0],-1), sigma_RoD, sigma0_u_RoD, ep_RoD)
            mmd_value_temp = -1 * (TEMP[0])
            mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
            STAT_u = torch.div(mmd_value_temp, mmd_std_temp)

            if args.BA:
                X_nat = torch.cat([real_imgs, Fake_imgs], 0).cuda()
                modelu_output_nat = featurizer_RoD(X_nat)
                TEMP_nat = MMDu(modelu_output_nat, real_imgs.shape[0], X_nat.view(X_nat.shape[0],-1), sigma_RoD, sigma0_u_RoD, ep_RoD)
                mmd_value_temp_nat = -1 * (TEMP_nat[0])
                mmd_std_temp_nat = torch.sqrt(TEMP_nat[1] + 10 ** (-8))
                STAT_u_nat = torch.div(mmd_value_temp_nat, mmd_std_temp_nat)
                STAT_u = 0.5 * STAT_u + 0.5 * STAT_u_nat

            STAT_u.backward()
            optimizer_F_RoD.step()
            if (epoch+1) % 100 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [Stat: %f]"
                    % (epoch, args.n_epochs, i,len(dataloader_class1), -STAT_u.item())
                )
    return featurizer_RoD,  sigma_RoD, sigma0_u_RoD, ep_RoD

def train_C2ST(dataloader_class1, dataloader_class2):
    discriminator = Discriminator()
    discriminator = torch.nn.DataParallel(discriminator).cuda()
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    for epoch in range(args.n_epochs):
        for i, data in enumerate(zip(dataloader_class1, dataloader_class2)):
            real_imgs = data[0][0]
            Fake_imgs = data[1][0]
            # Adversarial ground truths
            valid = Variable(Tensor(real_imgs.shape[0], 1).fill_(0.0), requires_grad=False)
            fake = Variable(Tensor(Fake_imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            real_imgs = Variable(real_imgs.type(Tensor))
            Fake_imgs = Variable(Fake_imgs.type(Tensor))
            X = torch.cat([real_imgs, Fake_imgs], 0).cuda()
            Y = torch.cat([valid, fake], 0).squeeze().long().cuda()
            optimizer_D.zero_grad()
            X = torch.cat([real_imgs, Fake_imgs], 0).cuda()
            output = discriminator(X)
            pred = output.max(1, keepdim=True)[1]
            loss_C = adversarial_loss(output, Y)
            loss_C.backward()
            optimizer_D.step()

            if (epoch+1) % 100 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] "
                    % (epoch, args.n_epochs, i, len(dataloader_class1), loss_C.item())
                )

    return discriminator

def train_MMD_G(Sv):
    Dxy = Pdist2(Sv[:args.n, :], Sv[args.n:, :])
    sigma0 = Dxy.median()
    sigma0.requires_grad = True
    optimizer_sigma0 = torch.optim.Adam([sigma0], lr=0.0005)
    for t in range(args.n_epochs):
        TEMPa = MMDu(Sv, args.n, Sv, 0, sigma0, is_smooth=False)
        mmd_value_tempa = -1 * (TEMPa[0] + 10 ** (-8))
        mmd_std_tempa = torch.sqrt(TEMPa[1] + 10 ** (-8))
        STAT_adaptive = torch.div(mmd_value_tempa, mmd_std_tempa)
        optimizer_sigma0.zero_grad()
        STAT_adaptive.backward(retain_graph=True)
        optimizer_sigma0.step()
        if t % 100 == 0:
            print("mmd: ", -1 * mmd_value_tempa.item(), "mmd_std: ", mmd_std_tempa.item(), "Statistic: ",
                  -1 * STAT_adaptive.item())
    # h_adaptive, threshold_adaptive, mmd_value_adaptive = TST_MMD_adaptive_bandwidth(Sv, N_per, args.n, Sv, sigma, sigma0, alpha,
                                                                                    # device, dtype)
    # print("h:", h_adaptive, "Threshold:", threshold_adaptive, "MMD_value:", mmd_value_adaptive)
    return sigma0

def train_ME(Sv):
    test_locs_ME, gwidth_ME = TST_ME(Sv, args.n, alpha, is_train=True, test_locs=1, gwidth=1, J=5, seed=15)
    # h_ME = TST_ME(Sv, args.n, alpha, is_train=False, test_locs=test_locs_ME, gwidth=gwidth_ME, J=5, seed=15)
    return test_locs_ME, gwidth_ME

def train_SCF(Sv):
    test_freqs_SCF, gwidth_SCF = TST_SCF(Sv, args.n, alpha, is_train=True, test_freqs=1, gwidth=1, J=5, seed=15)
    # h_SCF = TST_SCF(Sv, args.n, alpha, is_train=False, test_freqs=test_freqs_SCF, gwidth=gwidth_SCF, J=5, seed=15)
    return test_freqs_SCF, gwidth_SCF

def test_procedure(S, MMD_D_args, MMD_G_args, ME_args, SCF_args, C2ST_S_args, C2ST_L_args, MMD_RoD_args=None):
    Sv = S.view(2*n,-1)
    if args.WB:
        # MMD-D
        h_D, _, _ = TST_WBMMD_u(MMD_D_args[0](S), N_per, n, Sv, MMD_D_args[1], MMD_D_args[2], MMD_D_args[3], alpha, device, dtype, args.ln)
        # MMD-G
        h_G, _, _ = TST_MMD_adaptive_WB(Sv, N_per, n, 0, MMD_G_args, alpha, device, dtype, args.ln)
    else:
        # MMD-D
        h_D, _, _ = TST_MMD_u(MMD_D_args[0](S), N_per, n, Sv, MMD_D_args[1], MMD_D_args[2], MMD_D_args[3], alpha, device, dtype)
        # MMD-O
        h_G, _, _ = TST_MMD_adaptive_bandwidth(Sv, N_per, n, Sv, sigma, MMD_G_args, alpha, device, dtype)
    if args.robust_kernel:
        if args.WB:
            h_RoD, _, _ = TST_WBMMD_u(MMD_RoD_args[0](S), N_per, n, Sv, MMD_RoD_args[1], MMD_RoD_args[2], MMD_RoD_args[3], alpha, device, dtype, args.ln)
        else:
            h_RoD, _, _ = TST_MMD_u(MMD_RoD_args[0](S), N_per, n, Sv, MMD_RoD_args[1], MMD_RoD_args[2], MMD_RoD_args[3], alpha, device, dtype)
    else:
        h_RoD = 0
    # ME
    h_ME = TST_ME(Sv, n, alpha, is_train=False, test_locs=ME_args[0], gwidth=ME_args[1], J=5, seed=15)
    # SCF
    h_SCF = TST_SCF(Sv, n, alpha, is_train=False, test_freqs=SCF_args[0], gwidth=SCF_args[1], J=5, seed=15)
    # C2ST-S
    h_C2ST_S, _, _= TST_C2ST_D(S, n, N_per, alpha, C2ST_S_args, device, dtype)
    # C2ST-L
    h_C2ST_L, _, _ = TST_LCE_D(S, n, N_per, alpha, C2ST_L_args, device, dtype)

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
    torch.manual_seed(seed1)
    torch.cuda.manual_seed(seed1)
    np.random.seed(seed1)
    # Train MMD-D
    featurizer,  sigma, sigma0_u, ep = train_MMD_D(dataloader_class1, dataloader_class2)
    # Train MMD-G
    sigma0 = train_MMD_G(Sv)
    # Train C2ST
    discriminator = train_C2ST(dataloader_class1, dataloader_class2)
    # Train ME
    test_locs_ME, gwidth_ME =  train_ME(Sv)
    # Train SCF
    test_freqs_SCF, gwidth_SCF  = train_SCF(Sv)
    # Train MMD-RoD
    if args.robust_kernel:
        featurizer_RoD,  sigma_RoD, sigma0_u_RoD, ep_RoD = train_MMD_RoD(dataloader_class1, dataloader_class2)


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

        S = torch.cat([s1.cpu(), s2.cpu()], 0).cuda()
        Sv = S.view(2 * n, -1)
        
        if args.robust_kernel:
            H_D[k], H_G[k], H_ME[k], H_SCF[k], H_C2ST_S[k], H_C2ST_L[k], H_RoD[k], H_Ensemble[k] = test_procedure(S, 
                        MMD_D_args=(featurizer, sigma, sigma0_u, ep), 
                        MMD_G_args=sigma0,
                        ME_args=(test_locs_ME, gwidth_ME), 
                        SCF_args=(test_freqs_SCF, gwidth_SCF),
                        C2ST_L_args=discriminator, 
                        C2ST_S_args=discriminator, 
                        MMD_RoD_args=(featurizer_RoD, sigma_RoD, sigma0_u_RoD, ep_RoD))
        else:
            H_D[k], H_G[k], H_ME[k], H_SCF[k], H_C2ST_S[k], H_C2ST_L[k], H_RoD[k], H_Ensemble[k] = test_procedure(S, 
                        MMD_D_args=(featurizer, sigma, sigma0_u, ep), 
                        MMD_G_args=sigma0, 
                        ME_args=(test_locs_ME, gwidth_ME), 
                        SCF_args=(test_freqs_SCF, gwidth_SCF),
                        C2ST_L_args=discriminator, 
                        C2ST_S_args=discriminator)

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
        torch.manual_seed(seed2)
        torch.cuda.manual_seed(seed2)
        np.random.seed(seed2)
        # Train MMD-D
        featurizer_surrogate,  sigma_surrogate, sigma0_u_surrogate, ep_surrogate = train_MMD_D(dataloader_class1_surrogate, dataloader_class2_surrogate)
        # Train MMD-G
        sigma0_surrogate = train_MMD_G(Sv_surrogate)
        # Train C2ST
        discriminator_surrogate = train_C2ST(dataloader_class1_surrogate, dataloader_class2_surrogate)
        # Train ME
        test_locs_ME_surrogate, gwidth_ME_surrogate =  train_ME(Sv_surrogate)
        # Train SCF
        test_freqs_SCF_surrogate, gwidth_SCF_surrogate  = train_SCF(Sv_surrogate)
        
        TSTAttack = two_sample_test_attack(num_steps=args.num_steps, epsilon=args.epsilon,step_size=args.step_size, dynamic_eta=args.dynamic_eta,
                        verbose=args.verbose, max_scale=s2.max(), min_scale=s2.min(), 
                        MMD_D_args=(featurizer_surrogate, sigma_surrogate.detach(), sigma0_u_surrogate.detach(), ep_surrogate.detach()), 
                        MMD_G_args=sigma0_surrogate.detach(), 
                        ME_args=(test_locs_ME_surrogate, gwidth_ME_surrogate), 
                        C2ST_S_args=discriminator_surrogate, 
                        C2ST_L_args=discriminator_surrogate, 
                        SCF_args=(test_freqs_SCF_surrogate, gwidth_SCF_surrogate))
    
    else:
        print('===> White-box attack')
        if args.robust_kernel:
            TSTAttack = two_sample_test_attack(num_steps=args.num_steps, epsilon=args.epsilon,step_size=args.step_size, dynamic_eta=args.dynamic_eta,
                        verbose=args.verbose, max_scale=s2.max(), min_scale=s2.min(), 
                        MMD_D_args=(featurizer, sigma.detach(), sigma0_u.detach(), ep.detach()), 
                        MMD_G_args=sigma0.detach(), 
                        ME_args=(test_locs_ME, gwidth_ME), 
                        C2ST_S_args=discriminator, 
                        C2ST_L_args=discriminator, 
                        SCF_args=(test_freqs_SCF, gwidth_SCF),
                        MMD_RoD_args=(featurizer_RoD, sigma_RoD.detach(), sigma0_u_RoD.detach(), ep_RoD.detach()))
        else:
            TSTAttack = two_sample_test_attack(num_steps=args.num_steps, epsilon=args.epsilon,step_size=args.step_size, dynamic_eta=args.dynamic_eta,
                        verbose=args.verbose, max_scale=s2.max(), min_scale=s2.min(), 
                        MMD_D_args=(featurizer, sigma.detach(), sigma0_u.detach(), ep.detach()), 
                        MMD_G_args=sigma0.detach(), 
                        ME_args=(test_locs_ME, gwidth_ME), 
                        C2ST_S_args=discriminator, 
                        C2ST_L_args=discriminator, 
                        SCF_args=(test_freqs_SCF, gwidth_SCF))

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
        nat_s2 = Variable(nat_data_class2_test[Ind_te_class2[ind_Fake]].type(Tensor))

        adv_s2 = TSTAttack.attack(s1.cuda(), s2.cuda(), nat_s2.cuda(), weight=weight)
        print(torch.max(adv_s2 - nat_s2), torch.min(adv_s2 - nat_s2))
        
        if args.replace_P:
            np.random.seed(seed=seed2 * (k + 666) + N1)
            ind_real = np.random.choice(len(Ind_te_class1), n, replace=False)
            s1 = data_class1_test[Ind_te_class1[ind_real]]

        S = torch.cat([s1.cpu(), adv_s2.cpu()], 0).cuda()
        Sv = S.view(2 * n, -1)

        if args.robust_kernel:
            H_D_adv[k], H_G_adv[k], H_ME_adv[k], H_SCF_adv[k], H_C2ST_S_adv[k], H_C2ST_L_adv[k], H_RoD_adv[k], H_Ensemble_adv[k] = test_procedure(S, 
                        MMD_D_args=(featurizer, sigma, sigma0_u, ep), 
                        MMD_G_args=sigma0, 
                        ME_args=(test_locs_ME, gwidth_ME), 
                        SCF_args=(test_freqs_SCF, gwidth_SCF),
                        C2ST_L_args=discriminator, 
                        C2ST_S_args=discriminator, 
                        MMD_RoD_args=(featurizer_RoD, sigma_RoD, sigma0_u_RoD, ep_RoD))
        else:
            H_D_adv[k], H_G_adv[k], H_ME_adv[k], H_SCF_adv[k], H_C2ST_S_adv[k], H_C2ST_L_adv[k], H_RoD_adv[k], H_Ensemble_adv[k] = test_procedure(S, 
                        MMD_D_args=(featurizer, sigma, sigma0_u, ep), 
                        MMD_G_args=sigma0, 
                        ME_args=(test_locs_ME, gwidth_ME), 
                        SCF_args=(test_freqs_SCF, gwidth_SCF),
                        C2ST_L_args=discriminator, 
                        C2ST_S_args=discriminator)


        if H_Ensemble_adv[k] == 0:
            np.save('{}/FAKE_ORI_{}'.format(out_dir, save_index), nat_s2.cpu().numpy())
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