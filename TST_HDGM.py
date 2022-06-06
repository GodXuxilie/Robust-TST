import numpy as np
import torch
import argparse
from TST_utils import MatConvert, TST_LCE, MMDu, TST_MMD_adaptive_bandwidth, TST_ME, TST_SCF, TST_C2ST, C2ST_NN_fit, MMDu, TST_MMD_u, TST_WBMMD_u, TST_MMD_adaptive_WB
from TST_attack import two_sample_test_attack
import os

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=3000, help='number of samples') # 
parser.add_argument('--d', type=int, default=10, help='dimension of samples (default value is 10)')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--epsilon', type=float, default=0.05, help='perturbation bound')
parser.add_argument('--num_steps', type=int, default=50, help='maximum perturbation step K')
parser.add_argument('--step_size', type=float, default=0.05, help='step size')
parser.add_argument('--n_epochs', type=int, default=2000)
parser.add_argument('--float', type=float, default=0.5)
parser.add_argument('--ball', type=str, default='l_inf')
parser.add_argument('--lr_u', type=float, default=0.0001)
parser.add_argument('--type1', type=int, default=0, help='whether to test Type-1 error')
parser.add_argument('--dynamic_eta', type=int, default=1, help='whether to use dynamic stepsize scheduling')
parser.add_argument('--trails', type=int, default=10, help='repeating times')
parser.add_argument('--robust_kernel', type=int, default=0, help='whether to adversarially train deep kernels')
parser.add_argument('--attack_num', type=int, default=1, help='number of steps during adversarial training')
parser.add_argument('--verbose', type=int, default=0, help='whether to print logs')
parser.add_argument('--weight', type=str, default='25,1,1,50,1,1', help='attack weight')
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
weight_tmp = args.weight.split(",")
weight = [int(x) for x in weight_tmp]
weight = [len(weight)/sum(weight) for x in weight]
dtype = torch.float
device = torch.device("cuda")
N_per = 100 # permutation times
alpha = 0.05 # test threshold
d = args.d # dimension of data
n = args.n # number of samples in per mode
x_in = d # number of neurons in the input layer, i.e., dimension of data
H =3*d # number of neurons in the hidden layer
x_out = 3*d # number of neurons in the output layer
learning_rate = 0.00001
learning_ratea = 0.001
batch_size = min(n * 2, 128) # batch size for training C2ST-L and C2ST-S
N_epoch_C = 1000 # number of epochs for training C2ST-L and C2ST-S
N_epoch = 1000 # number of epochs for training MMD-O
K = args.trails # number of trails
N = 100 # # number of test sets
N_f = 100.0 # number of test sets (float)
seed1 = args.seed1
seed2 = args.seed2
np.random.seed(seed1)
torch.manual_seed(seed1)
torch.cuda.manual_seed(seed1)
torch.backends.cudnn.deterministic = True
is_cuda = True
Results = np.zeros([8,K])
ATTACKResults = np.zeros([8,K])

# Setup save directory
out_dir = 'results/HDGM_'
if args.robust_kernel:
    out_dir += 'robust_'
out_dir += str(args.num_steps) + '_' + str(n) + '_' + str(args.weight)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Generate variance and co-variance matrix of Q
Num_clusters = 2 # number of modes
mu_mx = np.zeros([Num_clusters,d])
mu_mx[1] = mu_mx[1] + 0.5
sigma_mx_1 = np.identity(d)
sigma_mx_2 = [np.identity(d),np.identity(d)]
sigma_mx_2[0][0,1] = 0.5
sigma_mx_2[0][1,0] = 0.5
sigma_mx_2[1][0,1] = -0.5
sigma_mx_2[1][1,0] = -0.5
N1 = Num_clusters*n
N2 = Num_clusters*n
s1 = np.zeros([n*Num_clusters, d])
s2 = np.zeros([n*Num_clusters, d])

class ModelLatentF(torch.nn.Module):
    """Latent space for both domains."""
    def __init__(self, x_in, H, x_out):
        """Init latent features."""
        super(ModelLatentF, self).__init__()
        self.restored = False
        self.latent = torch.nn.Sequential(
            torch.nn.Linear(x_in, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, x_out, bias=True),
        )
    def forward(self, input):
        """Forward the LeNet."""
        fealant = self.latent(input)
        return fealant


def train_MMD_D(s1, s2):
    if is_cuda:
        model_u = ModelLatentF(x_in, H, x_out).cuda()
        model_u = torch.nn.DataParallel(model_u)
    else:
        model_u = ModelLatentF(x_in, H, x_out)
    epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), device, dtype))
    epsilonOPT.requires_grad = True
    sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * d), device, dtype)
    sigmaOPT.requires_grad = True
    sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.1), device, dtype)
    sigma0OPT.requires_grad = True
    optimizer_u = torch.optim.Adam(list(model_u.parameters()) + [epsilonOPT] + [sigmaOPT] + [sigma0OPT],
                                   lr=learning_rate)
    for t in range(N_epoch):
        ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))
        sigma = sigmaOPT ** 2
        sigma0_u = sigma0OPT ** 2
        S = np.concatenate((s1, s2), axis=0)
        S = MatConvert(S, device, dtype)
        modelu_output = model_u(S)
        TEMP = MMDu(modelu_output, N1, S, sigma, sigma0_u, ep)
        mmd_value_temp = -1 * (TEMP[0]+10**(-8))
        mmd_std_temp = torch.sqrt(TEMP[1]+10**(-8))
        if mmd_std_temp.item() == 0:
            print('error!!')
        if np.isnan(mmd_std_temp.item()):
            print('error!!')
        STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
        optimizer_u.zero_grad()
        STAT_u.backward(retain_graph=True)
        optimizer_u.step()
        if t % 100 ==0:
            print("mmd_value: ", -1 * mmd_value_temp.item(), "mmd_std: ", mmd_std_temp.item(), "Statistic: ",
                  -1 * STAT_u.item())
    return model_u, sigma, sigma0_u, ep

def train_MMD_G(s1, s2):
    S = np.concatenate((s1, s2), axis=0)
    S = MatConvert(S, device, dtype)
    sigma0 = 2*d * torch.rand([1]).to(device, dtype)
    sigma0.requires_grad = True
    optimizer_sigma0 = torch.optim.Adam([sigma0], lr=learning_ratea)
    for t in range(N_epoch):
        TEMPa = MMDu(S, N1, S, 0, sigma0, is_smooth=False)
        mmd_value_tempa = -1 * (TEMPa[0]+10**(-8))
        mmd_std_tempa = torch.sqrt(TEMPa[1]+10**(-8))
        if mmd_std_tempa.item() == 0:
            print('error!!')
        if np.isnan(mmd_std_tempa.item()):
            print('error!!')
        STAT_adaptive = torch.div(mmd_value_tempa, mmd_std_tempa)
        optimizer_sigma0.zero_grad()
        STAT_adaptive.backward(retain_graph=True)
        optimizer_sigma0.step()
        if t % 100 == 0:
            print("mmd_value: ", -1 * mmd_value_tempa.item(), "mmd_std: ", mmd_std_tempa.item(), "Statistic: ",
                  -1 * STAT_adaptive.item())
    # h_adaptive, threshold_adaptive, mmd_value_adaptive = TST_MMD_adaptive_bandwidth(S, N_per, N1, S, 0, sigma0, alpha, device, dtype)
    # print("h:", h_adaptive, "Threshold:", threshold_adaptive, "MMD_value:", mmd_value_adaptive)
    return sigma0

def train_C2ST_L(s1, s2):
    S = np.concatenate((s1, s2), axis=0)
    S = MatConvert(S, device, dtype)
    y = (torch.cat((torch.zeros(N1, 1), torch.ones(N2, 1)), 0)).squeeze(1).to(device, dtype).long()
    pred, STAT_C2ST_S, model_C2ST_S, w_C2ST_S, b_C2ST_S = C2ST_NN_fit(S, y, N1, x_in, H, x_out, 0.001, N_epoch_C,
                                                              batch_size, device, dtype)
    return model_C2ST_S, w_C2ST_S, b_C2ST_S

def train_C2ST_S(s1, s2):
    S = np.concatenate((s1, s2), axis=0)
    S = MatConvert(S, device, dtype)
    y = (torch.cat((torch.zeros(N1, 1), torch.ones(N2, 1)), 0)).squeeze(1).to(device, dtype).long()
    pred, STAT_C2ST_L, model_C2ST_L, w_C2ST_L, b_C2ST_L = C2ST_NN_fit(S, y, N1, x_in, H, x_out, 0.001, N_epoch_C,
                                                              batch_size, device, dtype)
    return model_C2ST_L, w_C2ST_L, b_C2ST_L

def train_MMD_RoD(s1, s2):
    model_u_RoD = ModelLatentF(x_in, H, x_out).cuda()
    epsilonOPT_RoD = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), device, dtype))
    epsilonOPT_RoD.requires_grad = True
    sigmaOPT_RoD = MatConvert(np.ones(1) * np.sqrt(2 * d), device, dtype)
    sigmaOPT_RoD.requires_grad = True
    sigma0OPT_RoD = MatConvert(np.ones(1) * np.sqrt(0.1), device, dtype)
    sigma0OPT_RoD.requires_grad = True
    optimizer_u_RoD = torch.optim.Adam(list(model_u_RoD.parameters()) + [epsilonOPT_RoD] + [sigmaOPT_RoD] + [sigma0OPT_RoD],lr=args.lr)
    for t in range(N_epoch):
        ep_RoD = torch.exp(epsilonOPT_RoD) / (1 + torch.exp(epsilonOPT_RoD))
        sigma_RoD = sigmaOPT_RoD ** 2
        sigma0_u_RoD = sigma0OPT_RoD ** 2
        S = np.concatenate((s1, s2), axis=0)
        S = MatConvert(S, device, dtype)
        TSTAttack = two_sample_test_attack(num_steps=args.attack_num, epsilon=args.epsilon, step_size=args.step_size, dynamic_eta=args.dynamic_eta, 
                                            verbose=args.verbose, MMD_RoD_args=(model_u_RoD, sigma_RoD, sigma0_u_RoD, ep_RoD), 
                                            max_scale=s2.max(), min_scale=s2.min())
        nat_s2 = s2
        adv_s2 = TSTAttack.attack(torch.Tensor(s1).cuda(), torch.Tensor(s2).cuda(), torch.Tensor(nat_s2).cuda(), weight=[0,0,0,0,0,0,1])
        S = np.concatenate((s1, adv_s2.cpu().numpy()), axis=0)
        S = MatConvert(S, device, dtype)
        modelu_output = model_u_RoD(S)
        TEMP = MMDu(modelu_output, N1, S, sigma_RoD, sigma0_u_RoD, ep_RoD)
        mmd_value_temp = -1 * (TEMP[0]+10**(-8))
        mmd_std_temp = torch.sqrt(TEMP[1]+10**(-8))
        STAT_u = torch.div(mmd_value_temp, mmd_std_temp)

        if args.BA:
            S_nat = np.concatenate((s1, s2), axis=0)
            S_nat = MatConvert(S_nat, device, dtype)
            modelu_output_nat = model_u_RoD(S_nat)
            TEMP_nat = MMDu(modelu_output_nat, N1, S_nat, sigma_RoD, sigma0_u_RoD, ep_RoD)
            mmd_value_temp_nat = -1 * (TEMP_nat[0]+10**(-8))
            mmd_std_temp_nat = torch.sqrt(TEMP_nat[1]+10**(-8))
            STAT_u_nat = torch.div(mmd_value_temp_nat, mmd_std_temp_nat)
            STAT_u = 0.5 * STAT_u + 0.5 * STAT_u_nat

        optimizer_u_RoD.zero_grad()
        STAT_u.backward()
        optimizer_u_RoD.step()
        if t % 100 == 0:
            print("mmd_rod_value: ", -1 * mmd_value_temp.item(), "mmd_rod_std: ", mmd_std_temp.item(), "Statistic J: ",
                -1 * STAT_u.item())           
    # h_u_RoD, threshold_u_RoD, mmd_value_u_RoD = TST_WBMMD_u(model_u_RoD(S), N_per, N1, S, sigma_RoD, sigma0_u_RoD, ep_RoD, alpha, device, dtype, args.ln)
    return model_u_RoD, sigma_RoD, sigma0_u_RoD, ep_RoD

def train_ME(s1, s2):
    S = np.concatenate((s1, s2), axis=0)
    S = MatConvert(S, device, dtype)
    test_locs_ME, gwidth_ME = TST_ME(S, N1, alpha, is_train=True, test_locs=1, gwidth=1, J=5, seed=15)
    # h_ME = TST_ME(S, N1, alpha, is_train=False, test_locs=test_locs_ME, gwidth=gwidth_ME, J=5, seed=15)
    return test_locs_ME, gwidth_ME

def train_SCF(s1, s2):
    S = np.concatenate((s1, s2), axis=0)
    S = MatConvert(S, device, dtype)
    test_freqs_SCF, gwidth_SCF = TST_SCF(S, N1, alpha, is_train=True, test_freqs=1, gwidth=1, J=5, seed=15)
    # h_SCF = TST_SCF(S, N1, alpha, is_train=False, test_freqs=test_freqs_SCF, gwidth=gwidth_SCF, J=5, seed=15)
    return test_freqs_SCF, gwidth_SCF

def test_procedure(S, MMD_D_args, MMD_G_args, ME_args, SCF_args, C2ST_S_args, C2ST_L_args, MMD_RoD_args=None):
    if args.WB:
        # MMD-D
        h_D, _, _ = TST_WBMMD_u(MMD_D_args[0](S), N_per, N1, S, MMD_D_args[1], MMD_D_args[2], MMD_D_args[3], alpha, device, dtype, args.ln)
        # MMD-G
        h_G, _, _ = TST_MMD_adaptive_WB(S, N_per, N1, 0, MMD_G_args, alpha, device, dtype, args.ln)
    else:
        # MMD-D
        h_D, _, _ = TST_MMD_u(MMD_D_args[0](S), N_per, N1, S, MMD_D_args[1], MMD_D_args[2], MMD_D_args[3], alpha, device, dtype)
        # MMD-G
        h_G, _, _ = TST_MMD_adaptive_bandwidth(S, N_per, N1, S, sigma, MMD_G_args, alpha, device, dtype)
    if args.robust_kernel:
        if args.WB:
            h_RoD, _, _ = TST_WBMMD_u(MMD_RoD_args[0](S), N_per, N1, S, MMD_RoD_args[1], MMD_RoD_args[2], MMD_RoD_args[3], alpha, device, dtype, args.ln)
        else:
            h_RoD, _, _ = TST_MMD_u(MMD_RoD_args[0](S), N_per, N1, S, MMD_RoD_args[1], MMD_RoD_args[2], MMD_RoD_args[3], alpha, device, dtype)
    else:
        h_RoD = 0
    
    # C2ST-S
    h_C2ST_S, _, _ = TST_C2ST(S, N1, N_per, alpha, C2ST_S_args[0], C2ST_S_args[1], C2ST_S_args[2], device, dtype)
    # C2ST-L
    h_C2ST_L, _,_ = TST_LCE(S, N1, N_per, alpha, C2ST_L_args[0], C2ST_L_args[1], C2ST_L_args[2], device, dtype)
    # ME
    h_ME = TST_ME(S, N1, alpha, is_train=False, test_locs=ME_args[0], gwidth=ME_args[1], J=5, seed=15)
    # SCF
    h_SCF = TST_SCF(S, N1, alpha, is_train=False, test_freqs=SCF_args[0], gwidth=SCF_args[1], J=5, seed=15)

    if h_D == 0 and h_G ==0 and h_ME == 0 and h_SCF ==0 and h_C2ST_S == 0 and h_C2ST_L == 0 and h_RoD == 0:
        h_Ensemble = 0
    else:
        h_Ensemble = 1

    return h_D, h_G, h_ME, h_SCF, h_C2ST_S, h_C2ST_L, h_RoD, h_Ensemble

# Repeat experiments K times (K = 10) and report average test power (rejection rate)
for kk in range(K):
    # Setup random seed for sampling data
    torch.manual_seed(kk * 19 + n)
    torch.cuda.manual_seed(kk * 19 + n)
    # Generate HDGM-D
    for i in range(Num_clusters):
        np.random.seed(seed=seed1*kk + i + n)
        s1[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)
    for i in range(Num_clusters):
        np.random.seed(seed=seed2*kk + 1 + i + n)
        if not args.type1:
            s2[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_2[i], n)
        else:
            s2[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)

    S = np.concatenate((s1, s2), axis=0)
    S = MatConvert(S, device, dtype)
    
    # Setup random seed for training
    np.random.seed(seed1)
    torch.manual_seed(seed1)
    torch.cuda.manual_seed(seed1)    
    # Train MMD-D
    model_u, sigma, sigma0_u, ep = train_MMD_D(s1, s2)
    # Train MMD-G
    sigma0 = train_MMD_G(s1, s2)
    # Train C2ST-S
    model_C2ST_S, w_C2ST_S, b_C2ST_S = train_C2ST_S(s1, s2)
    # Train C2ST-L
    model_C2ST_L, w_C2ST_L, b_C2ST_L = train_C2ST_L(s1, s2)
    # Train ME
    test_locs_ME, gwidth_ME = train_ME(s1, s2)
    # Train SCF
    test_freqs_SCF, gwidth_SCF = train_SCF(s1, s2)
    # Train MMD-RoD:
    if args.robust_kernel:
        model_u_RoD, sigma_RoD, sigma0_u_RoD, ep_RoD = train_MMD_RoD(s1 ,s2)
    
    print('===> Test power in the benign setting')

    # Compute test power of MMD_D
    H_D = np.zeros(N)
    # Compute test power of MMD_G
    H_G = np.zeros(N)
    # Compute test power of ME
    H_ME = np.zeros(N)
    # Compute test power of SCF
    H_SCF = np.zeros(N)
    # Compute test power of C2ST_S
    H_C2ST_S = np.zeros(N)
    # Compute test power of C2ST_L
    H_C2ST_L = np.zeros(N)
    # Compute test power of MMD_RoD
    H_RoD = np.zeros(N)
    # Compute test power of Ensemble
    H_Ensemble = np.zeros(N)


    for k in range(N):
        # Generate HDGM-D
        for i in range(Num_clusters):
            np.random.seed(seed=seed1 * (k+2) + 2*kk + i + n)
            s1[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)
        for i in range(Num_clusters):
            np.random.seed(seed=seed2 * (k + 1) + 2*kk + i + n)
            if not args.type1:
                s2[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_2[i], n)
            else:
                s2[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)

        S = np.concatenate((s1, s2), axis=0)
        S = MatConvert(S, device, dtype)

        if args.robust_kernel:
            H_D[k], H_G[k], H_ME[k], H_SCF[k], H_C2ST_S[k], H_C2ST_L[k], H_RoD[k], H_Ensemble[k] = test_procedure(S, 
                        MMD_D_args=(model_u, sigma, sigma0_u, ep), 
                        MMD_G_args=sigma0, 
                        ME_args=(test_locs_ME, gwidth_ME), 
                        SCF_args=(test_freqs_SCF, gwidth_SCF),
                        C2ST_L_args=(model_C2ST_L, w_C2ST_L, b_C2ST_L), 
                        C2ST_S_args=(model_C2ST_S, w_C2ST_S, b_C2ST_S), 
                        MMD_RoD_args=(model_u_RoD, sigma_RoD, sigma0_u_RoD, ep_RoD))
        else:
            H_D[k], H_G[k], H_ME[k], H_SCF[k], H_C2ST_S[k], H_C2ST_L[k], H_RoD[k], H_Ensemble[k] = test_procedure(S, 
                        MMD_D_args=(model_u, sigma, sigma0_u, ep), 
                        MMD_G_args=sigma0, 
                        ME_args=(test_locs_ME, gwidth_ME), 
                        SCF_args=(test_freqs_SCF, gwidth_SCF),
                        C2ST_L_args=(model_C2ST_L, w_C2ST_L, b_C2ST_L), 
                        C2ST_S_args=(model_C2ST_S, w_C2ST_S, b_C2ST_S))

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
        s1_surrogate = np.zeros([n*Num_clusters, d])
        s2_surrogate = np.zeros([n*Num_clusters, d])
        # Setup random seed for sampling data
        torch.manual_seed(kk * 999 + n)
        torch.cuda.manual_seed(kk * 999 + n)
        # Generate HDGM-D
        for i in range(Num_clusters):
            np.random.seed(seed=seed1*kk + 999 + n)
            s1_surrogate[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)
        for i in range(Num_clusters):
            np.random.seed(seed=seed2*kk + 999 + i + n)
            if not args.type1:
                s2_surrogate[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_2[i], n)
            else:
                s2_surrogate[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)

        # Set up training random seed
        np.random.seed(seed2)
        torch.manual_seed(seed2)
        torch.cuda.manual_seed(seed2)
        # Train MMD-D
        model_u_surrogate, sigma_surrogate, sigma0_u_surrogate, ep_surrogate = train_MMD_D(s1_surrogate, s2_surrogate)
        # Train MMD-G
        sigma0_surrogate = train_MMD_G(s1_surrogate, s2_surrogate)
        # Train C2ST-S
        model_C2ST_S_surrogate, w_C2ST_S_surrogate, b_C2ST_S_surrogate = train_C2ST_S(s1_surrogate, s2_surrogate)
        # Train C2ST-L
        model_C2ST_L_surrogate, w_C2ST_L_surrogate, b_C2ST_L_surrogate = train_C2ST_L(s1_surrogate, s2_surrogate)
        # Train ME
        test_locs_ME_surrogate, gwidth_ME_surrogate = train_ME(s1_surrogate, s2_surrogate)
        # Train SCF
        test_freqs_SCF_surrogate, gwidth_SCF_surrogate = train_SCF(s1_surrogate, s2_surrogate)

        TSTAttack = two_sample_test_attack(num_steps=args.num_steps, epsilon=args.epsilon,step_size=args.step_size, dynamic_eta=args.dynamic_eta,
                        verbose=args.verbose, max_scale=s2_surrogate.max(), min_scale=s2_surrogate.min(), 
                        MMD_D_args=(model_u_surrogate, sigma_surrogate.detach(), sigma0_u_surrogate.detach(), ep_surrogate.detach()), 
                        MMD_G_args=sigma0.detach(), 
                        ME_args=(test_locs_ME_surrogate, gwidth_ME_surrogate), 
                        C2ST_S_args=(model_C2ST_S_surrogate, w_C2ST_S_surrogate, b_C2ST_S_surrogate), 
                        C2ST_L_args=(model_C2ST_L_surrogate, w_C2ST_L_surrogate, b_C2ST_L_surrogate), 
                        SCF_args=(test_freqs_SCF_surrogate, gwidth_SCF_surrogate))
    else:
        print('===> White-box attack')
        if args.robust_kernel:
            TSTAttack = two_sample_test_attack(num_steps=args.num_steps, epsilon=args.epsilon,step_size=args.step_size, dynamic_eta=args.dynamic_eta,
                        verbose=args.verbose, max_scale=s2.max(), min_scale=s2.min(), 
                        MMD_D_args=(model_u, sigma.detach(), sigma0_u.detach(), ep.detach()), 
                        MMD_G_args=sigma0.detach(), 
                        ME_args=(test_locs_ME, gwidth_ME), 
                        C2ST_S_args=(model_C2ST_S, w_C2ST_S, b_C2ST_S), 
                        C2ST_L_args=(model_C2ST_L, w_C2ST_L, b_C2ST_L), 
                        SCF_args=(test_freqs_SCF, gwidth_SCF),
                        MMD_RoD_args=(model_u_RoD, sigma_RoD.detach(), sigma0_u_RoD.detach(), ep_RoD.detach()))
        else:
            TSTAttack = two_sample_test_attack(num_steps=args.num_steps, epsilon=args.epsilon,step_size=args.step_size, dynamic_eta=args.dynamic_eta,
                        verbose=args.verbose, max_scale=s2.max(), min_scale=s2.min(), 
                        MMD_D_args=(model_u, sigma.detach(), sigma0_u.detach(), ep.detach()), 
                        MMD_G_args=sigma0.detach(), 
                        ME_args=(test_locs_ME, gwidth_ME), 
                        C2ST_S_args=(model_C2ST_S, w_C2ST_S, b_C2ST_S), 
                        C2ST_L_args=(model_C2ST_L, w_C2ST_L, b_C2ST_L), 
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
        # Generate HDGM-D
        for i in range(Num_clusters):
            np.random.seed(seed=seed1 * (k+2) + 2*kk + i + n)
            s1[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)
        for i in range(Num_clusters):
            np.random.seed(seed=seed2 * (k + 1) + 2*kk + i + n)
            if not args.type1:
                s2[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_2[i], n)
            else:
                s2[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)
        nat_s2 = s2
        adv_s2 = TSTAttack.attack(torch.Tensor(s1).cuda(), torch.Tensor(s2).cuda(), torch.Tensor(nat_s2).cuda(), weight=weight)
       
        if args.replace_P:
            for i in range(Num_clusters):
                np.random.seed(seed=(seed2+77) * (k + 1) + 2*kk + i + n)
                s1[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)

        S = np.concatenate((s1, adv_s2.cpu().numpy()), axis=0)
        S = MatConvert(S, device, dtype)

        if args.robust_kernel:
            H_D_adv[k], H_G_adv[k], H_ME_adv[k], H_SCF_adv[k], H_C2ST_S_adv[k], H_C2ST_L_adv[k], H_RoD_adv[k], H_Ensemble_adv[k] = test_procedure(S, 
                        MMD_D_args=( model_u, sigma, sigma0_u, ep), 
                        MMD_G_args=sigma0, 
                        ME_args=(test_locs_ME, gwidth_ME), 
                        SCF_args=(test_freqs_SCF, gwidth_SCF),
                        C2ST_L_args=(model_C2ST_L, w_C2ST_L, b_C2ST_L), 
                        C2ST_S_args=(model_C2ST_S, w_C2ST_S, b_C2ST_S), 
                        MMD_RoD_args=(model_u_RoD, sigma_RoD, sigma0_u_RoD, ep_RoD))
        else:
            H_D_adv[k], H_G_adv[k], H_ME_adv[k], H_SCF_adv[k], H_C2ST_S_adv[k], H_C2ST_L_adv[k], H_RoD_adv[k], H_Ensemble_adv[k] = test_procedure(S, 
                        MMD_D_args=( model_u, sigma, sigma0_u, ep), 
                        MMD_G_args=sigma0, 
                        ME_args=(test_locs_ME, gwidth_ME), 
                        SCF_args=(test_freqs_SCF, gwidth_SCF),
                        C2ST_L_args=(model_C2ST_L, w_C2ST_L, b_C2ST_L), 
                        C2ST_S_args=(model_C2ST_S, w_C2ST_S, b_C2ST_S))

        print("Round:", k+1, "MMD-D:", H_D_adv.sum(),  "MMD-G:", H_G_adv.sum(), "C2ST_S: ", H_C2ST_S_adv.sum(), "C2ST_L: ", 
                H_C2ST_L_adv.sum(), "ME:", H_ME_adv.sum(), "SCF:", H_SCF_adv.sum(), "MMD-RoD:", H_RoD_adv.sum(), 'Emsemble: ', H_Ensemble_adv.sum())

        if H_Ensemble_adv[k] == 0:
            np.save('{}/FAKE_ORI_{}'.format(out_dir, save_index), nat_s2)
            np.save('{}/FAKE_ADV_{}'.format(out_dir, save_index), adv_s2.cpu().numpy())
            np.save('{}/REAL_{}'.format(out_dir, save_index), s1)
            save_index += 1

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
    print("MMD-D: ", np.var(ATTACKResults[0]), "MMD-G: ", np.var(ATTACKResults[1]), 
            "C2ST-S: ", np.var(ATTACKResults[2]), "C2ST-L: ", np.var(ATTACKResults[3]), 
            "ME:", np.var(ATTACKResults[4]), "SCF: ", np.var(ATTACKResults[5]), 
            'MMD-RoD: ', np.var(ATTACKResults[6]), "Ensemble: ", np.var(ATTACKResults[7]))


np.save('{}/Benign_Results'.format(out_dir), Results)
np.save('{}/Adversarial_Results'.format(out_dir), ATTACKResults)