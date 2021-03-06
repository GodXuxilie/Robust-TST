import os
import numpy as np
import torch
import argparse
from TST_tools import MMD_D, MMD_G, C2ST_S, C2ST_L, ME, SCF, MMD_RoD
from TST_attack import two_sample_test_attack
from TST_utils import MatConvert

parser = argparse.ArgumentParser()
### experimental configuration ###
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed1', type=int, default=1102)
parser.add_argument('--seed2', type=int, default=819)
parser.add_argument('--trails', type=int, default=10, help='repeating times')
### data set configuration ###
parser.add_argument('--n', type=int, default=3000, help='number of data in each set')
parser.add_argument('--d', type=int, default=10, help='dimension of samples (default value is 10)')
parser.add_argument('--type1', type=int, default=0, help='whether to test Type-1 error')
### train and test procedure configuration ###
parser.add_argument('--n_epochs', type=int, default=2000, help='number of training epochs')
parser.add_argument('--WB', type=int, default=1, help='whether to use wild bootstrap')
parser.add_argument('--ln', type=float, default=0.5, help='hyper parameters in wild bootstrap')
### TST attack configuration ###
parser.add_argument('--num_steps', type=int, default=50, help='maximum perturbation step K')
parser.add_argument('--epsilon', type=float, default=0.05, help='perturbation bound')
parser.add_argument('--step_size', type=float, default=0.05, help='step size')
parser.add_argument('--dynamic_eta', type=int, default=1, help='whether to use dynamic stepsize scheduling')
parser.add_argument('--ball', type=str, default='l_inf', choices=['l_inf', 'l_2'])
parser.add_argument('--verbose', type=int, default=0, help='whether to print logs')
parser.add_argument('--weight', type=str, default='25,1,1,50,1,1', help='attack weight')
parser.add_argument('--adaptive_weight', type=int, default=0, help='whether to use adaptive reweighting')
parser.add_argument('--surrogate', type=int, default=0, help='whether to use surrogate non-parametric TSTs to attack target TSTs')
parser.add_argument('--replace_P', type=int, default=0, help='whether to replace P with P_prime')
### MMD-RoD configuration ###
parser.add_argument('--robust_kernel', type=int, default=0, help='whether to adversarially train deep kernels')
parser.add_argument('--lr_RoD', type=float, default=0.00001, help='learning rate for MMD-RoD')
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
d = args.d # dimension of data
n = args.n # number of samples in per mode
x_in = d # number of neurons in the input layer, i.e., dimension of data
H = 3*d # number of neurons in the hidden layer
x_out = 3*d # number of neurons in the output layer
learning_rate_MMD_D = 0.00001
learning_rate_MMD_G = 0.001
learning_rate_C2ST = 0.001
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

Results = np.zeros([8,K])
ATTACKResults = np.zeros([8,K])

weight_args = args.weight.split(",")
weight = [int(x) for x in weight_args]
weight = [x / sum(weight) for x in weight]

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
    epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), device, dtype))
    sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * d), device, dtype)
    sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.1), device, dtype)
    MMD_D_test = MMD_D(HD=False, model=ModelLatentF(x_in, H, x_out), 
                        parameters=(epsilonOPT, sigmaOPT, sigma0OPT), hyperparameters=(learning_rate_MMD_D, N_epoch))
    MMD_D_test.train(s1, s2)
    return MMD_D_test

def train_MMD_G(s1, s2):
    S = np.concatenate((s1, s2), axis=0)
    S = MatConvert(S, device, dtype)
    sigma0 = 2*d * torch.rand([1]).to(device, dtype)
    MMD_G_test = MMD_G(HD=False, parameters=sigma0, hyperparameters=(learning_rate_MMD_G, N_epoch))
    MMD_G_test.train(s1, s2)
    return MMD_G_test

def train_C2ST_S(s1, s2):
    C2ST_S_test = C2ST_S(HD=False, hyperparameters=(x_in, H, x_out, learning_rate_C2ST,
                                                                        N_epoch_C, batch_size))
    C2ST_S_test.train(s1, s2)
    return C2ST_S_test

def train_C2ST_L(s1, s2):
    C2ST_L_test = C2ST_L(HD=False, hyperparameters=(x_in, H, x_out, learning_rate_C2ST,
                                                                        N_epoch_C, batch_size))
    C2ST_L_test.train(s1, s2)      
    return C2ST_L_test

def train_MMD_RoD(s1, s2):
    epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), device, dtype))
    sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * d), device, dtype)
    sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.1), device, dtype)
    MMD_RoD_test = MMD_RoD(HD=False, model=ModelLatentF(x_in, H, x_out), parameters=(epsilonOPT, sigmaOPT, sigma0OPT), 
                                        hyperparameters=(args.lr_RoD, N_epoch, args.num_steps_RoD, 
                                        args.epsilon, args.step_size, args.dynamic_eta, args.verbose))
    MMD_RoD_test.train(s1, s2)
    return MMD_RoD_test

def train_ME(s1, s2):
    ME_test = ME(HD=False, hyperparameters=(alpha, 1,1,5,15))
    ME_test.train(s1, s2)
    return ME_test

def train_SCF(s1, s2):
    SCF_test = SCF(HD=False, hyperparameters=(alpha, 1,1,5,15))
    SCF_test.train(s1, s2)
    return SCF_test

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
    
    # Set up training random seed
    np.random.seed(seed1)
    torch.manual_seed(seed1)
    torch.cuda.manual_seed(seed1)
    # Train MMD-D
    MMD_D_test = train_MMD_D(s1, s2)
    # Train MMD-G
    MMD_G_test = train_MMD_G(s1, s2)
    # Train C2ST-S
    C2ST_S_test = train_C2ST_S(s1, s2)
    # Train C2ST-L
    C2ST_L_test = train_C2ST_L(s1, s2)
    # Train ME
    ME_test = train_ME(s1, s2)
    # Train SCF
    SCF_test = train_SCF(s1, s2)
    # Train MMD-RoD:
    if args.robust_kernel:
        MMD_RoD_test = train_MMD_RoD(s1 ,s2)
    
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
            H_D[k], H_G[k], H_ME[k], H_SCF[k], H_C2ST_S[k], H_C2ST_L[k], H_RoD[k], H_Ensemble[k] = test_procedure(s1, s2,
                    MMD_D_test, MMD_G_test, C2ST_S_test, C2ST_L_test, ME_test, SCF_test, MMD_RoD_test)
        else:
            H_D[k], H_G[k], H_ME[k], H_SCF[k], H_C2ST_S[k], H_C2ST_L[k], H_RoD[k], H_Ensemble[k] = test_procedure(s1, s2,
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
        MMD_D_test_surrogate = train_MMD_D(s1_surrogate, s2_surrogate)
        # Train MMD-G
        MMD_G_test_surrogate = train_MMD_G(s1_surrogate, s2_surrogate)
        # Train C2ST-S
        C2ST_S_test_surrogate = train_C2ST_S(s1_surrogate, s2_surrogate)
        # Train C2ST-L
        C2ST_L_test_surrogate = train_C2ST_L(s1_surrogate, s2_surrogate)
        # Train ME
        ME_test_surrogate = train_ME(s1_surrogate, s2_surrogate)
        # Train SCF
        SCF_test_surrogate = train_SCF(s1_surrogate, s2_surrogate)

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

        adv_s2 = TSTAttack.attack(s1, s2)

        if args.replace_P:
            for i in range(Num_clusters):
                np.random.seed(seed=(seed2+77) * (k + 1) + 2*kk + i + n)
                s1[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)

        if args.robust_kernel:
            H_D_adv[k], H_G_adv[k], H_ME_adv[k], H_SCF_adv[k], H_C2ST_S_adv[k], H_C2ST_L_adv[k], H_RoD_adv[k], H_Ensemble_adv[k] = test_procedure(s1, adv_s2,
                    MMD_D_test, MMD_G_test, C2ST_S_test, C2ST_L_test, ME_test, SCF_test, MMD_RoD_test)
        else:
            H_D_adv[k], H_G_adv[k], H_ME_adv[k], H_SCF_adv[k], H_C2ST_S_adv[k], H_C2ST_L_adv[k], H_RoD_adv[k], H_Ensemble_adv[k] = test_procedure(s1, adv_s2,
                    MMD_D_test, MMD_G_test, C2ST_S_test, C2ST_L_test, ME_test, SCF_test)

        print("Round:", k+1, "MMD-D:", H_D_adv.sum(),  "MMD-G:", H_G_adv.sum(), "C2ST_S: ", H_C2ST_S_adv.sum(), "C2ST_L: ", 
                H_C2ST_L_adv.sum(), "ME:", H_ME_adv.sum(), "SCF:", H_SCF_adv.sum(), "MMD-RoD:", H_RoD_adv.sum(), 'Emsemble: ', H_Ensemble_adv.sum())

        if H_Ensemble_adv[k] == 0:
            np.save('{}/FAKE_ORI_{}'.format(out_dir, save_index), s2)
            np.save('{}/FAKE_ADV_{}'.format(out_dir, save_index), adv_s2)
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