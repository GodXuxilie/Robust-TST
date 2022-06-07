import numpy as np
import torch
from TST_utils import MatConvert, Pdist2_S, MMDu, TST_LCE, TST_ME, TST_SCF, TST_C2ST, C2ST_NN_fit, MMDu, TST_MMD_u, TST_WBMMD_u, \
                        TST_MMD_adaptive_WB, h1_mean_var_gram,compute_ME_stat,compute_SCF_stat,TST_MMD_adaptive_bandwidth,TST_C2ST_D,TST_LCE_D
from TST_attack import two_sample_test_attack
import copy
from torch.autograd import Variable

class Two_Sample_Test:
    def __init__(self):
        pass
    def train(self, s1, s2):
        pass
    def test(self, s1, s2, N_per=100, alpha=0.05, WB=1, ln=0.5):
        pass
    def cal_test_criterion(self, X, n):
        pass

class MMD_D(Two_Sample_Test):
    def __init__(self, model, parameters, hyperparameters, device, dtype, HD):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.hyperparameters = hyperparameters
        self.HD = HD
        self.deep_model = model.to(device)
        self.deep_model = torch.nn.DataParallel(self.deep_model)
        self.epsilonOPT = parameters[0]
        self.sigmaOPT = parameters[1]
        self.sigma0OPT = parameters[2]
        self.epsilonOPT.requires_grad = True
        self.sigmaOPT.requires_grad = True
        self.sigma0OPT.requires_grad = True
        
    def train(self, s1, s2):
        print('==> begin training MMD-D')
        if not self.HD:
            n = len(s1)
            S = np.concatenate((s1, s2), axis=0)
            S = MatConvert(S, self.device, self.dtype)
            optimizer_u = torch.optim.Adam(list(self.deep_model.parameters())+[self.epsilonOPT]+[self.sigmaOPT]+[self.sigma0OPT], lr=self.hyperparameters[0]) #
            S = np.concatenate((s1, s2), axis=0)
            S = MatConvert(S, self.device, self.dtype)
            for t in range(self.hyperparameters[1]):
                ep = torch.exp(self.epsilonOPT)/(1+torch.exp(self.epsilonOPT))
                sigma = self.sigmaOPT ** 2
                sigma0_u = self.sigma0OPT ** 2
                S = np.concatenate((s1, s2), axis=0)
                S = MatConvert(S, self.device, self.dtype)
                optimizer_u.zero_grad()
                modelu_output = self.deep_model(S)
                TEMP = MMDu(modelu_output, n, S, sigma, sigma0_u, ep)
                mmd_value_temp = -1 * TEMP[0]
                mmd_std_temp = torch.sqrt(TEMP[1]+10**(-8))
                STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
                STAT_u.backward(retain_graph=True)
                optimizer_u.step()
                if t % 100 == 0:
                    print("mmd_value: ", -1 * mmd_value_temp.item(), "mmd_std: ", mmd_std_temp.item(), "Statistic J: ",
                            -1 * STAT_u.item())
        else:
            optimizer_u = torch.optim.Adam(list(self.deep_model.parameters())+[self.epsilonOPT]+[self.sigmaOPT]+[self.sigma0OPT], lr=self.hyperparameters[0])
            Tensor = torch.cuda.FloatTensor
            for epoch in range(self.hyperparameters[1]):
                for i, data in enumerate(zip(s1, s2)):
                    real_imgs = data[0][0]
                    Fake_imgs = data[1][0]
                    real_imgs = Variable(real_imgs.type(Tensor))
                    Fake_imgs = Variable(Fake_imgs.type(Tensor))
                    X = torch.cat([real_imgs, Fake_imgs], 0)
                    ep = torch.exp(self.epsilonOPT) / (1 + torch.exp(self.epsilonOPT))
                    sigma = self.sigmaOPT ** 2
                    sigma0_u = self.sigma0OPT ** 2
                    optimizer_u.zero_grad()
                    modelu_output = self.deep_model(X)
                    TEMP = MMDu(modelu_output, real_imgs.shape[0], X.view(X.shape[0],-1), sigma, sigma0_u, ep)
                    mmd_value_temp = -1 * (TEMP[0])
                    mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
                    STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
                    STAT_u.backward()
                    optimizer_u.step()
                    if (epoch+1) % 100 == 0:
                        print(
                            "[Epoch %d/%d] [Batch %d/%d][Stat J: %f]"
                            % (epoch, self.hyperparameters[1], i, len(s1), -STAT_u.item())
                        )
        
        self.sigma = sigma.detach()
        self.sigma0_u = sigma0_u.detach()
        self.ep = ep.detach()
        self.MMD_D_args = (self.deep_model, self.sigma, self.sigma0_u, self.ep)
        print('==> finish training MMD-D')

    def test(self, s1, s2, N_per=100, alpha=0.05, WB=1, ln=0.5):
        n = len(s1)
        if not self.HD:
            S = np.concatenate((s1, s2), axis=0)
            S = MatConvert(S, self.device, self.dtype)
        else:
            S = torch.cat([s1.cpu(),s2.cpu()],0).cuda()
        if WB:
            if not self.HD:
                h_D, _, _ = TST_WBMMD_u(self.MMD_D_args[0](S), N_per, n, S, self.MMD_D_args[1], self.MMD_D_args[2], self.MMD_D_args[3], alpha, self.device, self.dtype, ln)
            else:
                h_D, _, _ = TST_WBMMD_u(self.MMD_D_args[0](S), N_per, n, S.view(2 * n, -1), self.MMD_D_args[1], self.MMD_D_args[2], self.MMD_D_args[3], alpha, self.device, self.dtype, ln)
        else:
            if not self.HD:
                h_D, _, _ = TST_MMD_u(self.MMD_D_args[0](S), N_per, n, S, self.MMD_D_args[1], self.MMD_D_args[2], self.MMD_D_args[3], alpha, self.device, self.dtype)
            else:
                h_D, _, _ = TST_MMD_u(self.MMD_D_args[0](S), N_per, n, S.view(2 * n, -1), self.MMD_D_args[1], self.MMD_D_args[2], self.MMD_D_args[3], alpha, self.device, self.dtype)
        return h_D

    def cal_test_criterion(self, X, n):
        modelu_output = self.MMD_D_args[0](X)
        TEMP = MMDu(modelu_output, n, X.view(X.shape[0],-1), self.MMD_D_args[1], self.MMD_D_args[2], self.MMD_D_args[3])
        mmd_value_temp = TEMP[0]
        mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
        MMD_D_test_criterion = torch.div(mmd_value_temp, mmd_std_temp)
        return MMD_D_test_criterion

class MMD_RoD(Two_Sample_Test):
    def __init__(self, model, parameters, hyperparameters, device, dtype, HD):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.hyperparameters = hyperparameters
        self.HD = HD
        self.deep_model = model.to(device)
        self.deep_model = torch.nn.DataParallel(self.deep_model)
        self.epsilonOPT = parameters[0]
        self.sigmaOPT = parameters[1]
        self.sigma0OPT = parameters[2]
        self.epsilonOPT.requires_grad = True
        self.sigmaOPT.requires_grad = True
        self.sigma0OPT.requires_grad = True
        
    def train(self, s1, s2):
        print('==> begin training MMD-RoD')
        if not self.HD:
            n = len(s1)
            S = np.concatenate((s1, s2), axis=0)
            S = MatConvert(S, self.device, self.dtype)
            optimizer_u = torch.optim.Adam(list(self.deep_model.parameters())+[self.epsilonOPT]+[self.sigmaOPT]+[self.sigma0OPT], lr=self.hyperparameters[0]) #
            S = np.concatenate((s1, s2), axis=0)
            S = MatConvert(S, self.device, self.dtype)
            for t in range(self.hyperparameters[1]):
                ep = torch.exp(self.epsilonOPT)/(1+torch.exp(self.epsilonOPT))
                sigma = self.sigmaOPT ** 2
                sigma0_u = self.sigma0OPT ** 2
                self.sigma = sigma.detach()
                self.sigma0_u = sigma0_u.detach()
                self.ep = ep.detach()
                self.MMD_RoD_args = (self.deep_model, self.sigma, self.sigma0_u, self.ep)
                TSTAttack = two_sample_test_attack(num_steps=self.hyperparameters[2], epsilon=self.hyperparameters[3], step_size=self.hyperparameters[4], dynamic_eta=self.hyperparameters[5], 
                                                verbose=self.hyperparameters[6],max_scale=s2.max(), min_scale=s2.min(), test_args=[(self, 1)])
                adv_s2 = TSTAttack.attack(s1, s2)
                S = np.concatenate((s1, adv_s2), axis=0)
                S = MatConvert(S, self.device, self.dtype)
                optimizer_u.zero_grad()
                modelu_output = self.deep_model(S)
                TEMP = MMDu(modelu_output, n, S, sigma, sigma0_u, ep)
                mmd_value_temp = -1 * TEMP[0]
                mmd_std_temp = torch.sqrt(TEMP[1]+10**(-8))
                STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
                STAT_u.backward(retain_graph=True)
                optimizer_u.step()
                if t % 100 == 0:
                    print("mmd_value_robust: ", -1 * mmd_value_temp.item(), "mmd_std_robust: ", mmd_std_temp.item(), "Statistic J: ",
                            -1 * STAT_u.item())
        else:
            optimizer_u = torch.optim.Adam(list(self.deep_model.parameters())+[self.epsilonOPT]+[self.sigmaOPT]+[self.sigma0OPT], lr=self.hyperparameters[0])
            Tensor = torch.cuda.FloatTensor
            for epoch in range(self.hyperparameters[1]):
                for i, data in enumerate(zip(s1, s2)):
                    real_imgs = data[0][0]
                    Fake_imgs = data[1][0]
                    real_imgs = Variable(real_imgs.type(Tensor))
                    Fake_imgs = Variable(Fake_imgs.type(Tensor))
                    X = torch.cat([real_imgs, Fake_imgs], 0)
                    ep = torch.exp(self.epsilonOPT) / (1 + torch.exp(self.epsilonOPT))
                    sigma = self.sigmaOPT ** 2
                    sigma0_u = self.sigma0OPT ** 2
                    self.sigma = sigma.detach()
                    self.sigma0_u = sigma0_u.detach()
                    self.ep = ep.detach()
                    self.MMD_RoD_args = (self.deep_model, self.sigma, self.sigma0_u, self.ep)
                    TSTAttack = two_sample_test_attack(num_steps=self.hyperparameters[2], epsilon=self.hyperparameters[3], step_size=self.hyperparameters[4], dynamic_eta=self.hyperparameters[5], 
                                                    verbose=self.hyperparameters[6],max_scale=Fake_imgs.max(), min_scale=Fake_imgs.min(), test_args=[(self, 1)])
                    adv_s2 = TSTAttack.attack(s1, s2)
                    S = np.concatenate((s1, adv_s2), axis=0)
                    X = MatConvert(S, self.device, self.dtype)
                    optimizer_u.zero_grad()
                    modelu_output = self.deep_model(X)
                    TEMP = MMDu(modelu_output, real_imgs.shape[0], X.view(X.shape[0],-1), sigma, sigma0_u, ep)
                    mmd_value_temp = -1 * (TEMP[0])
                    mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
                    STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
                    STAT_u.backward()
                    optimizer_u.step()
                    if (epoch+1) % 100 == 0:
                        print(
                            "[Epoch %d/%d] [Batch %d/%d][Robust Stat J: %f]"
                            % (epoch, self.hyperparameters[1], i, len(s1), -STAT_u.item())
                        )
        
        self.sigma = sigma.detach()
        self.sigma0_u = sigma0_u.detach()
        self.ep = ep.detach()
        self.MMD_RoD_args = (self.deep_model, self.sigma, self.sigma0_u, self.ep)
        print('==> finish training MMD-RoD')

    def test(self, s1, s2, N_per=100, alpha=0.05, WB=1, ln=0.5):
        n = len(s1)
        if not self.HD:
            S = np.concatenate((s1, s2), axis=0)
            S = MatConvert(S, self.device, self.dtype)
        else:
            S = torch.cat([s1.cpu(),s2.cpu()],0).cuda()
        if WB:
            if not self.HD:
                h_D, _, _ = TST_WBMMD_u(self.MMD_RoD_args[0](S), N_per, n, S,self.MMD_RoD_args[1], self.MMD_RoD_args[2], self.MMD_RoD_args[3], alpha, self.device, self.dtype, ln)
            else:
                h_D, _, _ = TST_WBMMD_u(self.MMD_RoD_args[0](S), N_per, n, S.view(2 * n, -1) ,self.MMD_RoD_args[1], self.MMD_RoD_args[2], self.MMD_RoD_args[3], alpha, self.device, self.dtype, ln)
        else:
            if not self.HD:
                h_D, _, _ = TST_MMD_u(self.MMD_RoD_args[0](S), N_per, n, S, self.MMD_RoD_args[1], self.MMD_RoD_args[2], self.MMD_RoD_args[3], alpha, self.device, self.dtype)
            else:
                h_D, _, _ = TST_MMD_u(self.MMD_RoD_args[0](S), N_per, n, S.view(2 * n, -1), self.MMD_RoD_args[1], self.MMD_RoD_args[2], self.MMD_RoD_args[3], alpha, self.device, self.dtype)
        return h_D

    def cal_test_criterion(self, X, n):
        modelu_output = self.MMD_RoD_args[0](X)
        TEMP = MMDu(modelu_output, n, X.view(X.shape[0],-1), self.MMD_RoD_args[1], self.MMD_RoD_args[2], self.MMD_RoD_args[3])
        mmd_value_temp = TEMP[0]
        mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
        MMD_RoD_test_criterion = torch.div(mmd_value_temp, mmd_std_temp)
        return MMD_RoD_test_criterion

class MMD_G(Two_Sample_Test):
    def __init__(self):
        super().__init__()

    def __init__(self, device, dtype, HD, parameters, hyperparameters):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.hyperparameters = hyperparameters
        self.HD = HD
        self.sigma0 = parameters
        self.sigma0.requires_grad = True

    def train(self, s1, s2):
        print('==> begin training MMD-G')
        n = len(s1)
        if not self.HD:
            S = np.concatenate((s1, s2), axis=0)
            S = MatConvert(S, self.device, self.dtype)
        else:
            S = torch.cat([s1.cpu(),s2.cpu()],0).cuda()
            S = S.view(2 * n, -1)
        optimizer_sigma0 = torch.optim.Adam([self.sigma0], lr=self.hyperparameters[0])
        for t in range(self.hyperparameters[1]):
            TEMPa = MMDu(S, n, S, 0, self.sigma0, is_smooth=False)
            mmd_value_tempa = -1 * (TEMPa[0]+10**(-8))
            mmd_std_tempa = torch.sqrt(TEMPa[1]+10**(-8))
            STAT_adaptive = torch.div(mmd_value_tempa, mmd_std_tempa)
            optimizer_sigma0.zero_grad()
            STAT_adaptive.backward(retain_graph=True)
            optimizer_sigma0.step()
            if t % 100 == 0:
                print("mmd_value: ", -1 * mmd_value_tempa.item(), "mmd_std: ", mmd_std_tempa.item(), "Statistic J: ",
                        -1 * STAT_adaptive.item())
        print('==> finish training MMD-G')
    
    def test(self, s1, s2, N_per=100, alpha=0.05, WB=1, ln=0.5):
        n = len(s1)
        if not self.HD:
            S = np.concatenate((s1, s2), axis=0)
            S = MatConvert(S, self.device, self.dtype)
        else:
            S = torch.cat([s1.cpu(),s2.cpu()],0).cuda()
            S = S.view(2 * n, -1)
        if WB:
            h_G, _, _ = TST_MMD_adaptive_WB(S, N_per, n, 0, self.sigma0, alpha, self.device, self.dtype, ln)
        else:
            h_G, _, _ = TST_MMD_adaptive_bandwidth(S, N_per, n, S, self.sigma0, self.sigma0, alpha, self.device, self.dtype)
        return h_G
    
    def cal_test_criterion(self, X, n):
        Sv = X.view(2*n,-1)
        TEMPa = MMDu(Sv, n, Sv, self.sigma0, self.sigma0, is_smooth=False)
        mmd_value_tempa = TEMPa[0]
        mmd_std_tempa = torch.sqrt(TEMPa[1] + 10 ** (-8))
        MMD_G_test_criterion = torch.div(mmd_value_tempa, mmd_std_tempa)
        return MMD_G_test_criterion

class C2ST_S(Two_Sample_Test):
    def __init__(self, device, dtype, HD, hyperparameters, discriminator=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.hyperparameters = hyperparameters
        self.HD = HD
        self.discriminator = discriminator
    
    def train(self, s1, s2):
        print('==> begin training C2ST-S')
        if not self.HD:
            N1 = len(s1)
            N2 = len(s2)
            S = np.concatenate((s1, s2), axis=0)
            S = MatConvert(S, self.device, self.dtype)
            y = (torch.cat((torch.zeros(N1, 1), torch.ones(N2, 1)), 0)).squeeze(1).to(self.device, self.dtype).long()
            pred, STAT_C2ST_S, model_C2ST_S, w_C2ST_S, b_C2ST_S = C2ST_NN_fit(S, y, N1, self.hyperparameters[0],
                    self.hyperparameters[1],self.hyperparameters[2],self.hyperparameters[3],
                    self.hyperparameters[4],self.hyperparameters[5], self.device, self.dtype)
            self.C2ST_S_args = (model_C2ST_S, w_C2ST_S, b_C2ST_S)
        else:
            self.discriminator.cuda()
            self.discriminator = torch.nn.DataParallel(self.discriminator)
            optimizaer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.hyperparameters[0])
            Tensor = torch.cuda.FloatTensor
            for epoch in range(self.hyperparameters[1]):
                for i, data in enumerate(zip(s1, s2)):
                    real_imgs = data[0][0]
                    Fake_imgs = data[1][0]
                    valid = Variable(Tensor(real_imgs.shape[0], 1).fill_(0.0), requires_grad=False)
                    fake = Variable(Tensor(real_imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                    real_imgs = Variable(real_imgs.type(Tensor))
                    Fake_imgs = Variable(Fake_imgs.type(Tensor))
                    X = torch.cat([real_imgs, Fake_imgs], 0)
                    Y = torch.cat([valid, fake], 0).squeeze().long()
                    optimizaer_D.zero_grad()
                    X = torch.cat([real_imgs, Fake_imgs], 0).cuda()
                    loss_C = self.hyperparameters[2](self.discriminator(X), Y)
                    loss_C.backward()
                    optimizaer_D.step()
                    if (epoch+1) % 100 == 0:
                        print(
                            "[Epoch %d/%d] [Batch %d/%d] [CE loss: %f]"
                            % (epoch, self.hyperparameters[1], i, len(s1), loss_C.item(),)
                        )
            self.C2ST_S_args = self.discriminator
        print('==> finish training C2ST-S')
    
    def test(self, s1, s2, N_per=100, alpha=0.05, WB=1, ln=0.5):
        n = len(s1)
        S = np.concatenate((s1, s2), axis=0)
        S = MatConvert(S, self.device, self.dtype)
        if not self.HD:
            h_C2ST_S, _, _ = TST_C2ST(S, n, N_per, alpha, self.C2ST_S_args[0], self.C2ST_S_args[1], 
                            self.C2ST_S_args[2], self.device, self.dtype)
        else:
            h_C2ST_S, _, _ = TST_C2ST_D(S, n, N_per, alpha, self.C2ST_S_args, self.device, self.dtype)
        return h_C2ST_S

    def cal_test_criterion(self, X, n):
        if not self.HD:
            f = torch.nn.Softmax()
            output = f(self.C2ST_S_args[0](X).mm(self.C2ST_S_args[1]) + self.C2ST_S_args[2])
        else:
            output = self.C2ST_S_args(X)
        pred_C2ST = (((output - 1/2) / torch.abs(output - 1/2).detach()) + 1 ) / 2
        Dx_pred = pred_C2ST[:n, 0] 
        Dy_pred = pred_C2ST[n:, 0] 
        Kx = Pdist2_S(Dx_pred,Dx_pred)
        Ky = Pdist2_S(Dy_pred,Dy_pred)
        Kxy = Pdist2_S(Dx_pred,Dy_pred)
        TEMP_S = h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed=True, use_1sample_U=True)
        mmd_value_temp_S = TEMP_S[0]
        mmd_std_temp_S = torch.sqrt(TEMP_S[1] + 10 ** (-8))
        C2ST_S_test_criterion = torch.div(mmd_value_temp_S, mmd_std_temp_S)
        return C2ST_S_test_criterion

class C2ST_L(Two_Sample_Test):
    def __init__(self, device, dtype, HD, hyperparameters, discriminator=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.hyperparameters = hyperparameters
        self.HD = HD
        self.discriminator = discriminator
    
    def train(self, s1, s2):
        print('==> begin training C2ST-S')
        if not self.HD:
            N1 = len(s1)
            N2 = len(s2)
            S = np.concatenate((s1, s2), axis=0)
            S = MatConvert(S, self.device, self.dtype)
            y = (torch.cat((torch.zeros(N1, 1), torch.ones(N2, 1)), 0)).squeeze(1).to(self.device, self.dtype).long()
            _, _, model_C2ST_L, w_C2ST_L, b_C2ST_L = C2ST_NN_fit(S, y, N1, self.hyperparameters[0],
                    self.hyperparameters[1],self.hyperparameters[2],self.hyperparameters[3],
                    self.hyperparameters[4],self.hyperparameters[5], self.device, self.dtype)
            self.C2ST_L_args = (model_C2ST_L, w_C2ST_L, b_C2ST_L)
        else:
            self.discriminator.cuda()
            self.discriminator = torch.nn.DataParallel(self.discriminator)
            optimizaer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.hyperparameters[0])
            Tensor = torch.cuda.FloatTensor
            for epoch in range(self.hyperparameters[1]):
                for i, data in enumerate(zip(s1, s2)):
                    real_imgs = data[0][0]
                    Fake_imgs = data[1][0]
                    valid = Variable(Tensor(real_imgs.shape[0], 1).fill_(0.0), requires_grad=False)
                    fake = Variable(Tensor(real_imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                    real_imgs = Variable(real_imgs.type(Tensor))
                    Fake_imgs = Variable(Fake_imgs.type(Tensor))
                    X = torch.cat([real_imgs, Fake_imgs], 0)
                    Y = torch.cat([valid, fake], 0).squeeze().long()
                    optimizaer_D.zero_grad()
                    X = torch.cat([real_imgs, Fake_imgs], 0).cuda()
                    loss_C = self.hyperparameters[2](self.discriminator(X), Y)
                    loss_C.backward()
                    optimizaer_D.step()
                    if (epoch+1) % 100 == 0:
                        print(
                            "[Epoch %d/%d] [Batch %d/%d] [CE loss: %f]"
                            % (epoch, self.hyperparameters[1], i, len(s1), loss_C.item(),)
                        )
            self.C2ST_L_args = self.discriminator
        print('==> finish training C2ST-S')
    
    def test(self, s1, s2, N_per=100, alpha=0.05, WB=1, ln=0.5):
        n = len(s1)
        S = np.concatenate((s1, s2), axis=0)
        S = MatConvert(S, self.device, self.dtype)
        if not self.HD:
            h_C2ST_L, _,_ = TST_LCE(S, n, N_per, alpha, self.C2ST_L_args[0], self.C2ST_L_args[1], 
                            self.C2ST_L_args[2], self.device, self.dtype)
        else:
            h_C2ST_L, _, _ = TST_LCE_D(S, n, N_per, alpha, self.C2ST_L_args, self.device, self.dtype)
        return h_C2ST_L

    def cal_test_criterion(self, X, n):
        if not self.HD:
            f = torch.nn.Softmax()
            output = f(self.C2ST_L_args[0](X).mm(self.C2ST_L_args[1]) + self.C2ST_L_args[2])
        else:
            output = self.C2ST_L_args(X)
        pred_C2ST = (((output - 1/2) / torch.abs(output - 1/2).detach()) + 1 ) / 2
        Dx_pred = pred_C2ST[:n, 0] 
        Dy_pred = pred_C2ST[n:, 0] 
        Kx = Pdist2_S(Dx_pred,Dx_pred)
        Ky = Pdist2_S(Dy_pred,Dy_pred)
        Kxy = Pdist2_S(Dx_pred,Dy_pred)
        TEMP_L = h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed=True, use_1sample_U=True)
        mmd_value_temp_L = TEMP_L[0]
        mmd_std_temp_L = torch.sqrt(TEMP_L[1] + 10 ** (-8))
        C2ST_L_test_criterion = torch.div(mmd_value_temp_L, mmd_std_temp_L)
        return C2ST_L_test_criterion
    
class ME(Two_Sample_Test):
    def __init__(self, device, dtype, HD, hyperparameters):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.hyperparameters = hyperparameters
        self.HD = HD
    
    def train(self, s1, s2):
        print('==> begin training ME')
        n = len(s1)
        if not self.HD:
            S = np.concatenate((s1, s2), axis=0)
            S = MatConvert(S, self.device, self.dtype)
        else:
            S = torch.cat([s1.cpu(),s2.cpu()],0).cuda()
            S = S.view(2 * n, -1)
        test_locs_ME, gwidth_ME = TST_ME(S, n, self.hyperparameters[0], is_train=True, test_locs=self.hyperparameters[1],
                                             gwidth=self.hyperparameters[2], J=self.hyperparameters[3], seed=self.hyperparameters[4])
        self.ME_args = (test_locs_ME, gwidth_ME)
        print('==> finish training ME')
    
    def test(self, s1, s2, N_per=100, alpha=0.05, WB=1, ln=0.5):
        n = len(s1)
        if not self.HD:
            S = np.concatenate((s1, s2), axis=0)
            S = MatConvert(S, self.device, self.dtype)
        else:
            S = torch.cat([s1.cpu(),s2.cpu()],0).cuda()
            S = S.view(2 * n, -1)
        h_ME = TST_ME(S, n, alpha, is_train=False, test_locs=self.ME_args[0], gwidth=self.ME_args[1], 
                                    J=self.hyperparameters[3], seed=self.hyperparameters[4])
        return h_ME

    def cal_test_criterion(self, X, n):
        Sv = X.view(2*n,-1)
        T = torch.Tensor(self.ME_args[0]).cuda()
        try:
            ME_test_criterion = compute_ME_stat(Sv[0:n, :], Sv[n:, :], T, Sv[0:n, :], Sv[n:, :], T, self.ME_args[1], self.ME_args[1], epsilon=1)
            ME_test_criterion = ME_test_criterion.sum()
        except:
            print('cannot compute test criterion of ME.')
            ME_test_criterion= torch.tensor(0).cuda()
        return ME_test_criterion

class SCF(Two_Sample_Test):
    def __init__(self, device, dtype, HD, hyperparameters):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.hyperparameters = hyperparameters
        self.HD = HD
    
    def train(self, s1, s2):
        print('==> begin training SCF')
        n = len(s1)
        if not self.HD:
            S = np.concatenate((s1, s2), axis=0)
            S = MatConvert(S, self.device, self.dtype)
        else:
            S = torch.cat([s1.cpu(),s2.cpu()],0).cuda()
            S = S.view(2 * n, -1)
        test_freqs_SCF, gwidth_SCF = TST_SCF(S, n, self.hyperparameters[0], is_train=True, test_freqs=self.hyperparameters[1],
                                             gwidth=self.hyperparameters[2], J=self.hyperparameters[3], seed=self.hyperparameters[4])
        self.SCF_args = (test_freqs_SCF, gwidth_SCF)
        print('==> finish training SCF')
    
    def test(self, s1, s2, N_per=100, alpha=0.05, WB=1, ln=0.5):
        n = len(s1)
        if not self.HD:
            S = np.concatenate((s1, s2), axis=0)
            S = MatConvert(S, self.device, self.dtype)
        else:
            S = torch.cat([s1.cpu(),s2.cpu()],0).cuda()
            S = S.view(2 * n, -1)
        h_SCF = TST_SCF(S, n, alpha, is_train=False, test_freqs=self.SCF_args[0], gwidth=self.SCF_args[1], 
                                    J=self.hyperparameters[3], seed=self.hyperparameters[4])
        return h_SCF

    def cal_test_criterion(self, X, n):
        Sv = X.view(2*n,-1)
        T = torch.Tensor(self.SCF_args[0]).cuda()
        SCF_test_criterion = compute_SCF_stat(Sv[0:n, :], Sv[n:, :], T, self.SCF_args[1]).sum()
        return SCF_test_criterion