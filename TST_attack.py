import torch
import numpy as np
from torch.autograd import Variable
import copy

class two_sample_test_attack:
    def __init__(self, num_steps=50, epsilon=0.031, step_size=0.031, ball='l_inf', dynamic_eta=1, verbose=0, max_scale=1, min_scale=-1,
                    adaptive_weight=0, test_args=None) -> None:
        super(two_sample_test_attack, self).__init__()

        self.ball=ball
        self.epsilon=epsilon
        self.step_size=step_size
        self.num_steps = num_steps
        self.dynamic_eta = dynamic_eta
        self.verbose = verbose
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.adaptive_weight=adaptive_weight
        self.test_args=test_args

    def cal_loss(self,X, n):
        loss_list = []
        weight = []
        for test_item in self.test_args:
            if test_item[1] > 0:
                test_criterion = test_item[0].cal_test_criterion(X, n)
                loss_list.append(test_criterion)
                weight.append(test_item[1])

        # Adaptive reweighting
        if self.adaptive_weight:
            ada_weight = [torch.exp(loss_list[i]).item() for i in range(len(loss_list))]
            weight = torch.Tensor([x / np.sum(ada_weight) for x in ada_weight]).cuda()

        # weighted sum of test criteria
        loss = torch.tensor(0.0).cuda()
        for i in range(len(loss_list)):
            loss += weight[i] * loss_list[i]
        return loss, loss_list

    def check_optimization(self, loss_list, w_j_0, w_j_1, rho=0.75):
        count = 0
        for i in range(w_j_0, w_j_1, 1):
            if loss_list[i+1] < loss_list[i]:
                count += 1
        return count < rho * (w_j_1 - w_j_0)

    def update_delta(self, loss, opt_delta, delta, n, nat_Fake_imgs):
        opt_delta.zero_grad()
        loss.backward()
        if self.ball == 'l_inf':
            delta.grad.sign_()
        if self.ball == 'l_2':
            grad_norms = delta.grad.view(n, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
        opt_delta.step()
        if self.ball == 'l_inf':
            delta.data.clamp_(-self.epsilon, self.epsilon)
            delta.data.add_(nat_Fake_imgs).clamp_(self.min_scale, self.max_scale).sub_(nat_Fake_imgs)
        elif self.ball == 'l_2':
            delta.data.add_(nat_Fake_imgs)
            delta.data.clamp_(self.min_scale, self.max_scale).sub_(nat_Fake_imgs)
            delta.data.renorm_(p=2, dim=0, maxnorm=self.epsilon)
    
    def update_lr(self, opt, eta):
        for param_group in opt.param_groups:
            param_group["lr"] = eta

    def get_real_delta(self, delta, nat_Fake_imgs):
        if self.ball == 'l_inf':
            delta.data.clamp_(-self.epsilon, self.epsilon)
            delta.data.add_(nat_Fake_imgs).clamp_(self.min_scale, self.max_scale).sub_(nat_Fake_imgs)
        elif self.ball == 'l_2':
            delta.data.add_(nat_Fake_imgs)
            delta.data.clamp_(self.min_scale, self.max_scale).sub_(nat_Fake_imgs)
            delta.data.renorm_(p=2, dim=0, maxnorm=self.epsilon)
        return delta

    def attack(self,real_imgs, Fake_imgs):
        real_imgs = torch.Tensor(real_imgs).cuda()
        Fake_imgs = torch.Tensor(Fake_imgs).cuda()
        nat_Fake_imgs = copy.deepcopy(Fake_imgs)
        self.min_scale = torch.min(Fake_imgs).detach().item()
        self.max_scale = torch.max(Fake_imgs).detach().item()
        Tensor = torch.cuda.FloatTensor
        eta = self.step_size

        # Initialize noise
        delta = torch.rand_like(Variable(Fake_imgs.type(Tensor)), requires_grad=True).cuda()
        delta.data.mul_(0.001)

        opt_delta = torch.optim.SGD([delta], lr=eta)
        n = Fake_imgs.shape[0]
        loss_list = torch.zeros(size=(1, self.num_steps+1)).squeeze().cuda()

        x_0 = nat_Fake_imgs + self.get_real_delta(delta, nat_Fake_imgs)
        X_0 = torch.cat([real_imgs, x_0], 0)
        with torch.no_grad():
            loss_0, loss_train_list_0 = self.cal_loss(X_0, n)
        loss_list[0] = loss_0.detach()

        loss, loss_train_list = self.cal_loss(X_0, n)
        self.update_delta(loss, opt_delta, delta, n, nat_Fake_imgs)
        
        x_1 = nat_Fake_imgs + self.get_real_delta(delta, nat_Fake_imgs).detach()
        X_1 = torch.cat([real_imgs, x_1], 0)
        with torch.no_grad():
            loss_1, loss_train_list_1 = self.cal_loss(X_1, n)
        loss_list[1] = loss_1.detach()

        if loss_1.item() < loss_0.item():
            x_min = Variable(x_1, requires_grad=False).cuda()
            loss_min = loss_1.detach().item()
        else:
            x_min = Variable(x_0, requires_grad=False).cuda()
            loss_min = loss_0.detach().item()
        x_k_0 = Variable(x_0, requires_grad=True).cuda() # x_{k-1}
        x_k_1 = nat_Fake_imgs + self.get_real_delta(delta, nat_Fake_imgs).detach()
        
        W = torch.LongTensor([0,2,4,8,12,16,24,32,40,48,56,64,72,88])
        eta_list = torch.Tensor([0]*len(W))
        eta_list[0] = eta
        loss_min_list = torch.Tensor([0]*len(W))
        loss_min_list[0] = loss_0.detach()
        w_index = 1

        delta_min = self.get_real_delta(delta, nat_Fake_imgs).detach()

        for t in range(1, self.num_steps, 1):
            X = torch.cat([real_imgs, nat_Fake_imgs + self.get_real_delta(delta, nat_Fake_imgs)], 0)
            loss, loss_train_list= self.cal_loss(X, n)
            self.update_delta(loss, opt_delta, delta, Fake_imgs, nat_Fake_imgs)
            x_k_2 = nat_Fake_imgs + self.get_real_delta(delta, nat_Fake_imgs)
            X = torch.cat([real_imgs, x_k_2], 0)
            with torch.no_grad():
                loss_k_2, loss_train_list = self.cal_loss(X, n)
            loss_list[t+1] = loss_k_2.detach()
            if loss_k_2.item() < loss_min:
                delta_min = x_k_2 - nat_Fake_imgs
                loss_min = loss_k_2.detach().item()
                loss_min_list[w_index] = loss_min

            if self.dynamic_eta:
                if (t+1) == W[w_index]:
                    w_j_0 = W[w_index-1].item()
                    w_j_1 = W[w_index].item()
                    if self.check_optimization(loss_list, w_j_0, w_j_1):
                        eta /= 2
                        delta.data = delta_min.data
                        x_k_2 = nat_Fake_imgs + self.get_real_delta(delta, nat_Fake_imgs)
                        eta_list[w_index] = eta
                    if eta_list[w_index] == eta_list[w_index-1] and loss_min_list[w_index] == loss_min_list[w_index-1]:
                        eta /= 2
                        delta.data = delta_min.data
                        x_k_2 = nat_Fake_imgs + self.get_real_delta(delta, nat_Fake_imgs)
                        eta_list[w_index] = eta
                    if w_index < len(W) - 1:
                        w_index += 1
                        loss_min_list[w_index] = loss_min_list[w_index-1]
                        eta_list[w_index] == eta_list[w_index-1]
                    self.update_lr(opt_delta, eta)
                
            x_k_0 = x_k_1.detach()
            x_k_1 = x_k_2

            if (t+1) % 10 == 0 and self.verbose:
                print(
                    "[Step %d/%d] [Step size: %f] [Loss : %f] [Loss_D : %f] [Loss_G : %f] [Loss_S: %f] [Loss_L: %f] [Loss_ME: %f] [Loss_SCF: %f]"
                    % (t+1, self.num_steps, eta, loss.item(), loss_train_list[0], loss_train_list[1],loss_train_list[2],loss_train_list[3],loss_train_list[4],loss_train_list[5])
                )
        

        adv_Fake_imgs = nat_Fake_imgs + delta_min.detach()
        X = torch.cat([real_imgs, adv_Fake_imgs], 0)
        with torch.no_grad():
            loss, loss_train_list = self.cal_loss(X, n)
        if self.verbose:
            if self.num_steps == 1:
                t=0
            print(
                "[Step %d/%d] [Step size: %f] [Loss : %f] [Loss_D : %f] [Loss_G : %f] [Loss_C2ST_S: %f] [Loss_C2ST_L: %f] [Loss_ME: %f] [Loss_SCF: %f]"
                % (t+1, self.num_steps, eta, loss.item(), loss_train_list[0], loss_train_list[1],loss_train_list[2],loss_train_list[3],loss_train_list[4],loss_train_list[5])
            )
       
        return adv_Fake_imgs.detach().cpu().numpy()