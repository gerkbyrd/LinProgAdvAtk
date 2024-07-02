import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import scipy
import numpy as np
import time
from tqdm import tqdm
from timeit import default_timer as timer
def _pgd(model, X, y, epsilon, niters=40, alpha=0.01):
    out = model(X)
    ce = nn.CrossEntropyLoss()(out, y)
    err = (out.data.max(1)[1] != y.data).float().sum()  / X.size(0)

    X_pgd = Variable(X.data, requires_grad=True)
    for i in range(niters):
        opt = optim.Adam([X_pgd], lr=1e-3)
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = alpha*X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)

        # adjust to be within [-epsilon, epsilon]
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)

    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum() / X.size(0)
    return err, err_pgd

def pgd(loader, model, epsilon, niters=100, alpha=0.01, verbose=False,
        robust=False):
    return attack(loader, model, epsilon, verbose=verbose, atk=_pgd,
                  robust=robust)

def train_madry(loader, model, epsilon, opt, epoch, steps=40, stepsize=0.01):#for CIFAR, default is steps=7, stepsize=2
    model.train()
    epoch_loss, epoch_err = 0,0
    epoch_ploss, epoch_perr = 0,0
    nsamp=0
    for i, (X,y) in tqdm(enumerate(loader)):
        X,y = X.cuda(), y.cuda()
        # # perturb
        X_pgd = Variable(X, requires_grad=True)
        for _ in range(steps):
            opt_pgd = optim.Adam([X_pgd], lr=1e-3)
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(model(X_pgd), Variable(y))
            loss.backward()
            eta = stepsize*X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)

            # adjust to be within [-epsilon, epsilon]
            eta = torch.clamp(X_pgd.data - X, -epsilon, epsilon)
            X_pgd.data = X + eta
            X_pgd.data = torch.clamp(X_pgd.data, 0, 1)

        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

        pout = model(Variable(X_pgd.data))
        pce = nn.CrossEntropyLoss()(pout, Variable(y))
        perr = (pout.data.max(1)[1] != y).float().sum()  / X.size(0)

        #print(out.data.max(1)[1])
        #print(pout.data.max(1)[1])
        #print(y)

        #print(ce.item())
        #print(pce.item())
        opt.zero_grad()
        pce.backward()
        opt.step()

        epoch_loss=epoch_loss+ce.item()
        epoch_ploss=epoch_ploss+pce.item()
        epoch_err, epoch_perr = epoch_err+err*X.size(0), epoch_perr+perr*X.size(0)
        nsamp=nsamp+X.size(0)
        #input([epoch_loss/nsamp, epoch_err/nsamp, epoch_ploss/nsamp, epoch_perr/nsamp])

    return epoch_loss/nsamp, epoch_err/nsamp, epoch_ploss/nsamp, epoch_perr/nsamp

def evaluate_madry(loader, model, epsilon, epoch, steps=40, stepsize=0.01):
    model.eval()
    epoch_loss, epoch_err = 0,0
    epoch_perr = 0
    nsamp=0
    for i, (X,y) in tqdm(enumerate(loader)):
        X,y = X.cuda(), y.cuda()
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)
        # # perturb
        _, perr = _pgd(model, Variable(X), Variable(y), epsilon=epsilon, niters=steps, alpha=stepsize)

        # measure accuracy and record loss
        epoch_loss=epoch_loss+ce.item()
        epoch_err = epoch_err+err
        epoch_perr = epoch_perr+perr
        nsamp=nsamp+X.size(0)
    return epoch_loss/nsamp, epoch_err/nsamp, epoch_perr/nsamp



def train_em(loader, model, epsilon, opt, epoch, alpha=0.95, interpol=10):
    model.train()
    #end = time.time()
    epoch_loss, epoch_err = 0,0
    epoch_ploss, epoch_perr = 0,0
    nsamp=0
    for i, (X,y) in tqdm(enumerate(loader)):

        X,y = X.cuda(), y.cuda()
        #data_time.update(time.time() - end)

        # # perturb
        X_em = X.clone()
        adlo=torch.permute(( (epsilon/interpol) *torch.arange(-interpol,interpol+1))*torch.ones(X_em[0].shape)[:,:,:,None],(3,0,1,2))
        X_em_top=X_em[:,None,:,:,:]+adlo.cuda()
        #print(X_em.shape[0])
        for ii in range(X_em.shape[0]):

            outs=model(X_em_top[ii,:,:,:,:])

            #one constraint per label!=gt
            """
            A_ub=-np.ones((len(outs[0]) - 1, np.prod(X_em[0].shape) + 1))
            b_ub=np.zeros(len(outs[0]) - 1)
            outs=outs-outs[:,y[ii]][:,None]
            #
            labind=0
            for idx in range(len(outs[0])):
                if idx==y[ii].cpu():
                    continue
                coefs=np.linalg.lstsq(torch.cat([X_em_top[ii,:,:,:,:].view(2*interpol+1,-1), torch.ones((2*interpol+1,1)).cuda()],dim=1).cpu().detach().numpy(), outs[:,idx].cpu().detach().numpy())[0]#constraints
                b_ub[labind]=-coefs[-1]
                A_ub[labind,1:]=coefs[:-1]
                labind=labind+1
            """
            #one constraint for any label!=gt
            A_ub=-np.ones((1, np.prod(X_em[0].shape) + 1))
            #z=timer()
            idx=[i for i in range(len(outs[0]))]
            idx.remove(y[ii].cpu().detach().numpy())
            #input(idx)
            #z=timer()-z
            #a=timer()
            outz=torch.sum(outs[:,idx], dim=1)
            outz=outz-outs[:,y[ii]]


            coefs=np.linalg.lstsq(torch.cat([X_em_top[ii,:,:,:,:].view(2*interpol+1,-1), torch.ones((2*interpol+1,1)).cuda()],dim=1).cpu().detach().numpy(), outz.cpu().detach().numpy())[0]#constraints

            b_ub=-coefs[-1]
            A_ub[0,1:]=coefs[:-1]

            tbounds=[(None,None)]
            xbounds=[(lb,ub) for lb,ub in zip(X_em_top[ii,0,:,:,:].flatten().cpu().detach().numpy(), X_em_top[ii,-1,:,:,:].flatten().cpu().detach().numpy())]

            c=[1] + [0 for i in range(len(xbounds))]


            result = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=tbounds+xbounds)

            #print([z, timer()-a])
            #input(result.x.shape)
            t, Xemj = result.x[0], result.x[1:]
            X_em[ii,:,:,:]=torch.from_numpy(Xemj).cuda().view(X_em[ii,:,:,:].shape)

        # adjust to be within [-epsilon, epsilon]
        eta = torch.clamp(X_em - X, -epsilon, epsilon)
        X_em = X + eta
        X_em = torch.clamp(X_em, 0, 1)

        out = model(X)
        ce = nn.CrossEntropyLoss()(out, y)
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

        pout = model(X_em)
        pce = nn.CrossEntropyLoss()(pout, y)
        perr = (pout.data.max(1)[1] != y).float().sum()  / X.size(0)

        loss=alpha*ce + (1-alpha)*pce
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss=epoch_loss+ce.item()
        epoch_ploss=epoch_ploss+pce.item()
        epoch_err, epoch_perr = epoch_err+err*X.size(0), epoch_perr+perr*X.size(0)
        nsamp=nsamp+X.size(0)

    return epoch_loss/nsamp, epoch_err/nsamp, epoch_ploss/nsamp, epoch_perr/nsamp

def train_pwl(loader, model, epsilon, opt, epoch, alpha=0.95, interpol=10):
    model.train()
    #end = time.time()
    epoch_loss, epoch_err = 0,0
    epoch_ploss, epoch_perr = 0,0
    nsamp=0
    for i, (X,y) in tqdm(enumerate(loader)):

        X,y = X.cuda(), y.cuda()
        #data_time.update(time.time() - end)

        # # perturb
        X_em = X.clone()
        adlo=torch.permute(( (epsilon/interpol)*torch.arange(-interpol,interpol+1))*torch.ones(X_em[0].shape)[:,:,:,None],(3,0,1,2))
        X_em_top=X_em[:,None,:,:,:]+adlo.cuda()
        #print(X_em.shape[0])
        #input("tix")
        for ii in range(X_em.shape[0]):

            outs=model(X_em_top[ii,:,:,:,:])

            #one constraint per label!=gt
            """
            A_ub=-np.ones((len(outs[0]) - 1, np.prod(X_em[0].shape) + 1))
            b_ub=np.zeros(len(outs[0]) - 1)
            outs=outs-outs[:,y[ii]][:,None]
            #
            labind=0
            for idx in range(len(outs[0])):
                if idx==y[ii].cpu():
                    continue
                coefs=np.linalg.lstsq(torch.cat([X_em_top[ii,:,:,:,:].view(2*interpol+1,-1), torch.ones((2*interpol+1,1)).cuda()],dim=1).cpu().detach().numpy(), outs[:,idx].cpu().detach().numpy())[0]#constraints
                b_ub[labind]=-coefs[-1]
                A_ub[labind,1:]=coefs[:-1]
                labind=labind+1
            """
            #one constraint for any label!=gt
            #n constraints = piecewise linear segments
            A_ub=-np.ones((2*interpol, np.prod(X_em[0].shape) + 1))
            slopesx=X_em_top[ii,1:,:,:,:]-X_em_top[ii,:-1,:,:,:]
            #input(slopesx.shape)
            #z=timer()
            idx=[i for i in range(len(outs[0]))]
            idx.remove(y[ii].cpu().detach().numpy())
            #input(idx)
            #z=timer()-z
            #a=timer()
            outz=torch.sum(outs[:,idx], dim=1)
            outz=outz-outs[:,y[ii]]
            slopesy=outz[1:]-outz[:-1]
            #input(slopesy.shape)

            slopes=(slopesy)[:,None,None,None]*(1/slopesx)
            #print(slopes.shape)
            #input(A_ub.shape)
            #print(slopesx[0,0,0,0])
            #print(slopesy[0])
            #input(slopes[0,0,0,0])
            intercepts=outz[:-1] - torch.diag(torch.matmul(slopes.view(2*interpol,-1), X_em_top[ii,:-1,:,:,:].reshape(-1,2*interpol)))



            #coefs=np.linalg.lstsq(torch.cat([X_em_top[ii,:,:,:,:].view(2*interpol+1,-1), torch.ones((2*interpol+1,1)).cuda()],dim=1).cpu().detach().numpy(), outz.cpu().detach().numpy())[0]#constraints
            #input("SCREAAAAAAAAAM")
            b_ub=-intercepts.cpu().detach().numpy()
            A_ub[:,1:]=slopes.view(2*interpol,-1).cpu().detach().numpy()
            #print(np.min(A_ub))
            #print(np.max(A_ub))
            #if (np.max(A_ub)==np.inf):
            #    input(A_ub)
            tbounds=[(None,None)]
            xbounds=[(lb,ub) for lb,ub in zip(X_em_top[ii,0,:,:,:].flatten().cpu().detach().numpy(), X_em_top[ii,-1,:,:,:].flatten().cpu().detach().numpy())]

            c=[1] + [0 for i in range(len(xbounds))]


            result = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=tbounds+xbounds)

            #print([z, timer()-a])
            #input(result.x.shape)
            t, Xemj = result.x[0], result.x[1:]
            X_em[ii,:,:,:]=torch.from_numpy(Xemj).cuda().view(X_em[ii,:,:,:].shape)

        # adjust to be within [-epsilon, epsilon]
        eta = torch.clamp(X_em - X, -epsilon, epsilon)
        X_em = X + eta
        X_em = torch.clamp(X_em, 0, 1)

        out = model(X)
        ce = nn.CrossEntropyLoss()(out, y)
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

        pout = model(X_em)
        pce = nn.CrossEntropyLoss()(pout, y)
        perr = (pout.data.max(1)[1] != y).float().sum()  / X.size(0)

        loss=alpha*ce + (1-alpha)*pce
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss=epoch_loss+ce.item()
        epoch_ploss=epoch_ploss+pce.item()
        epoch_err, epoch_perr = epoch_err+err*X.size(0), epoch_perr+perr*X.size(0)
        nsamp=nsamp+X.size(0)

    return epoch_loss/nsamp, epoch_err/nsamp, epoch_ploss/nsamp, epoch_perr/nsamp

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
