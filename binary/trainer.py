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
import matplotlib.pyplot as plt
#import imageio
from PIL import Image
#from scipy.spatial import ConvexHull
#import thdecomp as th
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN, KMeans

def _pgd(model, X, y, epsilon, niters=40, alpha=0.01, inbounds=[0.0,1.0]):
    out = model(X).squeeze(1)
    ce = nn.BCEWithLogitsLoss()(out, y.float())
    err = ((out.data>0).float() != y.data).float().sum()  / X.size(0)

    X_pgd = Variable(X.data, requires_grad=True)
    for i in range(niters):
        opt = optim.Adam([X_pgd], lr=1e-3)
        opt.zero_grad()
        loss = nn.BCEWithLogitsLoss()(model(X_pgd).squeeze(1), y.float())
        loss.backward()
        eta = alpha*X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)

        # adjust to be within [-epsilon, epsilon]
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd.data = torch.clamp(X_pgd.data, inbounds[0], inbounds[1])

    err_pgd = ((model(X_pgd).squeeze(1).data>0).float() != y.data).float().sum() / X.size(0)
    return err, err_pgd

def _lin(model, X, y, epsilon, inbounds=[0.0,1.0]):
    #input("hey")
    out = model(X).squeeze(1)
    ce = nn.BCEWithLogitsLoss()(out, y.float())
    err = ((out.data>0).float() != y.data).float().sum()  / X.size(0)

    #X_pgd = Variable(X.data, requires_grad=True)
    eta = model[-1].weight.detach().sign().view(28,28)*epsilon
    delta=-(2*y.float()[:,None,None,None]-1)*eta
    #input(delta.shape)
    #input(X.shape)
    #print(X_pgd.view(X_pgd.size(0), -1).shape)
    #input(eta.shape)#alpha*X_pgd.grad.data.sign()
    #X_pgd = Variable(X_pgd.data.view(X_pgd.data.size(0), -1) + eta, requires_grad=True)
    #input(X_pgd.shape)
        # adjust to be within [-epsilon, epsilon]
    #eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
    X_pgd = Variable(X.data + delta, requires_grad=True)
    X_pgd.data = torch.clamp(X_pgd.data, inbounds[0], inbounds[1])

    err_pgd = ((model(X_pgd).squeeze(1).data>0).float() != y.data).float().sum() / X.size(0)
    return err, err_pgd

def pgd(loader, model, epsilon, niters=100, alpha=0.01, verbose=False,
        robust=False):
    return attack(loader, model, epsilon, verbose=verbose, atk=_pgd,
                  robust=robust)

def train_madry(loader, model, epsilon, opt, epoch, steps=40, stepsize=0.01, inbounds=[0.0,1.0]):#for CIFAR, default is steps=7, stepsize=2
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
            X_pgd.data = torch.clamp(X_pgd.data, inbounds[0], inbounds[1])

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

def evaluate_madry(loader, model, epsilon, epoch, steps=40, stepsize=0.01, inbounds=[0.0,1.0]):
    model.eval()
    epoch_loss, epoch_err = 0,0
    epoch_perr = 0
    nsamp=0
    for i, (X,y) in tqdm(enumerate(loader)):
        X,y = X.cuda(), y.cuda()
        out = model(Variable(X)).squeeze(1)
        #input(out)
        ce = nn.BCEWithLogitsLoss()(out, Variable(y).float())
        err = ((out.data>0).float() != y).float().sum()  / X.size(0)
        # # perturb
        _, perr = _pgd(model, Variable(X), Variable(y), epsilon=epsilon, niters=steps, alpha=stepsize, inbounds=inbounds)

        # measure accuracy and record loss
        epoch_loss=epoch_loss+ce.item()
        #epoch_err = epoch_err+err
        #epoch_perr = epoch_perr+perr
        epoch_err, epoch_perr = epoch_err+err*X.size(0), epoch_perr+perr*X.size(0)
        nsamp=nsamp+X.size(0)
    return epoch_loss/nsamp, epoch_err/nsamp, epoch_perr/nsamp

def evaluate_lin(loader, model, epsilon, epoch, inbounds=[0.0,1.0]):
    model.eval()
    epoch_loss, epoch_err = 0,0
    epoch_perr = 0
    nsamp=0
    for i, (X,y) in tqdm(enumerate(loader)):
        X,y = X.cuda(), y.cuda()
        out = model(Variable(X)).squeeze(1)
        ce = nn.BCEWithLogitsLoss()(out, Variable(y).float())
        err = ((out.data>0).float() != y).float().sum()  / X.size(0)
        # # perturb
        _, perr = _lin(model, Variable(X), Variable(y), epsilon=epsilon, inbounds=inbounds)

        # measure accuracy and record loss
        epoch_loss=epoch_loss+ce.item()
        #epoch_err = epoch_err+err
        #epoch_perr = epoch_perr+perr
        epoch_err, epoch_perr = epoch_err+err*X.size(0), epoch_perr+perr*X.size(0)
        nsamp=nsamp+X.size(0)
    return epoch_loss/nsamp, epoch_err/nsamp, epoch_perr/nsamp


def train_clean(loader, model, opt, epoch):#for CIFAR, default is steps=7, stepsize=2
    model.train()
    epoch_loss, epoch_err = 0,0
    nsamp=0
    for i, (X,y) in tqdm(enumerate(loader)):
        X,y = X.cuda(), y.cuda()

        #input(X.shape)
        out = model(Variable(X)).squeeze(1)
        ce = nn.BCEWithLogitsLoss()(out,y.float())
        err = ((out.data>0).float() != y).float().sum()  / X.size(0)

        opt.zero_grad()
        ce.backward()
        opt.step()

        epoch_loss=epoch_loss+ce.item()
        epoch_err = epoch_err+err*X.size(0)
        nsamp=nsamp+X.size(0)
        #input([epoch_loss/nsamp, epoch_err/nsamp, epoch_ploss/nsamp, epoch_perr/nsamp])

    return epoch_loss/nsamp, epoch_err/nsamp

def train_pwl(loader, model, epsilon, opt, epoch, alpha=0.95, interpol=10, inbounds=[0.0,1.0], rowling=500, segs=1, precomp=None):
    model.train()
    #end = time.time()
    epoch_loss, epoch_err = 0,0
    epoch_ploss, epoch_perr = 0,0
    nsamp=0
    attacks=[]
    tgts=[]
    for i, (X,y) in tqdm(enumerate(loader)):
        X,y = X.cuda(), y.cuda()
        X_em = X.clone()
        size_c, size_x, size_y = X_em.shape[1:]
        if precomp==None:
            for ii in range(X_em.shape[0]):
                #random sampling
                if rowling>0:
                    #samp=np.random.randint(-interpol,interpol,size_c*size_x*size_y*rowling).reshape(-1,size_c,size_x,size_y)
                    samp=np.random.uniform(-epsilon,epsilon,size_x*size_y*rowling).reshape(-1,1,size_x,size_y)
                #brute-force (only on the boundary)
                else:
                    samp=np.array([np.ones((size_c, size_x, size_y))*x*epsilon/interpol for x in range(-interpol, interpol+1)]).reshape(-1,size_c,size_x,size_y).reshape(-1,imsz,imsz).astype(np.float32)

                samp=(np.unique(samp,axis=0)).astype(np.float32)
                #input(samp.shape)
                samp=np.tile(samp, (1,size_c,1,1))
                rowling=samp.shape[0]

                samp=torch.from_numpy(samp).cuda()
                #input(samp.shape)
                sm=(samp+X_em[ii].squeeze(0))#.unsqueeze(1)
                #input(sm.shape)

                outs=model(sm)
                outsog=model(X_em[ii,:,:,:].unsqueeze(0))

                #one constraint for best label!=gt
                SM=sm.detach().cpu().numpy().reshape(-1,size_c*size_x*size_y)

                A_ub=-np.ones((segs, np.prod(X_em[0].shape) + 1))
                b_ub=np.zeros(segs)
                idx=[i for i in range(len(outs[0]))]
                outsel=outsog.clone()
                outsel[0,y[ii]]=-np.inf

                outz, idx =torch.max(outsel[0,idx].unsqueeze(0), dim=1)
                outz=-(outs[:,idx].flatten()-outs[:,y[ii]])

                outz=outz.detach().cpu().numpy()
                for disp in range(segs):
                    pwlmod=LinearRegression()
                    pwlmod.fit(SM[int(rowling/segs)*disp:int(rowling/segs)*(disp+1)+1], outz[int(rowling/segs)*disp:int(rowling/segs)*(disp+1)+1])
                    A_ub[disp,1:]=pwlmod.coef_
                    b_ub[disp]=-pwlmod.intercept_
                tbounds=[(None,None)]
                #xbounds=[(lb,ub) for lb,ub in zip(X_em_top[ii,0,:,:,:].flatten().cpu().detach().numpy(), X_em_top[ii,-1,:,:,:].flatten().cpu().detach().numpy())]
                xbounds=[(lb,ub) for lb,ub in zip(X_em[ii,:,:,:].flatten().cpu().detach().numpy() - epsilon, X_em[ii,:,:,:].flatten().cpu().detach().numpy() + epsilon)]

                c=[1] + [0 for i in range(len(xbounds))]


                result = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=tbounds+xbounds)

                if result.success:
                    t, Xemj = result.x[0], result.x[1:]
                    X_em[ii,:,:,:]=torch.from_numpy(Xemj).cuda().view(X_em[ii,:,:,:].shape)
            X_em = torch.clamp(X_em, inbounds[0], inbounds[1])
            attacks.append(X_em)
            tgts.append(y)
            yem=y
        else:
            X_em=precomp[0][i]
            yem=precomp[1][i]

        out = model(X)
        ce = nn.CrossEntropyLoss()(out, y)
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

        pout = model(X_em)
        pce = nn.CrossEntropyLoss()(pout, yem)
        perr = (pout.data.max(1)[1] != yem).float().sum()  / X.size(0)

        loss=alpha*ce + (1-alpha)*pce
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss=epoch_loss+ce.item()
        epoch_ploss=epoch_ploss+pce.item()
        epoch_err, epoch_perr = epoch_err+err*X.size(0), epoch_perr+perr*X.size(0)
        nsamp=nsamp+X.size(0)

    return epoch_loss/nsamp, epoch_err/nsamp, epoch_ploss/nsamp, epoch_perr/nsamp, [attacks, tgts]

def evaluate_pwl(loader, model, epsilon, epoch, interpol=10, inbounds=[0.0, 1.0], rowling=500,segs=1, use_clusters=False, approximate=True):
    model.eval()
    #end = time.time()
    epoch_loss, epoch_err = 0,0
    epoch_ploss, epoch_perr = 0,0
    nsamp=0
    for i, (X,y) in tqdm(enumerate(loader)):
        X,y = X.cuda(), y.cuda()
        X_em = X.clone()
        size_c, size_x, size_y = X_em.shape[1:]
        #input(X_em.shape[1:])
        #print([size_c, size_x, size_y])
        #input(y.shape)
        for ii in range(X_em.shape[0]):
            #random sampling
            if rowling>0:
                #samp=np.random.randint(-interpol,interpol,size_c*size_x*size_y*rowling).reshape(-1,size_c,size_x,size_y)
                samp=np.random.uniform(-epsilon,epsilon,size_x*size_y*rowling).reshape(-1,1,size_x,size_y)
            #brute-force (only on the boundary)
            else:
                samp=np.array([np.ones((size_c, size_x, size_y))*x*epsilon/interpol for x in range(-interpol, interpol+1)]).reshape(-1,size_c,size_x,size_y).reshape(-1,imsz,imsz).astype(np.float32)

            #samp=(np.unique(samp,axis=0)*epsilon/interpol).astype(np.float32)
            samp=(np.unique(samp,axis=0)).astype(np.float32)
            #input(samp.shape)
            samp=np.tile(samp, (1,size_c,1,1))
            #input(samp.shape)
            rowling=samp.shape[0]
            og=samp.shape
            #clustering:
            if use_clusters:
                clustering=KMeans(n_clusters=segs).fit(samp.reshape(rowling,-1))
                kmlabs=[x for x in np.unique(clustering.labels_) if x!=-1]
            else:
                kmlabs=[i for i in range(segs)]
            #input(kmlabs)
            segs=len(kmlabs)

            samp=torch.from_numpy(samp).cuda()
            #input(samp.shape)
            sm=(samp+X_em[ii].squeeze(0))#.unsqueeze(1)
            #input(sm.shape)

            outs=model(sm)
            outsog=model(X_em[ii,:,:,:].unsqueeze(0))
            #input(outs.shape)
            #input(outsog.shape)

            #one constraint for best label!=gt
            SM=sm.detach().cpu().numpy().reshape(-1,size_c*size_x*size_y)

            A_ub=-np.ones((segs, np.prod(X_em[0].shape) + 1))
            b_ub=np.zeros(segs)
            idx=[i for i in range(len(outs[0]))]
            outsel=outsog.clone()
            #outsel[0,y[ii]]=-np.inf


            if approximate:
                if not y[ii]:
                    outz=-torch.abs(outs)
                else:
                    outz=torch.abs(outs)
                #print(y[ii])
                #input(outz)


                outz=outz.detach().cpu().numpy()
                for disp, kml in enumerate(kmlabs):
                    pwlmod=LinearRegression()
                    if not use_clusters:
                        pwlmod.fit(SM[int(rowling/segs)*disp:int(rowling/segs)*(disp+1)+1], outz[int(rowling/segs)*disp:int(rowling/segs)*(disp+1)+1])
                    else:
                        pwlmod.fit(SM[np.where(clustering.labels_==kml)], outz[np.where(clustering.labels_==kml)])
                    A_ub[disp,1:]=pwlmod.coef_
                    b_ub[disp]=-pwlmod.intercept_
            else:
                #NO LINEARIZATION NEEDED
                #print(A_ub.shape)
                #print(model[1].weight.shape)
                #input(model[1].bias)
                if not y[ii]:
                    A_ub[0,1:]=-model[1].weight.detach().cpu().numpy()
                    b_ub[0]=model[1].bias.detach().cpu().numpy()
                else:
                    A_ub[0,1:]=model[1].weight.detach().cpu().numpy()
                    b_ub[0]=-model[1].bias.detach().cpu().numpy()

            tbounds=[(None,None)]
            #xbounds=[(lb,ub) for lb,ub in zip(X_em_top[ii,0,:,:,:].flatten().cpu().detach().numpy(), X_em_top[ii,-1,:,:,:].flatten().cpu().detach().numpy())]
            xbounds=[(lb,ub) for lb,ub in zip(X_em[ii,:,:,:].flatten().cpu().detach().numpy() - epsilon, X_em[ii,:,:,:].flatten().cpu().detach().numpy() + epsilon)]

            c=[1] + [0 for i in range(len(xbounds))]


            result = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=tbounds+xbounds)

            if result.success:
                t, Xemj = result.x[0], result.x[1:]
                X_em[ii,:,:,:]=torch.from_numpy(Xemj).cuda().view(X_em[ii,:,:,:].shape)

        X_em = torch.clamp(X_em, inbounds[0], inbounds[1])
        out = model(X)
        #print(out.shape)
        #input(y.shape)
        ##print(type(out))
        #input(type(y))
        ce = nn.BCEWithLogitsLoss()(out.squeeze(1), y.float())
        #err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)
        #print(out.data>0)
        #print(y)
        ##print(y.shape)
        #print(out.shape)
        #print(X_em.size(0))
        #print(X.size(0))
        #input(((out.squeeze(1).data>0).float() != y).shape)

        err = ((out.squeeze(1).data>0).float() != y).float().sum()  / X.size(0)
        #input("Once")
        pout = model(X_em)
        pce = nn.BCEWithLogitsLoss()(pout.squeeze(1), y.float())
        #perr = (pout.data.max(1)[1] != y).float().sum()  / X.size(0)
        perr = ((pout.squeeze(1).data>0).float() != y).float().sum()  / X.size(0)
        #print(err)
        #input(perr)


        epoch_loss=epoch_loss+ce.item()
        epoch_ploss=epoch_ploss+pce.item()
        epoch_err, epoch_perr = epoch_err+err*X.size(0), epoch_perr+perr*X.size(0)
        nsamp=nsamp+X.size(0)

    return epoch_loss/nsamp, epoch_err/nsamp, epoch_perr/nsamp
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
