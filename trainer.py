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
import imageio
from PIL import Image
from scipy.spatial import ConvexHull
import thdecomp as th
from sklearn.linear_model import LinearRegression, Ridge, Lasso

def _pgd(model, X, y, epsilon, niters=40, alpha=0.01, inbounds=[0.0,1.0]):
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
        X_pgd.data = torch.clamp(X_pgd.data, inbounds[0], inbounds[1])

    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum() / X.size(0)
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
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)
        # # perturb
        _, perr = _pgd(model, Variable(X), Variable(y), epsilon=epsilon, niters=steps, alpha=stepsize, inbounds=inbounds)

        # measure accuracy and record loss
        epoch_loss=epoch_loss+ce.item()
        #epoch_err = epoch_err+err
        #epoch_perr = epoch_perr+perr
        epoch_err, epoch_perr = epoch_err+err*X.size(0), epoch_perr+perr*X.size(0)
        nsamp=nsamp+X.size(0)
    return epoch_loss/nsamp, epoch_err/nsamp, epoch_perr/nsamp



def train_em(loader, model, epsilon, opt, epoch, alpha=0.95, interpol=10, inbounds=[0.0,1.0]):
    model.train()
    #end = time.time()
    epoch_loss, epoch_err = 0,0
    epoch_ploss, epoch_perr = 0,0
    nsamp=0
    for i, (X,y) in tqdm(enumerate(loader)):

        X,y = X.cuda(), y.cuda()
        X_em = X.clone()
        adlo=torch.permute(( (epsilon/interpol) *torch.arange(-interpol,interpol+1))*torch.ones(X_em[0].shape)[:,:,:,None],(3,0,1,2))
        X_em_top=X_em[:,None,:,:,:]+adlo.cuda()

        for ii in range(X_em.shape[0]):

            outs=model(X_em_top[ii,:,:,:,:])
            #one constraint per label!=gt
            #"""
            A_ub=-np.ones((len(outs[0]) - 1, np.prod(X_em[0].shape) + 1))
            b_ub=np.zeros(len(outs[0]) - 1)

            outs=-(outs-outs[:,y[ii]][:,None])#-outs

            #input(outs)
            labind=0
            for idx in range(len(outs[0])):
                if idx==y[ii].cpu():
                    continue
                coefs=np.linalg.lstsq(torch.cat([X_em_top[ii,:,:,:,:].view(2*interpol+1,-1), torch.ones((2*interpol+1,1)).cuda()],dim=1).cpu().detach().numpy(),
                                        outs[:,idx].cpu().detach().numpy(), rcond=-1)[0]#constraints
                b_ub[labind]=-coefs[-1]
                A_ub[labind,1:]=coefs[:-1]
                labind=labind+1
            #"""
            #one constraint for any label!=gt
            """
            A_ub=-np.ones((1, np.prod(X_em[0].shape) + 1))
            #z=timer()
            idx=[i for i in range(len(outs[0]))]
            idx.remove(y[ii].cpu().detach().numpy())
            #input(idx)
            #z=timer()-z
            #a=timer()
            outz=torch.sum(outs[:,idx], dim=1)
            outz=outz-outs[:,y[ii]]
            #input(outz.shape)


            coefs=np.linalg.lstsq(torch.cat([X_em_top[ii,:,:,:,:].view(2*interpol+1,-1), torch.ones((2*interpol+1,1)).cuda()],dim=1).cpu().detach().numpy(), outz.cpu().detach().numpy())[0]#constraints
            #input(coefs.shape)

            b_ub=-coefs[-1]
            A_ub[0,1:]=coefs[:-1]
            #input(A_ub)
            """

            tbounds=[(0,None)]
            xbounds=[(lb,ub) for lb,ub in zip(X_em_top[ii,0,:,:,:].flatten().cpu().detach().numpy(), X_em_top[ii,-1,:,:,:].flatten().cpu().detach().numpy())]

            c=[1] + [0 for i in range(len(xbounds))]


            result = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=tbounds+xbounds)

            #print([z, timer()-a])
            #input(result.x.shape)
            input(result.success)
            if result.success:
                t, Xemj = result.x[0], result.x[1:]
                X_em[ii,:,:,:]=torch.from_numpy(Xemj).cuda().view(X_em[ii,:,:,:].shape)

        X_em = torch.clamp(X_em, inbounds[0], inbounds[1])

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

def evaluate_em(loader, model, epsilon, epoch, interpol=10, inbounds=[0.0,1.0], rowling=500):
    model.eval()
    epoch_loss, epoch_err = 0,0
    epoch_ploss, epoch_perr = 0,0
    nsamp=0
    for i, (X,y) in tqdm(enumerate(loader)):
        X,y = X.cuda(), y.cuda()

        X_em = X.clone()
        #adlo=torch.permute(( (epsilon/interpol) *torch.arange(-interpol,interpol+1))*torch.ones(X_em[0].shape)[:,:,:,None],(3,0,1,2))
        #X_em_top=X_em[:,None,:,:,:]+adlo.cuda()

        #input(X_em.shape)
        #input(rowling)
        for ii in range(X_em.shape[0]):
            if False:
                X_emnp=X_em[ii,:,:,:].detach().cpu().numpy()
                Xemj=X_emnp+np.random.uniform(-epsilon, -epsilon, size=X_emnp.size).reshape(X_emnp.shape)
                X_em[ii,:,:,:]=torch.from_numpy(Xemj).cuda().view(X_em[ii,:,:,:].shape)
            else:
                #patchlike
                """
                sz=2
                _=np.random.randint(0,27-sz,2)
                a, b = _[0], _[1]
                meshes=[np.linspace(-epsilon,epsilon,2*interpol+1) for i in range(sz*sz)]
                gridall=np.array(np.meshgrid(*tuple(map(tuple,np.array(meshes))))).T.reshape(-1,sz,sz)
                samp=np.zeros((len(gridall),28,28))
                samp[:,a:a+sz,b:b+sz]=gridall
                samp=samp.astype(np.float32)
                """

                #random sampling
                samp=np.random.randint(-interpol,interpol,28*28*rowling).reshape(-1,28,28)
                samp=(np.unique(samp,axis=0)*epsilon/interpol).astype(np.float32)

                rowling=samp.shape[0]
                samp=torch.from_numpy(samp).cuda()


                sm=(samp+X_em[ii].squeeze(0)).unsqueeze(1)

                #I=X_em[ii,:,:,:].cpu().numpy().squeeze(0)
                #lo, hi=I.min(), I.max()
                #print(lo)
                #input(hi)
                #I8 = (((I - lo) / (hi - lo)) * 255.9).astype(np.uint8)
                #img = Image.fromarray(I8)
                #img.save("clean_mz.png")
                #for xi in range(rowling):

                    #klin=X_em[ii,:,:,:].cpu().numpy().squeeze(0) + samp[xi,:,:].cpu().numpy()
                    #klin1=X_em[ii,:,:,:].cpu().numpy().squeeze(0) + np.min(samp[xi,:,:])
                    #klin2=X_em[ii,:,:,:].cpu().numpy().squeeze(0) + np.max(samp[xi,:,:])
                    #dirt=X_em[0,:,:].cpu().numpy().squeeze(0)
                    #print(np.max(klin))
                    #input(np.min(klin))

                    #I=klin
                    #I8 = (((I - lo) / (hi - lo)) * 255.9).astype(np.uint8)
                    #img = Image.fromarray(I8)
                    #img.save("clean_{}.png".format(xi))

                    #I=klin1
                    #I8 = (((I - lo) / (hi - lo)) * 255.9).astype(np.uint8)
                    #img = Image.fromarray(I8)
                    #img.save("clean_{}_min.png".format(xi))

                    #I=klin2
                    #I8 = (((I - lo) / (hi - lo)) * 255.9).astype(np.uint8)
                    #img = Image.fromarray(I8)
                    #img.save("clean_{}_max.png".format(xi))
                    #print(np.max(samp[xi]))
                    #input(np.min(samp[xi]))
                    #I=dirt
                    #I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)
                    #img = Image.fromarray(I8)
                    #img.save("dirt.png")
                #input("nestea")



                #outs=model(X_em_top[ii,:,:,:,:])
                outs=model(sm)
                outsog=model(X_em[ii,:,:,:].unsqueeze(0))
                #print(y[ii])
                #print(outsog)
                #input(outs)

                #one constraint per label!=gt
                """
                A_ub=-np.ones((len(outs[0]) - 1, np.prod(X_em[0].shape) + 1))
                b_ub=np.zeros(len(outs[0]) - 1)

                outs=-(outs-outs[:,y[ii]][:,None])#-outs
                #_,indss=torch.sort(outs, dim=1)
                #print(outs)
                #input(indss)
                #input(outs)
                labind=0
                for idx in range(len(outs[0])):
                    if idx==y[ii].cpu():
                        continue
                    coefs=np.linalg.lstsq(torch.cat([sm.view(rowling,-1), torch.ones((rowling,1)).cuda()],dim=1).cpu().detach().numpy(),
                                            outs[:,idx].cpu().detach().numpy(), rcond=-1)[0]#constraints
                    b_ub[labind]=-coefs[-1]
                    A_ub[labind,1:]=coefs[:-1]
                    labind=labind+1


                """
                #one constraint for best label!=gt
                #"""
                A_ub=-np.ones((1, np.prod(X_em[0].shape) + 1))
                idx=[i for i in range(len(outs[0]))]
                idx.remove(y[ii].cpu().detach().numpy())
                #input(outsog.shape)
                #outz, _ =torch.max(outs[interpol+1,idx].unsqueeze(0), dim=1)
                #input(_)
                outz, _ =torch.max(outsog[0,idx].unsqueeze(0), dim=1)
                #input(_)
                outz=-(outs[:,_].flatten()-outs[:,y[ii]])
                coefs=np.linalg.lstsq(torch.cat([sm.view(rowling,-1), torch.ones((rowling,1)).cuda()],dim=1).cpu().detach().numpy(),
                                        outz.cpu().detach().numpy(), rcond=-1)[0]#constraints

                b_ub=-coefs[-1]
                A_ub[0,1:]=coefs[:-1]

                #"""
                #one constraint for any label!=gt
                """
                A_ub=-np.ones((1, np.prod(X_em[0].shape) + 1))
                idx=[i for i in range(len(outs[0]))]
                idx.remove(y[ii].cpu().detach().numpy())
                outz=torch.sum(outs[:,idx], dim=1)
                outz=outz-outs[:,y[ii]]
                #input(outz.shape)


                coefs=np.linalg.lstsq(torch.cat([X_em_top[ii,:,:,:,:].view(2*interpol+1,-1), torch.ones((2*interpol+1,1)).cuda()],dim=1).cpu().detach().numpy(), outz.cpu().detach().numpy())[0]#constraints
                #input(coefs.shape)

                b_ub=-coefs[-1]
                A_ub[0,1:]=coefs[:-1]
                """
                tbounds=[(None,None)]
                xbounds=[(lb,ub) for lb,ub in zip(X_em[ii,:,:,:].flatten().cpu().detach().numpy() - epsilon, X_em[ii,:,:,:].flatten().cpu().detach().numpy() + epsilon)]

                c=[1] + [0 for i in range(len(xbounds))]

                """
                protrakt=X_em_top[ii,:,:,:,:].view(2*interpol+1,-1).cpu().numpy()@A_ub[:,1:].T
                for jju in range(len(b_ub)):
                    idx=jju+(jju>=y[ii].cpu())
                    input(idx)
                    print(outs[:,idx])
                    print(-b_ub[jju] + protrakt[:,jju])
                input()
                """
                result = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=tbounds+xbounds)

                #print([z, timer()-a])
                #input(result.x.shape)
                #print(result.x[0])
                #print(result.x[1:])
                #print(result.success)
                if result.success:
                    t, Xemj = result.x[0], result.x[1:]
                    #outsog=model(X_em[ii,:,:,:].unsqueeze(0))
                    X_em[ii,:,:,:]=torch.from_numpy(Xemj).cuda().view(X_em[ii,:,:,:].shape)
                    #print(X_em[ii,:,:,:].shape)
                    #outsem=model(X_em[ii,:,:,:].unsqueeze(0))
                    #_,indss=torch.sort(outsog, dim=1)
                    #print(indss[0][0])
                    #_,indss=torch.sort(outsem, dim=1)
                    #print(indss[0][0])
                    #print(outsog-outsog[0,y[ii]])
                    #print(outsem-outsem[0,y[ii]])
                    #input(t)

        # adjust to be within [-epsilon, epsilon]
        #eta = torch.clamp(X_em - X, -epsilon, epsilon)
        #X_em = X + eta
        X_em = torch.clamp(X_em, inbounds[0], inbounds[1])
        #print(model(X_em))
        out = model(X)
        ce = nn.CrossEntropyLoss()(out, y)
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

        pout = model(X_em)
        #input(pout)
        pce = nn.CrossEntropyLoss()(pout, y)
        perr = (pout.data.max(1)[1] != y).float().sum()  / X.size(0)
        #print(err)
        #print(perr)
        #print(X_em[0].shape)
        #input(X[0].shape)
        """
        klin=X[0,:,:].cpu().numpy().squeeze(0)
        dirt=X_em[0,:,:].cpu().numpy().squeeze(0)
        from PIL import Image
        I=klin
        I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)
        img = Image.fromarray(I8)
        img.save("clean.png")
        I=dirt
        I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)
        img = Image.fromarray(I8)
        img.save("dirt.png")
        input("nestea")
        """

        epoch_loss=epoch_loss+ce.item()
        epoch_ploss=epoch_ploss+pce.item()
        epoch_err, epoch_perr = epoch_err+err*X.size(0), epoch_perr+perr*X.size(0)
        nsamp=nsamp+X.size(0)
    return epoch_loss/nsamp, epoch_err/nsamp, epoch_perr/nsamp

def train_clean(loader, model, opt, epoch):#for CIFAR, default is steps=7, stepsize=2
    model.train()
    epoch_loss, epoch_err = 0,0
    nsamp=0
    for i, (X,y) in tqdm(enumerate(loader)):
        X,y = X.cuda(), y.cuda()


        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

        opt.zero_grad()
        ce.backward()
        opt.step()

        epoch_loss=epoch_loss+ce.item()
        epoch_err = epoch_err+err*X.size(0)
        nsamp=nsamp+X.size(0)
        #input([epoch_loss/nsamp, epoch_err/nsamp, epoch_ploss/nsamp, epoch_perr/nsamp])

    return epoch_loss/nsamp, epoch_err/nsamp

def train_pwl(loader, model, epsilon, opt, epoch, alpha=0.95, interpol=10, inbounds=[0.0,1.0]):
    model.train()
    #end = time.time()
    epoch_loss, epoch_err = 0,0
    epoch_ploss, epoch_perr = 0,0
    nsamp=0
    for i, (X,y) in tqdm(enumerate(loader)):

        X,y = X.cuda(), y.cuda()
        X_em = X.clone()
        adlo=torch.permute(( (epsilon/interpol)*torch.arange(-interpol,interpol+1))*torch.ones(X_em[0].shape)[:,:,:,None],(3,0,1,2))
        X_em_top=X_em[:,None,:,:,:]+adlo.cuda()
        for ii in range(X_em.shape[0]):

            outs=model(X_em_top[ii,:,:,:,:])

            #one constraint per label!=gt
            #"""
            A_ub=-np.ones((2*interpol*(len(outs[0]) - 1), np.prod(X_em[0].shape) + 1))
            b_ub=np.zeros(2*interpol*(len(outs[0]) - 1))
            outs=-(outs-outs[:,y[ii]][:,None])
            slopesx=X_em_top[ii,1:,:,:,:]-X_em_top[ii,:-1,:,:,:]
            #input(outs.shape)
            labind=0
            for idx in range(len(outs[0])):
                if idx==y[ii].cpu():
                    continue
                #print(outs[:,idx])
                slopesy=outs[1:,idx]-outs[:-1, idx]
                slopes=(slopesy)[:,None,None,None]*(1/slopesx)
                intercepts=outs[:-1,idx] - torch.diag(torch.matmul(slopes.view(2*interpol,-1), X_em_top[ii,:-1,:,:,:].reshape(-1,2*interpol)))
                b_ub[2*interpol*labind:2*interpol*(labind+1)]=-intercepts.cpu().detach().numpy()
                A_ub[2*interpol*labind:2*interpol*(labind+1),1:]=slopes.view(2*interpol,-1).cpu().detach().numpy()
            #"""
            #one constraint for any label!=gt
            #n constraints = piecewise linear segments
            """
            A_ub=-np.ones((2*interpol, np.prod(X_em[0].shape) + 1))
            slopesx=X_em_top[ii,1:,:,:,:]-X_em_top[ii,:-1,:,:,:]
            idx=[i for i in range(len(outs[0]))]
            idx.remove(y[ii].cpu().detach().numpy())
            outz=torch.sum(outs[:,idx], dim=1)
            outz=outz-outs[:,y[ii]]
            slopesy=outz[1:]-outz[:-1]

            slopes=(slopesy)[:,None,None,None]*(1/slopesx)
            intercepts=outz[:-1] - torch.diag(torch.matmul(slopes.view(2*interpol,-1), X_em_top[ii,:-1,:,:,:].reshape(-1,2*interpol)))
            b_ub=-intercepts.cpu().detach().numpy()
            A_ub[:,1:]=slopes.view(2*interpol,-1).cpu().detach().numpy()
            """
            tbounds=[(None,None)]
            xbounds=[(lb,ub) for lb,ub in zip(X_em_top[ii,0,:,:,:].flatten().cpu().detach().numpy(), X_em_top[ii,-1,:,:,:].flatten().cpu().detach().numpy())]

            c=[1] + [0 for i in range(len(xbounds))]


            result = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=tbounds+xbounds)

            #print([z, timer()-a])
            #input(result.x.shape)
            if result.success:
                t, Xemj = result.x[0], result.x[1:]
                X_em[ii,:,:,:]=torch.from_numpy(Xemj).cuda().view(X_em[ii,:,:,:].shape)

        # adjust to be within [-epsilon, epsilon]
        #eta = torch.clamp(X_em - X, -epsilon, epsilon)
        #X_em = X + eta
        X_em = torch.clamp(X_em, inbounds[0], inbounds[1])

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

def evaluate_pwl(loader, model, epsilon, epoch, interpol=10, inbounds=[0.0, 1.0], rowling=500,segs=1):
    model.eval()
    #end = time.time()
    epoch_loss, epoch_err = 0,0
    epoch_ploss, epoch_perr = 0,0
    nsamp=0
    imsz=32
    for i, (X,y) in tqdm(enumerate(loader)):

        X,y = X.cuda(), y.cuda()
        #data_time.update(time.time() - end)

        # # perturb
        X_em = X.clone()
        #input(X_em.shape)
        #print(np.min(X_em.cpu().detach().numpy()))
        #X_em = torch.clamp(X_em, 0, 1)
        #input(np.min(X_em.cpu().detach().numpy()))
        #adlo=torch.permute(( (epsilon/interpol)*torch.arange(-interpol,interpol+1))*torch.ones(X_em[0].shape)[:,:,:,None],(3,0,1,2))
        #X_em_top=X_em[:,None,:,:,:]+adlo.cuda()
        #print(X_em.shape[0])
        #input("tix")
        import warnings
        warnings.filterwarnings("ignore")
        for ii in range(X_em.shape[0]):
            #random sampling
            if rowling>0:
                samp=np.random.randint(-interpol,interpol,imsz*imsz*rowling).reshape(-1,imsz,imsz)
            #brute-force (only on the boundary)
            else:
                samp=np.array([np.ones((imsz,imsz))*x*epsilon/interpol for x in range(-interpol, interpol+1)]).reshape(-1,imsz,imsz).astype(np.float32)

            samp=(np.unique(samp,axis=0)*epsilon/interpol).astype(np.float32)
            rowling=samp.shape[0]

            samp=np.tile(samp, 3).reshape(rowling, 3, imsz, imsz)

            samp=torch.from_numpy(samp).cuda()
            #input(samp.shape)
            sm=(samp+X_em[ii].squeeze(0))#.unsqueeze(1)
            #input(sm.shape)

            outs=model(sm)
            outsog=model(X_em[ii,:,:,:].unsqueeze(0))
            #input(outs.shape)
            #input(outsog.shape)

            #one constraint per label!=gt
            """
            A_ub=-np.ones((2*interpol*(len(outs[0]) - 1), np.prod(X_em[0].shape) + 1))
            b_ub=np.zeros(2*interpol*(len(outs[0]) - 1))
            outs=-(outs-outs[:,y[ii]][:,None])
            slopesx=X_em_top[ii,1:,:,:,:]-X_em_top[ii,:-1,:,:,:]
            #input(outs.shape)
            labind=0
            for idx in range(len(outs[0])):
                if idx==y[ii].cpu():
                    continue
                #print(outs[:,idx])
                slopesy=outs[1:,idx]-outs[:-1, idx]
                slopes=(slopesy)[:,None,None,None]*(1/slopesx)
                intercepts=outs[:-1,idx] - torch.diag(torch.matmul(slopes.view(2*interpol,-1), X_em_top[ii,:-1,:,:,:].reshape(-1,2*interpol)))
                b_ub[2*interpol*labind:2*interpol*(labind+1)]=-intercepts.cpu().detach().numpy()
                A_ub[2*interpol*labind:2*interpol*(labind+1),1:]=slopes.view(2*interpol,-1).cpu().detach().numpy()
            """
            #one constraint for best label!=gt
            #"""
            #print("huh")
            SM=sm.detach().cpu().numpy().reshape(-1,3*imsz*imsz)
            #q1,q2,q3=np.percentile(SM, 25, axis=0), np.percentile(SM, 50, axis=0), np.percentile(SM, 75, axis=0)
            #Q=np.hstack([q1.reshape(-1,1), q2.reshape(-1,1), q3.reshape(-1,1)])
            #interval=[list(q) for q in list(Q)]
            A_ub=-np.ones((segs, np.prod(X_em[0].shape) + 1))
            b_ub=np.zeros(segs)
            idx=[i for i in range(len(outs[0]))]
            outsel=outsog.clone()
            outsel[0,y[ii]]=-np.inf
            #print(y[ii])
            #print(outsog)
            #print(outsel)
            #print(idx)
            #idx.remove(y[ii].cpu().detach().numpy())
            #print(idx)
            outz, idx =torch.max(outsel[0,idx].unsqueeze(0), dim=1)
            #input(idx)
            outz=-(outs[:,idx].flatten()-outs[:,y[ii]])
            #input("ISILDUR")

            #slopesx=X_em[ii,1:,:,:,:]-X_em_top[ii,:-1,:,:,:]
            #input(sm.shape)
            #print("huh")
            #SZM=th.multivar_decomp_uniform(SM,[-epsilon/2, epsilon/2]

            #input("ISILDUR")
            #input(SM.shape)

            outz=outz.detach().cpu().numpy()
            #input(outz.shape)
            #pwlmod.fit(SM, outz)
            #A_ub=-np.ones((1, np.prod(X_em[0].shape) + 1))
            #b_ub=0#np.zeros(segs)
            #b_ub=b_ub-pwlmod.intercept_
            #print(pwlmod.coef_[:10])
            #print(pwlmod.coef_[:10:2])
            #input(pwlmod.coef_[1:10:2])
            #print(outz.shape)
            #input(SM.shape)
            for disp in range(segs):
                pwlmod=LinearRegression()
                pwlmod.fit(SM[int(rowling/segs)*disp:int(rowling/segs)*(disp+1)+1], outz[int(rowling/segs)*disp:int(rowling/segs)*(disp+1)+1])
                A_ub[disp,1:]=pwlmod.coef_
                b_ub[disp]=-pwlmod.intercept_
                #coefs=np.linalg.lstsq(np.hstack([SM[int(rowling/segs)*disp:int(rowling/segs)*(disp+1)], np.ones((int(rowling/segs),1))]),
                                        #outz[int(rowling/segs)*disp:int(rowling/segs)*(disp+1)], rcond=-1)[0]
                #A_ub[disp,1:]=coefs[:-1]
                #b_ub[disp] = -coefs[-1]
            #A_ub[:,1:]=pwlmod.coef_#[disp::segs]
            #"""
            #input("ISILDUR")
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

            slopes=(slopesy)[:,None,None,None]*(1/slopesx)
            intercepts=outz[:-1] - torch.diag(torch.matmul(slopes.view(2*interpol,-1), X_em_top[ii,:-1,:,:,:].reshape(-1,2*interpol)))
            b_ub=-intercepts.cpu().detach().numpy()
            A_ub[:,1:]=slopes.view(2*interpol,-1).cpu().detach().numpy()
            """
            tbounds=[(None,None)]
            #xbounds=[(lb,ub) for lb,ub in zip(X_em_top[ii,0,:,:,:].flatten().cpu().detach().numpy(), X_em_top[ii,-1,:,:,:].flatten().cpu().detach().numpy())]
            xbounds=[(lb,ub) for lb,ub in zip(X_em[ii,:,:,:].flatten().cpu().detach().numpy() - epsilon, X_em[ii,:,:,:].flatten().cpu().detach().numpy() + epsilon)]

            c=[1] + [0 for i in range(len(xbounds))]


            result = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=tbounds+xbounds)

            #print([z, timer()-a])
            #input(result.x.shape)
            #print(result.success)
            if result.success:
                t, Xemj = result.x[0], result.x[1:]
                X_em[ii,:,:,:]=torch.from_numpy(Xemj).cuda().view(X_em[ii,:,:,:].shape)
                #outsem=model(X_em[ii,:,:,:].unsqueeze(0))
                #_,indss=torch.sort(outsog, dim=1)
                #print(indss[0][0])
                #_,indss=torch.sort(outsem, dim=1)
                #print(indss[0][0])
                #input(t)

        X_em = torch.clamp(X_em, inbounds[0], inbounds[1])
        out = model(X)
        ce = nn.CrossEntropyLoss()(out, y)
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

        pout = model(X_em)
        pce = nn.CrossEntropyLoss()(pout, y)
        perr = (pout.data.max(1)[1] != y).float().sum()  / X.size(0)
        #print(err)
        #print(perr)


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
