# import waitGPU
# import setGPU
# waitGPU.wait(utilization=20, available_memory=10000, interval=60)
# waitGPU.wait(gpu_ids=[1,3], utilization=20, available_memory=10000, interval=60)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random

from trainer import *
from polytope import *
import math
import numpy as np
import argparse

#args = pblm.argparser(epsilon = 0.0347, starting_epsilon=0.001, batch_size = 50,
#            opt='sgd', lr=0.05)
parser = argparse.ArgumentParser()
parser.add_argument("--save",action='store_true',help="save detection results to txt")
parser.add_argument("--savedir",default='emres/',type=str,help="directory to save eval results")
parser.add_argument("--opt",default='adam',type=str,help="optimizer")
parser.add_argument("--model",default='small',type=str,help="optimizer")
parser.add_argument("--model_path",default='./whatever.pth',type=str,help="optimizer")
parser.add_argument("--dataset",default='cifar',type=str,help="dataset")
parser.add_argument("--prefix",default='robust_model',type=str,help="optimizer")
parser.add_argument("--method",default='madry',type=str,help="optimizer")
parser.add_argument("--lr",default=1e-4,type=float,help="learning rate")
parser.add_argument("--epsilon",default=2/255,type=float,help="")
parser.add_argument("--alpha",default=0.95,type=float,help="")
parser.add_argument("--interpol",default=3,type=int,help="half the # of discrete points for the mesh around an image")
parser.add_argument("--row",default=500,type=int,help="random samples taken from the mesh")
parser.add_argument("--segs",default=1,type=int,help="segments for pwl (use 1 for EMRob)")
parser.add_argument("--use_clusters",action='store_true',help="do clustering on sampled points for linear approximation")
parser.add_argument("--approximate",action='store_true',help="use lienar approximation, not closed form of the linear constraint")

parser.add_argument("--eval",action='store_true',help="evaluate adversarially trained model")
parser.add_argument("--eval_method",default='madry',type=str,help="adversarial attack for eval")


parser.add_argument("--madry_eps",default=2/255,type=float,help="")
parser.add_argument("--madry_steps",default=20,type=int,help="")

parser.add_argument("--epochs",default=1000,type=int,help="")
parser.add_argument("--seed",default=1234,type=int,help="")
parser.add_argument("--batch_size",default=32,type=int,help="training batch size")
parser.add_argument("--test_batch_size",default=32,type=int,help="test batch size")
args = parser.parse_args()

import warnings
warnings.filterwarnings("ignore")

if args.dataset=='cifar':
    inbounds=[-2.1555557,2.64]
    train_loader, _ = cifar_loaders(args.batch_size)
    _, test_loader = cifar_loaders(args.test_batch_size)
    if args.model=='small':
        model = cifar_model().cuda()
    elif args.model == 'large':
        model = cifar_model_large().cuda()
    elif args.model == 'resnet':
        model = model_resnet(cifar=True)
elif args.dataset=='mnist':
    inbounds=[0.0,1.0]
    train_loader, _ = mnist_loaders(args.batch_size)
    _, test_loader = mnist_loaders(args.test_batch_size)
    if args.model=='small':
        model = mnist_model().cuda()
    elif args.model == 'large':
        model = mnist_model_large().cuda()
    elif args.model == 'resnet':
        model = model_resnet(cifar=False)
    elif args.model == 'linear':
        model = mnist_model_linear().cuda()


torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(0)
np.random.seed(0)

sampler_indices = []

if args.opt == 'adam':
    opt = optim.Adam(model.parameters(), lr=args.lr)
else:
    opt = optim.SGD(model.parameters(), lr=1.)
    #raise ValueError("Unknown optimizer")

best_err=1
best_perr, best_loss=None, None
patience=0
if not args.eval:
    for t in range(args.epochs):
        if args.method=='madry':
            loss, err, ploss, perr = train_madry(train_loader, model, args.epsilon, opt, t, steps=args.madry_steps, stepsize=args.madry_eps, inbounds=inbounds)
            #vloss, verr, vperr = evaluate_madry(test_loader, model, args.epsilon, t, steps=20, stepsize=2/255)
        # robust training
        elif args.method == 'pwl':
            if t==0:
                loss, err, ploss, perr, atks = train_pwl(train_loader, model, args.epsilon, opt, t, alpha=args.alpha, interpol=args.interpol, inbounds=inbounds, rowling=args.row, segs=args.segs)
                precomp=atks
            else:
                loss, err, ploss, perr, _ = train_pwl(train_loader, model, args.epsilon, opt, t, alpha=args.alpha, interpol=args.interpol, inbounds=inbounds, rowling=args.row, segs=args.segs, precomp=precomp)
            #vloss, verr, vploss, vperr = evaluate_baseline(test_loader, model[0], t)
        else:
            loss, err=train_clean(train_loader, model, opt, t)
            ploss, perr=0,0
            """
            train_robust(train_loader, model[0], opt, epsilon, t,
               train_log, args.verbose, args.real_time,
               norm_type=args.norm_train, bounded_input=False, clip_grad=1,
               **kwargs)
            err = evaluate_robust(test_loader, model[0], args.epsilon, t,
               test_log, args.verbose, args.real_time,
               norm_type=args.norm_test, bounded_input=False,
               **kwargs)
            """
        print('Epoch: {}'
              'Loss {} ({})\t'
              'Adv. Error {} ({})\t'
              'Error {} ({})'.format(
                  t, loss, best_loss, perr, best_perr, err, best_err))
        if err < best_err:
            patience=0
            print("SAVE")
            best_err = err
            best_loss = loss
            best_perr = perr
            name=args.prefix + "_" + args.method + "_" + args.dataset + "_" + args.model + "_eps_" + str(args.epsilon) + "_alpha_"+str(args.alpha)
            if args.method=='pwl':
                name=name+"_nseg_"+str(args.segs)
            torch.save({
                'state_dict' : [m.state_dict() for m in model],
                'err' : best_err,
                'epoch' : t},
                name+"_best.pth")
        else:
            patience=patience+1
        if patience > 50:
            print("ENOUGH!")
            break

else:
    checkpoint = torch.load(args.model_path)
    for sd,m in zip(checkpoint['state_dict'], model):
        m.load_state_dict(sd)
    #model.load_state_dict(checkpoint['state_dict'])
    model=model.eval()
    if args.eval_method=='em':
        vloss, verr, vperr = evaluate_em(test_loader, model, args.epsilon, 0, interpol=args.interpol, inbounds=inbounds, rowling=args.row)
    elif args.eval_method=='pwl':
        vloss, verr, vperr = evaluate_pwl(test_loader, model, args.epsilon, 0, interpol=args.interpol, inbounds=inbounds, rowling=args.row, segs=args.segs, use_clusters=args.use_clusters, approximate=args.approximate)
    elif args.eval_method=='linear':
        #input("Lins")
        vloss, verr, vperr = evaluate_lin(test_loader, model, args.epsilon, 0, inbounds=inbounds)
    else:
        #input(args.madry_eps)
        vloss, verr, vperr = evaluate_madry(test_loader, model, args.epsilon, 0, steps=args.madry_steps, stepsize=args.madry_eps, inbounds=inbounds)
    line='Loss {}\tAdv. Error {}\tError {}'.format(vloss, vperr, verr)
    print(line)
    if args.save:
        deer=os.path.join(args.savedir)
        if not os.path.exists(deer):
            os.makedirs(deer)
        txtpath = deer + args.model_path.split('/')[-1].split('.pth')[0]
        with open(txtpath + '.txt', 'w+') as f:
            f.write('\n'.join([line]))
