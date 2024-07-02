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
parser.add_argument("--opt",default='adam',type=str,help="optimizer")
parser.add_argument("--model",default='small',type=str,help="optimizer")
parser.add_argument("--dataset",default='cifar',type=str,help="dataset")
parser.add_argument("--prefix",default='robust_model',type=str,help="optimizer")
parser.add_argument("--method",default='madry',type=str,help="optimizer")
parser.add_argument("--lr",default=1e-4,type=float,help="learning rate")
parser.add_argument("--epsilon",default=2/255,type=float,help="")
parser.add_argument("--alpha",default=0.95,type=float,help="")
parser.add_argument("--interpol",default=10,type=int,help="")

parser.add_argument("--madry_eps",default=2/255,type=float,help="")
parser.add_argument("--madry_steps",default=20,type=float,help="")

parser.add_argument("--epochs",default=40,type=int,help="")
parser.add_argument("--seed",default=1234,type=int,help="")
parser.add_argument("--batch_size",default=32,type=int,help="training batch size")
parser.add_argument("--test_batch_size",default=32,type=int,help="test batch size")
args = parser.parse_args()


if args.dataset=='cifar':
    train_loader, _ = cifar_loaders(args.batch_size)
    _, test_loader = cifar_loaders(args.test_batch_size)
    if args.model=='small':
        model = cifar_model().cuda()
    elif args.model == 'large':
        model = cifar_model_large().cuda()
    elif args.model == 'resnet':
        model = model_resnet(cifar=True)
elif args.dataset=='mnist':
    train_loader, _ = mnist_loaders(args.batch_size)
    _, test_loader = mnist_loaders(args.test_batch_size)
    if args.model=='small':
        model = mnist_model().cuda()
    elif args.model == 'large':
        model = mnist_model_large().cuda()
    elif args.model == 'resnet':
        model = model_resnet(cifar=False)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(0)
np.random.seed(0)

sampler_indices = []

#model = [select_model(args.model)]





if args.opt == 'adam':
    opt = optim.Adam(model.parameters(), lr=args.lr)
else:
    raise ValueError("Unknown optimizer")

best_err=1
best_perr, best_loss=None, None
patience=0
for t in range(args.epochs):
    # standard training
    if args.method == 'em':
        loss, err, ploss, perr = train_em(train_loader, model, args.epsilon, opt, t, alpha=args.alpha, interpol=args.interpol)
        #vloss, verr, vploss, vperr = evaluate_baseline(test_loader, model[0], t)
    # madry training
    elif args.method=='madry':
        loss, err, ploss, perr = train_madry(train_loader, model, args.epsilon, opt, t, steps=args.madry_eps, stepsize=args.madry_steps)
        #vloss, verr, vperr = evaluate_madry(test_loader, model, args.epsilon, t, steps=20, stepsize=2/255)
    # robust training
    elif args.method == 'pwl':
        loss, err, ploss, perr = train_pwl(train_loader, model, args.epsilon, opt, t, alpha=args.alpha, interpol=args.interpol)
        #vloss, verr, vploss, vperr = evaluate_baseline(test_loader, model[0], t)
    # madry training
    else:
        pass
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
        torch.save({
            'state_dict' : [m.state_dict() for m in model],
            'err' : best_err,
            'epoch' : t},
            args.prefix + "_" + args.method + "_" + args.dataset + "_" + args.model + "_eps_" + str(args.epsilon) + "_best.pth")
    else:
        patience=patience+1
    if patience > 10:
        print("ENOUGH!")
        break

    #torch.save({
    #    'state_dict': [m.state_dict() for m in model],
    #    'err' : err,
    #    'epoch' : t},
    #    args.prefix + "_checkpoint.pth")
