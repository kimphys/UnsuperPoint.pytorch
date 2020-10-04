import os
import sys

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

import torch
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np
from tqdm import tqdm
from PIL import Image

from network import Unsuperpoint
from args import args
from loss import ComputeLoss
from utils import save_train_samples

is_cuda = torch.cuda.is_available()

os.environ["RANK"] = "0"

class MyTrainDataset(Dataset):
    def __init__(self, img_path_file):
        f = open(img_path_file, 'r')
        img_list = f.read().splitlines()
        f.close()

        self.img_list = img_list
    
    def __getitem__(self, index):
        img_1 = Image.open(self.img_list[index]).convert('RGB')
        img_2 = img_1.copy()

        angle = random.randint(-30,30)
        img_2 = transforms.functional.rotate(img_2,angle)            

        custom_transform = transforms.Compose([transforms.Resize((args.HEIGHT,args.WIDTH)),
                                               transforms.ToTensor()])
        
        img_1 = custom_transform(img_1)
        img_2 = custom_transform(img_2)

        return img_1, img_2, angle

    def __len__(self):

        return len(self.img_list)

def setup(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destropy_process_group

def main():
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.gpu is not None:    
        if not len(args.gpu) > torch.cuda.device_count():
            ngpus_per_node = len(args.gpu)
        else:
            print("We will use all available GPUs")
            ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = torch.cuda.device_count()
    
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.rank)

    model = Unsuperpoint()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    epoch = 0

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        else:
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        # print("Training from scratch")
        pass
    
    img_path_file = args.dataset

    trainloader = DataLoader(MyTrainDataset(img_path_file), batch_size=args.batch_size, shuffle=True, num_workers=4)

    for ep in range(epoch, args.epochs):

        pbar = tqdm(trainloader)
        idx = 0

        for imgs_A, imgs_B, angle in pbar:
        # for imgs_A, imgs_B, angle in trainloader:
        
            '''
            if ep == 0 and idx ==0:
                save_train_samples(imgs_A,imgs_B,idx)
            '''

            if args.gpu is not None:
                imgs_A = imgs_A.cuda(args.gpu, non_blocking=True)
                imgs_B = imgs_B.cuda(args.gpu, non_blocking=True)

            optimizer.zero_grad()

            preds_A, preds_B = model(imgs_A,imgs_B)

            scr_A, pos_A_r, pos_A, desc_A = preds_A[0], preds_A[1], preds_A[2], preds_A[3]
            scr_B, pos_B_r, pos_B, desc_B = preds_B[0], preds_B[1], preds_B[2], preds_B[3]
            
            loss = ComputeLoss(angle, scr_A, scr_B, pos_A_r, pos_B_r, pos_A, pos_B, desc_A, desc_B)
            loss.backward()
            optimizer.step()

            idx += 1

        if (ep + 1) % args.save_per_epoch == 0:
            # Save model
            torch.save({
                        'epoch': ep,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                    }, args.save_model_dir + 'ckpt_{}.pt'.format(ep))
        

    print('Finished training')

if __name__ == "__main__":
    main()