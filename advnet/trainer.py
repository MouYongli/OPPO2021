import argparse
import os
import os.path as osp
import numpy as np
import datetime
import glob
import torch
from torch.autograd import Variable
from datasets import TripletDataset
from models import EmbeddingNet
from losses import TripletLoss, ContrastiveLoss

here = osp.dirname(osp.abspath(__file__))

def train_triplet_epoch(epoch, train_loader, model, loss_fn, optimizer, cuda, experiment_dir):
    model.train()
    train_loss = 0.0
    for batch_idx, sample in enumerate(train_loader):
        anchor, positive, negative = sample["img"], sample["same_img"], sample["other_img"]
        if cuda:
            anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
        anchor, positive, negative = Variable(anchor), Variable(positive), Variable(negative)
        optimizer.zero_grad()
        embedding_anchor, embedding_positive, embedding_negative = model(anchor), model(positive), model(negative)
        loss = loss_fn(embedding_anchor, embedding_positive, embedding_negative)
        train_loss += loss.data.item()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader)
    with open(osp.join(experiment_dir.out, 'triplet_log.csv'), 'a') as f:
        log = [epoch, train_loss]
        log = map(str, log)
        f.write(','.join(log) + '\n')

def test_triplet_epoch(epoch, test_loader, model, loss_fn, cuda, experiment_dir):
    model.eval()
    test_loss = 0.0
    for batch_idx, sample in enumerate(test_loader):
        anchor, positive, negative = sample["img"], sample["same_img"], sample["other_img"]
        if cuda:
            anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
        anchor, positive, negative = Variable(anchor), Variable(positive), Variable(negative)
        embedding_anchor, embedding_positive, embedding_negative = model(anchor), model(positive), model(negative)
        loss = loss_fn(embedding_anchor, embedding_positive, embedding_negative)
        train_loss += loss.data.item()
    train_loss /= len(train_loader)
    with open(osp.join(experiment_dir.out, 'triplet_log.csv'), 'a') as f:
        log = [epoch, train_loss]
        log = map(str, log)
        f.write(','.join(log) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
    parser.add_argument('--pretrain-epochs', type=int, default=0, help='epochs pretrain embeddingnet')
    parser.add_argument('--epochs', type=int, default=200, help='epochs')
    parser.add_argument('--lr', type=float, default=1.0e-10, help='learning rate',)
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay',)
    parser.add_argument('--seed', type=int, default=1337, help='seed for randomness',)
    parser.add_argument('--margin', type=float, default=1.0, help='margin in loss function',)
    args = parser.parse_args()

    # Set up experiment dir
    directory = os.path.join(here, 'run')
    runs = sorted(glob.glob(os.path.join(directory, 'experiment_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    experiment_dir = os.path.join(directory, 'experiment_{}'.format(str(run_id)))
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    if not osp.exists(osp.join(experiment_dir, 'triplet_log.csv')):
        with open(osp.join(experiment_dir.out, 'triplet_log.csv'), 'w') as f:
            f.write(','.join(['epoch', 'iteration']) + '\n')


    # Set up gpu configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    # Set up seed for randomness
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    # Define dataset
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_triplet_loader = torch.utils.data.DataLoader(TripletDataset(args, split='train'), batch_size=8, shuffle=True, **kwargs)
    valid_triplet_loader = torch.utils.data.DataLoader(TripletDataset(args, split='valid'), batch_size=1, shuffle=False, **kwargs)

    # Define model
    dnet = EmbeddingNet()
    if cuda:
        dnet = dnet.cuda()

    # Define optimizer
    triplet_optim = torch.optim.Adam(dnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Define loss function
    triplet_loss = TripletLoss(args.margin)
    contrastive_loss = ContrastiveLoss(args.margin)

    for epoch in range(args.pretrain_epochs):
        train_triplet_epoch(train_triplet_loader, dnet, triplet_loss, triplet_optim, cuda, experiment_dir)


