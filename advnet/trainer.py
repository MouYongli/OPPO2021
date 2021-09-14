import argparse
import os
import os.path as osp
import numpy as np
import glob
import torch
from torch.autograd import Variable
from datasets import TripletDataset, FaceDataset
from models import EmbeddingNet, GaussianBlur, NoiseGenerator
from losses import TripletLoss

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
    print("Epoch {} Triplet Loss {}".format(epoch, train_loss))
    # with open(osp.join(experiment_dir.out, 'triplet_log.csv'), 'a') as f:
    #     log = [epoch, train_loss]
    #     log = map(str, log)
    #     f.write(','.join(log) + '\n')

# def test_triplet_epoch(epoch, test_loader, model, cuda, experiment_dir):
#     model.eval()
#     test_loss = 0.0
#     test_embedding_vectors = []
#     test_pids = []
#     for batch_idx, sample in enumerate(test_loader):
#         imgs, person_id, example_id = sample["img"], sample["person_id"], sample["example_id"]
#         if cuda:
#             imgs = imgs.cuda()
#         imgs = Variable(imgs)
#         embedding_imgs = model(imgs)
#         for img, pid in zip(embedding_imgs, person_id):
#             test_embedding_vectors.append(img)
#             test_pids.append(img)
#     for pid in np.unique(test_pids):
#         pass

def train_adv_epoch(epoch, train_loader, advnet, gaussian_blur, embedding_net, adv_optim, cuda, experiment_dir, alpha_contra_mse=1.0, alpha_tv_l1=0.1, alpha_tv_l2=0.2, alpha_noise_l2=0.5, alpha_embedding_l2=0.5):
    advnet.train()
    gaussian_blur.train()
    embedding_net.train()
    train_loss = 0.0
    for batch_idx, sample in enumerate(train_loader):
        imgs, person_id, example_id = sample["img"], sample["person_id"], sample["example_id"]
        if cuda:
            imgs = imgs.cuda()
        imgs = Variable(imgs)
        adv_optim.zero_grad()
        noise = gaussian_blur(advnet(imgs))
        noised_img = imgs + noise
        contra_mse = (embedding_net(imgs) - embedding_net(noised_img)).pow(2).sum(1).sqrt().mean()
        diff1 = noised_img[:, :, :, :-1] - noised_img[:, :, :, 1:]
        diff2 = noised_img[:, :, :-1, :] - noised_img[:, :, 1:, :]
        diff3 = noised_img[:, :, 1:, :-1] - noised_img[:, :, :-1, 1:]
        diff4 = noised_img[:, :, :-1, :-1] - noised_img[:, :, 1:, 1:]
        tv_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
        tv_l1 = diff1.abs().mean() + diff2.abs().mean() + diff3.abs().mean() + diff4.abs().mean()
        noise_l2 = noise.pow(2).sum(1).sqrt().mean()
        embedding_l2 = embedding_net(noise).pow(2).sum(1).sqrt().mean()
        loss = alpha_contra_mse * contra_mse + alpha_tv_l1 * tv_l1 + alpha_tv_l2 * tv_l2 + alpha_noise_l2 * noise_l2 + alpha_embedding_l2 * embedding_l2
        train_loss += loss.data.item()
        loss.backward()
        adv_optim.step()
    train_loss /= len(train_loader)
    print("Train Adversarial Attacker Epoch {}, MSE {}, TV l1 {} l2 {}, Noise l2 {}, Embedding vector l2 {}".format(epoch, contra_mse, tv_l1, tv_l2, noise_l2, embedding_l2))
    # with open(osp.join(experiment_dir.out, 'adv_log.csv'), 'a') as f:
    #     log = [epoch, train_loss]
    #     log = map(str, log)
    #     f.write(','.join(log) + '\n')

def adversarial_attack(test_loader, advnet, gaussian_blur, cuda, experiment_dir):
    advnet.eval()
    gaussian_blur.eval()
    for batch_idx, sample in enumerate(test_loader):
        imgs, person_id, example_id = sample["img"], sample["person_id"], sample["example_id"]
        if cuda:
            imgs = imgs.cuda()
        imgs = Variable(imgs)
        with torch.no_grad():
            noise = gaussian_blur(advnet(imgs))
        noised_img = imgs + noise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
    parser.add_argument('--pretrain-epochs', type=int, default=0, help='epochs pretrain embeddingnet')
    parser.add_argument('--epochs', type=int, default=200, help='epochs')
    parser.add_argument('--lr', type=float, default=1.0e-10, help='learning rate',)
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay',)
    parser.add_argument('--seed', type=int, default=1337, help='seed for randomness',)
    parser.add_argument('--margin', type=float, default=1.0, help='margin in loss function',)
    parser.add_argument('--kernel-size', type=int, default=5, help='kernel size of gaussian blur',)
    parser.add_argument('--alpha-contra-mse', type=float, default=1.0, help='weight of contrastive mse loss',)
    parser.add_argument('--alpha-tv-l1', type=float, default=0.1, help='weight of tv l1 loss',)
    parser.add_argument('--alpha-tv-l2', type=float, default=0.2, help='weight of tv l2 loss',)
    parser.add_argument('--alpha-noise-l2', type=float, default=0.5, help='weight of noise l2 loss',)
    parser.add_argument('--alpha-embedding-l2', type=float, default=0.5, help='weight of embedding l2 loss',)
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
    train_triplet_loader = torch.utils.data.DataLoader(TripletDataset(args, split='train', transform=True), batch_size=8, shuffle=True, **kwargs)
    test_triplet_loader = torch.utils.data.DataLoader(TripletDataset(args, split='test', transform=True), batch_size=8, shuffle=False, **kwargs)
    train_adv_loader = torch.utils.data.DataLoader(FaceDataset(args, split='train', transform=True), batch_size=8, shuffle=True, **kwargs)
    test_adv_loader = torch.utils.data.DataLoader(FaceDataset(args, split='test', transform=True), batch_size=8, shuffle=False, **kwargs)
    # Define model
    embedding_net = EmbeddingNet()
    advnet = NoiseGenerator()
    gaussian_blur = GaussianBlur(args.kernel_size)
    if cuda:
        embedding_net = embedding_net.cuda()
        advnet = advnet.cuda()
        gaussian_blur = gaussian_blur.cuda()


    # Define optimizer
    triplet_optim = torch.optim.Adam(embedding_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    adv_optim = torch.optim.Adam(advnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Define loss function
    triplet_loss = TripletLoss(args.margin)

    for epoch in range(args.pretrain_epochs):
        train_triplet_epoch(epoch, train_triplet_loader, embedding_net, triplet_loss, triplet_optim, cuda, experiment_dir)

    for epoch in range(args.epochs):
        train_adv_epoch(epoch, train_adv_loader, advnet, gaussian_blur, embedding_net, adv_optim, cuda, experiment_dir, alpha_contra_mse=args.alpha_contra_mse, alpha_tv_l1=args.alpha_tv_l1, alpha_tv_l2=args.alpha_tv_l2, alpha_noise_l2=args.alpha_noise_l2, alpha_embedding_l2=args.alpha_embedding_l2)


