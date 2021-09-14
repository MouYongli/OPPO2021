import numpy as np
import torch
from torch.autograd import Variable
from datasets import TripletDataset, FaceDataset
from models import EmbeddingNet, GaussianBlur, NoiseGenerator
from losses import TripletLoss

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pretrain-epochs', type=int, default=0, help='epochs pretrain embeddingnet')
    parser.add_argument('--epochs', type=int, default=200, help='epochs')
    parser.add_argument('--lr', type=float, default=1.0e-10, help='learning rate',)
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay',)
    parser.add_argument('--seed', type=int, default=1337, help='seed for randomness',)
    parser.add_argument('--margin', type=float, default=1.0, help='margin in loss function',)
    parser.add_argument('--kernel-size', type=int, default=1, help='kernel size of gaussian blur',)
    args = parser.parse_args()
    cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(TripletDataset(args, split='train', transform=True), batch_size=8, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(TripletDataset(args, split='test', transform=True), batch_size=8, shuffle=True, **kwargs)
    embeddingnet = EmbeddingNet()
    embedding_net = EmbeddingNet()
    advnet = NoiseGenerator()
    gaussian_blur = GaussianBlur(args.kernel_size)
    if cuda:
        embedding_net = embedding_net.cuda()
        advnet = advnet.cuda()
        gaussian_blur = gaussian_blur.cuda()
    triplet_loss = TripletLoss(args.margin)
    triplet_optim = torch.optim.Adam(embeddingnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    advnet.train()
    gaussian_blur.train()
    embeddingnet.train()
    # for batch_idx, sample in enumerate(train_loader):
    #     anchor, positive, negative = sample["img"], sample["same_img"], sample["other_img"]
    #     if cuda:
    #         anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
    #     anchor, positive, negative = Variable(anchor), Variable(positive), Variable(negative)
    #     triplet_optim.zero_grad()
    #     embedding_anchor, embedding_positive, embedding_negative = embeddingnet(anchor), embeddingnet(positive), embeddingnet(negative)
    #     loss = triplet_loss(embedding_anchor, embedding_positive, embedding_negative)
    #     print(loss.data.item())
    #     loss.backward()
    #     triplet_optim.step()

    imgs = torch.rand((8,3,112,112)).cuda()
    imgs = Variable(imgs)
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
    print(contra_mse)
    print(tv_l2)
    print(tv_l1)
    print(noise_l2)
    print(embedding_l2)


    # embeddingnet.eval()
    # test_embedding_vectors = []
    # test_pids = []
    # for batch_idx, sample in enumerate(test_loader):
    #     imgs, person_id, example_id = sample["img"], sample["person_id"], sample["example_id"]
    #     if cuda:
    #         imgs = imgs.cuda()
    #     imgs = Variable(imgs)
    #     embedding_imgs = embeddingnet(imgs).data.cpu()
    #     for img, pid in zip(embedding_imgs, person_id):
    #         print(img.shape)
    #         test_pids.append(pid)

    # from itertools import compress
    # test_embedding_vectors = [torch.rand((4096,)) for i in range(30)]
    # test_pids = [j for i in range(3) for j in range(10)]
    # for pid in np.unique(test_pids):
    #     same_img_embeddings = list(compress(test_embedding_vectors, test_pids == pid))
    #     for img_embedding in same_img_embeddings:
    #         print(pid, img_embedding.shape)