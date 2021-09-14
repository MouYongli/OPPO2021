import os
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
import random
import torch
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    """
    Crack500 dataset
    """
    def __init__(self, args, base_dir=None, split='train', transform=True):
        """
        :param base_dir: path to crack500 dataset directory: crop_size, train_crop_strategy, val_crop_strategy, test_crop_strategy
        :param split: train/val/test
        :param transform: transform to apply
        """
        super().__init__()
        if base_dir is not None:
            self.base_dir = base_dir
        else:
            pwd = osp.dirname(osp.abspath(__file__))
            self.base_dir = osp.join(pwd, '../data/images')
        self.split = split
        self.args = args
        self.transform = transform

        image_files = []
        for subdir in os.listdir(self.base_dir):
            for file_name in os.listdir(osp.join(self.base_dir, subdir)):
                if file_name.endswith('.jpg'):
                    image_files.append(["{}/{}".format(subdir, file_name), subdir, file_name[-5:-4]])
        self.images_df = pd.DataFrame(image_files, columns=["file_path", "person_id", "example_id"])

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, idx):
        if self.split == "train":
            person_id, example_id = self.images_df.loc[idx, "person_id"], self.images_df.loc[idx, "example_id"]
            other_person_list = self.images_df["person_id"].unique()
            other_person_id = random.choice(np.delete(other_person_list,(other_person_list == person_id)))
            other_person_example_list = self.images_df[self.images_df["person_id"] == other_person_id]["example_id"].unique()
            other_person_exampel_id = random.choice(other_person_example_list)
            same_person_exmaple_list = self.images_df[self.images_df["person_id"] == person_id]["example_id"].unique()
            same_person_exmaple_id = random.choice(np.delete(same_person_exmaple_list,(same_person_exmaple_list == example_id)))

            img = skimage.io.imread(osp.join(self.base_dir, "{}/{}_{}.jpg".format(person_id, person_id, example_id)))
            same_img = skimage.io.imread(osp.join(self.base_dir, "{}/{}_{}.jpg".format(person_id, person_id, same_person_exmaple_id)))
            other_img = skimage.io.imread(osp.join(self.base_dir, "{}/{}_{}.jpg".format(other_person_id, other_person_id, other_person_exampel_id)))
            img = skimage.transform.resize(img, (self.args.img_size, self.args.img_size))
            same_img = skimage.transform.resize(same_img, (self.args.img_size, self.args.img_size))
            other_img = skimage.transform.resize(other_img, (self.args.img_size, self.args.img_size))
            if self.transform:
                img, same_img, other_img = self.numpy_to_tensor(img), self.numpy_to_tensor(same_img), self.numpy_to_tensor(other_img)
            sample = {"person_id": int(person_id), "example_id": int(example_id), "img": img, "same_img": same_img, "other_img": other_img}
        elif self.split == "test":
            person_id, example_id = self.images_df.loc[idx, "person_id"], self.images_df.loc[idx, "example_id"]
            img = skimage.io.imread(osp.join(self.base_dir, "{}/{}_{}.jpg".format(person_id, person_id, example_id)))
            img = skimage.transform.resize(img, (self.args.img_size, self.args.img_size))
            if self.transform:
                img = self.numpy_to_tensor(img)
            sample = {"person_id": int(person_id), "example_id": int(example_id), "img": img}
        else:
            raise  NotImplementedError
        return sample

    def numpy_to_tensor(self, img):
        img = img.astype(np.float64)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img

    def tensor_to_numpy(self, img):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img = img.astype(np.uint8)
        # img = img[:, :, ::-1]
        return img

class FaceDataset(Dataset):
    """
    Crack500 dataset
    """
    def __init__(self, args, base_dir=None, split='train', transform=True):
        """
        :param base_dir: path to crack500 dataset directory: crop_size, train_crop_strategy, val_crop_strategy, test_crop_strategy
        :param split: train/val/test
        :param transform: transform to apply
        """
        super().__init__()
        if base_dir is not None:
            self.base_dir = base_dir
        else:
            pwd = osp.dirname(osp.abspath(__file__))
            self.base_dir = osp.join(pwd, '../data/images')
        self.split = split
        self.args = args
        self.transform = transform
        image_files = []
        for subdir in os.listdir(self.base_dir):
            for file_name in os.listdir(osp.join(self.base_dir, subdir)):
                if file_name.endswith('.jpg'):
                    image_files.append(["{}/{}".format(subdir, file_name), subdir, file_name[-5:-4]])
        self.images_df = pd.DataFrame(image_files, columns=["file_path", "person_id", "example_id"])

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, idx):
        person_id, example_id = self.images_df.loc[idx, "person_id"], self.images_df.loc[idx, "example_id"]
        img = skimage.io.imread(osp.join(self.base_dir, "{}/{}_{}.jpg".format(person_id, person_id, example_id)))
        height, width = img.shape[0], img.shape[1]
        img = skimage.transform.resize(img, (self.args.img_size, self.args.img_size))
        if self.transform:
            img = self.numpy_to_tensor(img)
        sample = {"person_id": int(person_id), "example_id": int(example_id), "height": height, "width":width, "img": img}
        return sample

    def numpy_to_tensor(self, img):
        img = img.astype(np.float64)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img

    def tensor_to_numpy(self, img):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img = img.astype(np.uint8)
        # img = img[:, :, ::-1]
        return img

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Siamese Dataset")
    parser.add_argument('--model', type=str, default='deeplab', choices=['deeplab', 'linknet', 'unet', 'fcn'],
                        help='model name (default: deeplab)')
    parser.add_argument('--img-size', type=int, default=112, help='image size (default: 112)')
    args = parser.parse_args()
    train_triplet_dataset = TripletDataset(args, split="train")
    sample = train_triplet_dataset[0]
    # img, same_img, other_img = sample["img"], sample["same_img"], sample["other_img"]
    # plt.subplot(131)
    # plt.imshow(img)
    # plt.subplot(132)
    # plt.imshow(same_img)
    # plt.subplot(133)
    # plt.imshow(other_img)
    # plt.show()

    test_triplet_dataset = TripletDataset(args, split="test")
    sample = test_triplet_dataset[0]
    img, person_id, example_id = sample["img"], sample["person_id"], sample["example_id"]
    print("person id", person_id, example_id)
    print(person_id.__repr__)
    plt.imshow(img)
    plt.show()
