import pickle
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import random
import os

import utils_net


class binarization_dataset(Dataset):
    def __init__(self, root_dir, un_paired=True):
        self.root_dir = root_dir

        with open("Train_List.pickle", "rb") as handle:
            self.Train_Noisy_List, self.Train_Clean_List = pickle.load(handle)

        if un_paired:
            random.shuffle(self.Train_Noisy_List)
            random.shuffle(self.Train_Clean_List)

        transform_list_rgb = [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
        self.transform_normalze = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )
        self.transform_doc_rgb = transforms.Compose(transform_list_rgb)

    def __len__(self):
        return len(self.Train_Noisy_List)

    def __getitem__(self, item):
        deg_img_rgb = Image.open(
            os.path.join(self.root_dir, self.Train_Noisy_List[item])
        ).convert("RGB")
        clean_img_rgb = Image.open(
            os.path.join(self.root_dir, self.Train_Clean_List[item])
        ).convert("RGB")

        clean_img_rgb = self.transform_doc_rgb(clean_img_rgb)  # noise image
        deg_img_rgb = self.transform_doc_rgb(deg_img_rgb)  # clean image

        clean_img_rgb = 1.0 - clean_img_rgb
        clean_img_rgb = self.transform_normalze(clean_img_rgb)
        deg_img_rgb = self.transform_normalze(deg_img_rgb)

        return clean_img_rgb, deg_img_rgb


class Interface:
    """preserve order of the return of __getitem__"""

    def __init__(self, d):
        self.d = d

    def __len__(self):
        return len(self.d)

    def __getitem__(self, i):
        degraded, clean = self.d[i]
        return clean, degraded


def get_dataloader(opt):
    # trainset = binarization_dataset(root_dir=opt.root_dir)
    if opt.patch_size == 0:
        patch = None
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop(
                    (opt.max_size, opt.max_size),
                    fill=1.0,
                    pad_if_needed=True,
                    padding_mode="symmetric",
                ),
            ]
        )
    else:
        patch = utils_net.Patch(
            size=opt.patch_size, step=opt.patch_step if opt.patch_step > -1 else None
        )
        transform = transforms.ToTensor()
    if opt.in_chan == 3:
        print("WARNING: color input will be changed to gray scale using PCA fit!")
    trainset = Interface(
        utils_net.Dataset(
            opt.hdw_files,
            opt.cad_files,
            patch=patch,
            transform=transform,
            eps=0.9,
            limit_memory=opt.limit_mem,
            cache_file=".udbnet_train.cache",
            pca_fit=opt.in_chan == 3,
            seed=42,
        )
    )
    dataloader_train = DataLoader(
        trainset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=opt.workers,
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    return dataloader_train
