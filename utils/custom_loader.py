from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple
import torchvision.transforms as tforms
import torch
import os
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
import random
from utils.custom_transform import Binarize, Scale_0_1
import time

class QuickDraw_FS(VisionDataset):
    """ The QuickDraw dataset

        Args:
            root (string): Root directory of dataset where the archive *.pt are stored.
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
        """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            transform_variation: Optional[Callable] = None,

            target_transform: Optional[Callable] = None,
            sample_per_class: Optional[int] = 500,
            exemplar_type: Optional[str] = 'first',
            augment_class: Optional[bool] = False,
            train_flag: Optional[bool] = None,
            concat: Optional[bool] = False

    ) -> None:
        super(QuickDraw_FS, self).__init__(root, transform=transform, target_transform=target_transform)
        self.transform_variation = transform_variation
        self.param_RandomAffine = [(-90, 90), (0.3, 0.3), (0.5, 1.5), (-30, 30, -30, 30)]
        # {'degrees': (-90, 90),
        # 'translate': (0.3, 0.3),
        # 'scale': (0.5, 1.5),
        # 'shear': (-30, 30, -30, 30)
        self.train_flag = train_flag
        self.concat = concat

        self.sample_per_class = sample_per_class
        self.augment_class = augment_class
        if self.augment_class:
            self.augmentation_class = [
                TF.hflip, TF.vflip,
                lambda img: TF.rotate(img, 0),
                tforms.RandomAffine(*self.param_RandomAffine)
            ]


        if self.sample_per_class < 500:
            loaded_file = np.load(root+'/all_qd_fs.npz')
        elif 500 <= self.sample_per_class <= 2000:
            loaded_file = np.load(root + '/all_qd_fs_2000.npz')
        else:
            raise ValueError("nb sample should be lower than 2000")

        images = loaded_file['data']

        if train_flag is not None:
            if self.train_flag:
                selected_labels = np.arange(550)
            else:
                selected_labels = np.arange(550, len(images))
        else:
            selected_labels = np.arange(len(images))

        self.exemplar_type = exemplar_type

        if self.exemplar_type in ['prototype', 'first']:
            exemplar = images[selected_labels, 0]
            self.exemplar = torch.from_numpy(exemplar)
        elif self.exemplar_type == 'shuffle':
            self.exemplar = None
        else:
            raise NotImplementedError()
        self.variation = images[selected_labels, :self.sample_per_class].reshape(-1, 1, 48, 48)
        self.variation = torch.from_numpy(self.variation)
        intermediary = np.arange(len(selected_labels)).reshape(-1, 1)
        self.targets = torch.from_numpy(np.repeat(intermediary, self.sample_per_class, axis=1).flatten())

        id_list = []
        class_id_list = []

        for each_elem in range(self.variation.size(0)):
            id_list.append(each_elem)
            label = self.targets[each_elem]
            class_id_list.append(int(label.item()))

        self.df = pd.DataFrame(data={'id': id_list, 'class_id': class_id_list})

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img_variation, target = self.variation[index].view(48, 48), int(self.targets[index])

        idx_exemplar = target

        if self.exemplar_type in ['prototype', 'first']:
            img_exemplar = self.exemplar[idx_exemplar].view(48, 48)
        elif self.exemplar_type == 'shuffle':
            item_exemplar = self.df[self.df['class_id'] == target].sample(1)['id'].values[0]
            img_exemplar = self.variation[item_exemplar].view(48, 48)
            # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img_variation = Image.fromarray(img_variation.numpy())
        img_exemplar = Image.fromarray(img_exemplar.numpy())
        if self.augment_class:
            rnd_idx = np.random.choice(np.arange(len(self.augmentation_class)), p=[0.2, 0.2, 0.2, 0.4])
            if rnd_idx == 3:
                param = self.augmentation_class[rnd_idx].get_params(*self.param_RandomAffine, img_size=[48, 48])
                img_variation = TF.affine(img_variation, *param)
                img_exemplar = TF.affine(img_exemplar, *param)


            else:
                trans = self.augmentation_class[rnd_idx]
                img_variation = trans(img_variation)
                img_exemplar = trans(img_exemplar)

        if self.transform is not None:
            img_variation = self.transform(img_variation)
            img_exemplar = self.transform(img_exemplar)
        if self.transform_variation is not None:
            img_variation = self.transform_variation(img_variation)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_variation, img_exemplar, target

    def __len__(self) -> int:
        return len(self.variation)

class QuickDraw_FS_clust(VisionDataset):
    """ The QuickDraw dataset

        Args:
            root (string): Root directory of dataset where the archive *.pt are stored.
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
        """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            transform_variation: Optional[Callable] = None,

            target_transform: Optional[Callable] = None,
            sample_per_class: Optional[int] = 500,
            exemplar_type: Optional[str] = 'first',
            augment_class: Optional[bool] = False,
            train_flag:Optional[bool] = None

    ) -> None:
        super(QuickDraw_FS_clust, self).__init__(root, transform=transform, target_transform=target_transform)
        self.transform_variation = transform_variation
        self.param_RandomAffine = [(-90, 90), (0.3, 0.3), (0.5, 1.5), (-30, 30, -30, 30)]
        # {'degrees': (-90, 90),
        # 'translate': (0.3, 0.3),
        # 'scale': (0.5, 1.5),
        # 'shear': (-30, 30, -30, 30)
        self.train_flag = train_flag

        self.sample_per_class = sample_per_class
        self.augment_class = augment_class
        if self.augment_class:
            self.augmentation_class = [
                TF.hflip, TF.vflip,
                lambda img: TF.rotate(img, 0),
                tforms.RandomAffine(*self.param_RandomAffine)
            ]

        if self.sample_per_class == 500:
            loaded_file = np.load(root + '/all_qd_fs_shuffled.npz')
        else:
            raise ValueError("nb sample should not be different than 500")


        images = loaded_file['data']
        prototype = loaded_file['prototype']

        if train_flag is not None:
            if self.train_flag:
                selected_labels = np.arange(550)
            else:
                selected_labels = np.arange(550, len(images))
        else:
            selected_labels = np.arange(len(images))

        self.exemplar_type = exemplar_type


        if self.exemplar_type in ['prototype', 'first']:
            exemplar = prototype[selected_labels,0]
            self.exemplar = torch.from_numpy(exemplar)
        elif self.exemplar_type == 'shuffle':
            self.exemplar = None
        else:
            raise NotImplementedError()
        self.variation = images[selected_labels, :self.sample_per_class].reshape(-1, 1, 48, 48)
        self.variation = torch.from_numpy(self.variation)
        intermediary = np.arange(len(selected_labels)).reshape(-1, 1)
        self.targets = torch.from_numpy(np.repeat(intermediary, self.sample_per_class, axis=1).flatten())

        id_list = []
        class_id_list = []

        for each_elem in range(self.variation.size(0)):
            id_list.append(each_elem)
            label = self.targets[each_elem]
            class_id_list.append(int(label.item()))

        self.df = pd.DataFrame(data={'id': id_list, 'class_id': class_id_list})

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        #tic = time.time()
        img_variation, target = self.variation[index].view(48, 48), int(self.targets[index])
        #toc = time.time()
        #duration = toc - tic
        #print(f'loading variation {duration:0.6f}')
        idx_exemplar = target

        #tic = time.time()
        if self.exemplar_type in ['prototype', 'first']:
            img_exemplar = self.exemplar[idx_exemplar].view(48, 48)
        elif self.exemplar_type == 'shuffle':
            item_exemplar = self.df[self.df['class_id'] == target].sample(1)['id'].values[0]
            img_exemplar = self.variation[item_exemplar].view(48, 48)
            # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #toc = time.time()
        #duration = toc - tic
        #print(f'loading exemplar {duration:0.6f}')

        #tic = time.time()
        img_variation = Image.fromarray(img_variation.numpy())
        img_exemplar = Image.fromarray(img_exemplar.numpy())
        #toc = time.time()
        #duration = toc - tic
        #print(f'To Pil {duration:0.6f}')
        if self.augment_class:
            rnd_idx = np.random.choice(np.arange(len(self.augmentation_class)), p=[0.2, 0.2, 0.2, 0.4])
            if rnd_idx == 3:
                param = self.augmentation_class[rnd_idx].get_params(*self.param_RandomAffine, img_size=[48, 48])
                img_variation = TF.affine(img_variation, *param)
                img_exemplar = TF.affine(img_exemplar, *param)


            else:
                trans = self.augmentation_class[rnd_idx]
                img_variation = trans(img_variation)
                img_exemplar = trans(img_exemplar)

        #tic = time.time()
        if self.transform is not None:
            img_variation = self.transform(img_variation)
            img_exemplar = self.transform(img_exemplar)
        #toc = time.time()
        #duration = toc - tic
        #print(f'transform {duration:0.6f}')
        if self.transform_variation is not None:
            img_variation = self.transform_variation(img_variation)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_variation, img_exemplar, target

    def __len__(self) -> int:
        return len(self.variation)


class Contrastive_augmentation(object):
    def __init__(self, target_size, strength='normal'):
        super().__init__()
        self.target_size = target_size

        if strength == 'normal':
            self.tr_augment = tforms.Compose([
                tforms.RandomChoice([
                    tforms.RandomResizedCrop(self.target_size, scale=(0.75, 1.33), ratio=(0.8, 1.2)),
                    tforms.RandomAffine((-15, 15), scale=(0.75, 1.33), translate=(0.1, 0.1), shear=(-10, 10),
                                            fillcolor=0),
                    tforms.RandomPerspective(distortion_scale=0.5, p=0.5, fill=0)
                ])
            ])

        elif strength == 'light':
            self.tr_augment = tforms.Compose([
                tforms.RandomChoice([
                    tforms.RandomResizedCrop(self.target_size, scale=(0.90, 1.1), ratio=(0.9, 1.1)),
                    tforms.RandomAffine((-7, 7), scale=(0.90, 1.1), translate=(0.05, 0.05), shear=(-5, 5),
                                            fillcolor=0),
                    tforms.RandomPerspective(distortion_scale=0.25, p=0.5, fill=0)
                ])
            ])
        elif strength == 'strong':
            self.tr_augment = tforms.Compose([
                tforms.RandomChoice([
                    tforms.RandomResizedCrop(self.target_size, scale=(0.5, 1.5), ratio=(0.6, 1.4)),
                    tforms.RandomAffine((-30, 30), scale=(0.5, 1.5), translate=(0.2, 0.2), shear=(-20, 20),
                                            fillcolor=0),
                    tforms.RandomPerspective(distortion_scale=0.5, p=0.5, fill=0)
                ])
            ])

        self.tranform_special = tforms.Compose([
            tforms.ToTensor(),
            #Scale_0_1(),
        ])

    def __call__(self, input_image, proto=None):
        device = input_image.device
        input_image = input_image.cpu()
        all_image_1 = torch.zeros_like(input_image)
        if proto is not None:
            all_image_2 = torch.zeros_like(proto)
        else:
            all_image_2 = torch.zeros_like(input_image)
        for idx_image in range(input_image.size(0)):
            image = tforms.functional.to_pil_image(input_image[idx_image], mode="L")
            #image = Image.fromarray(tensor[input_image, 0, :, :].cpu().numpy(), mode = 'L')
            image_tr_1 = self.tr_augment(image)
            image_tr_1 = self.tranform_special(image_tr_1)
            all_image_1[idx_image, 0, :, :] = image_tr_1

            if proto is not None:
                proto_im = tforms.functional.to_pil_image(proto[idx_image], mode="L")
                image_tr_2 = self.tr_augment(proto_im)
            else:
                image_tr_2 = self.tr_augment(image)
            image_tr_2 = self.tranform_special(image_tr_2)
            all_image_2[idx_image, 0, :, :] = image_tr_2
        return all_image_1.to(device), all_image_2.to(device)

class Cont_ProtAug(object):
    def __init__(self, target_size, alpha=1):
        super().__init__()
        self.target_size = target_size


        self.tr_augment = tforms.Compose([
            tforms.RandomChoice([
                tforms.RandomResizedCrop(self.target_size, scale=(1-alpha*0.1, 1+alpha*0.1), ratio=(1-alpha*0.1, 1+alpha*0.1)),
                tforms.RandomAffine((-7, 7), scale=(1-alpha*0.1, 1+alpha*0.1), translate=(alpha*0.05, alpha*0.05), shear=(-5, 5),
                                        fillcolor=0),
                tforms.RandomPerspective(distortion_scale=0.25, p=0.5, fill=0)
            ])
        ])


        self.tranform_special = tforms.Compose([
            tforms.ToTensor(),
            #Scale_0_1(),
        ])

    def __call__(self, input_image):
        device = input_image.device
        input_image = input_image.cpu()
        all_image_1 = torch.zeros_like(input_image)
        for idx_image in range(input_image.size(0)):
            image = tforms.functional.to_pil_image(input_image[idx_image], mode="L")
            #image = Image.fromarray(tensor[input_image, 0, :, :].cpu().numpy(), mode = 'L')
            image_tr_1 = self.tr_augment(image)
            image_tr_1 = self.tranform_special(image_tr_1)
            all_image_1[idx_image, 0, :, :] = image_tr_1
        return all_image_1.to(device)

class Contrastive_augmentation_fast(object):
    def __init__(self, target_size, strength='normal'):
        super().__init__()
        self.target_size = target_size

        if strength == 'normal':
            self.tr_augment = tforms.Compose([
                tforms.RandomChoice([
                    tforms.RandomResizedCrop(self.target_size, scale=(0.75, 1.33), ratio=(0.8, 1.2)),
                    tforms.RandomAffine((-15, 15), scale=(0.75, 1.33), translate=(0.1, 0.1), shear=(-10, 10),
                                            fillcolor=0),
                    tforms.RandomPerspective(distortion_scale=0.5, p=0.5, fill=0)
                ])
            ])

        elif strength == 'light':
            self.tr_augment = tforms.Compose([
                tforms.RandomChoice([
                    tforms.RandomResizedCrop(self.target_size, scale=(0.90, 1.1), ratio=(0.9, 1.1)),
                    tforms.RandomAffine((-7, 7), scale=(0.90, 1.1), translate=(0.05, 0.05), shear=(-5, 5),
                                            fillcolor=0),
                    tforms.RandomPerspective(distortion_scale=0.25, p=0.5, fill=0)
                ])
            ])
        elif strength == 'strong':
            self.tr_augment = tforms.Compose([
                tforms.RandomChoice([
                    tforms.RandomResizedCrop(self.target_size, scale=(0.5, 1.5), ratio=(0.6, 1.4)),
                    tforms.RandomAffine((-30, 30), scale=(0.5, 1.5), translate=(0.2, 0.2), shear=(-20, 20),
                                            fillcolor=0),
                    tforms.RandomPerspective(distortion_scale=0.75, p=0.5, fill=0)
                ])
            ])

        self.tr_augment_n = tforms.Lambda(lambda x: torch.stack([self.tr_augment(x_) for x_ in x]))

        self.tranform_special = tforms.Compose([
            tforms.ToTensor(),
            #Scale_0_1(),
        ])

    def __call__(self, input_image):
        all_image_1 = self.tr_augment_n(input_image)
        all_image_2 = self.tr_augment_n(input_image)
        return all_image_1, all_image_2