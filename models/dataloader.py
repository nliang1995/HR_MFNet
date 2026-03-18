# -*- encoding: utf-8 -*-
'''
@File    :   dataloader.py
@Time    :   2025-12-15
@Author  :   niuliang 
@Version :   1.0
@Contact :   niuliang@cumt.edu.cn
'''


import os
import torch
import torchvision.transforms.functional as tf
import random
import numpy as np

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset

from models import utils


# fix randomness on DataLoader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# https://github.com/rwightman/pytorch-image-models/blob/d72ac0db259275233877be8c1d4872163954dfbb/timm/data/loader.py
class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def is_image(src):
    return True if os.path.splitext(src)[1].lower() in ['.jpg', '.png', '.tif', '.ppm','.gif','.bmp'] else False


class Image2ImageLoader_resize(Dataset):

    def __init__(self,
                 x_path,
                 y_path,
                 z_path,
                 mode,
                 **kwargs):

        self.mode = mode
        self.args = kwargs['args']
        self.use_fov = bool(getattr(self.args, 'use_fov', False))
        if self.use_fov and (not z_path or not os.path.isdir(z_path)):
            raise ValueError('use_fov is true, but z_path is missing or invalid.')
        if not self.use_fov:
            z_path = None

        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]

        x_img_name = os.listdir(x_path)
        y_img_name = os.listdir(y_path)
        x_img_name = filter(is_image, x_img_name)
        y_img_name = filter(is_image, y_img_name)
        if self.use_fov:
            z_img_name = os.listdir(z_path)
            z_img_name = filter(is_image, z_img_name)

        self.img_x_path = []
        self.img_y_path = []
        self.img_z_path = []

        x_img_name = sorted(x_img_name)
        y_img_name = sorted(y_img_name)
        if self.use_fov:
            z_img_name = sorted(z_img_name)
            img_paths = zip(x_img_name, y_img_name, z_img_name)
            for item in img_paths:
                self.img_x_path.append(x_path + os.sep + item[0])
                self.img_y_path.append(y_path + os.sep + item[1])
                self.img_z_path.append(z_path + os.sep + item[2])
        else:
            img_paths = zip(x_img_name, y_img_name)
            for item in img_paths:
                self.img_x_path.append(x_path + os.sep + item[0])
                self.img_y_path.append(y_path + os.sep + item[1])

        assert len(self.img_x_path) == len(self.img_y_path), 'Images in directory must have same file indices!!'
        if self.use_fov:
            assert len(self.img_x_path) == len(self.img_z_path), 'Images in directory must have same file indices!!'

        print(f'{utils.Colors.LIGHT_RED}Mounting data on memory...{self.__class__.__name__}:{self.mode}{utils.Colors.END}')
        self.img_x = []
        self.img_y = []
        self.img_z = []
        for index in range(len(self.img_x_path)):
            x_path = self.img_x_path[index]
            y_path = self.img_y_path[index]
            self.img_x.append(Image.open(x_path).convert('RGB'))
            self.img_y.append(Image.open(y_path).convert('L'))
            if self.use_fov:
                z_path = self.img_z_path[index]
                self.img_z.append(Image.open(z_path).convert('L'))

    def transform(self, image, target, fov):
        if fov is None:
            fov = Image.new('L', image.size, color=255)

        resize_h = self.args.input_size[0]
        resize_w = self.args.input_size[1]
        image = tf.resize(image, [resize_h, resize_w])
        target = tf.resize(target, [resize_h, resize_w], interpolation=InterpolationMode.NEAREST)
        fov = tf.resize(fov, [resize_h, resize_w], interpolation=InterpolationMode.NEAREST)

        if not self.mode == 'validation':
            random_gen = random.Random()  # thread-safe random

            if (random_gen.random() < 0.8) and self.args.transform_cutmix:
                rand_n = random_gen.randint(0, self.__len__() - 1)     # randomly generates reference image on dataset
                image_refer = Image.open(self.img_x_path[rand_n]).convert('RGB')
                target_refer = Image.open(self.img_y_path[rand_n]).convert('L')
                image, target = utils.cut_mix(image, target, image_refer, target_refer)

            if (random_gen.random() < 0.8) and self.args.transform_rand_resize:
                rand_h = (random_gen.random() * 1.5) + 0.5  # [0.5, 2.0]
                rand_w = (random_gen.random() * 1.5) + 0.5
                resize_h = int((self.args.input_size[0] * rand_h).__round__())
                resize_w = int((self.args.input_size[1] * rand_w).__round__())

                image = tf.resize(image, [resize_h, resize_w])
                target = tf.resize(target, [resize_h, resize_w], interpolation=InterpolationMode.NEAREST)
                fov = tf.resize(fov, [resize_h, resize_w], interpolation=InterpolationMode.NEAREST)

            if hasattr(self.args, 'transform_rand_crop'):
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(int(self.args.transform_rand_crop), int(self.args.transform_rand_crop)))
                image = tf.crop(image, i, j, h, w)
                target = tf.crop(target, i, j, h, w)
                fov = tf.crop(fov, i, j, h, w)

            if (random_gen.random() < 0.5) and self.args.transform_hflip:
                image = tf.hflip(image)
                target = tf.hflip(target)
                fov = tf.hflip(fov)

            if (random_gen.random() < 0.8) and self.args.transform_jitter:
                transform = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                image = transform(image)

            if (random_gen.random() < 0.5) and self.args.transform_blur:
                kernel_size = int((random.random() * 10 + 2.5).__round__())    # random kernel size 3 to 11
                if kernel_size % 2 == 0:
                    kernel_size -= 1
                transform = transforms.GaussianBlur(kernel_size=kernel_size)
                image = transform(image)

            # recommend to use at the end.
            if (random_gen.random() < 0.3) and self.args.transform_perspective:
                start_p, end_p = transforms.RandomPerspective.get_params(image.width, image.height, distortion_scale=0.5)
                image = tf.perspective(image, start_p, end_p)
                target = tf.perspective(target, start_p, end_p, interpolation=InterpolationMode.NEAREST)
                fov = tf.perspective(fov, start_p, end_p, interpolation=InterpolationMode.NEAREST)

        image_tensor = tf.to_tensor(image)
        target_tensor = torch.tensor(np.array(target))
        fov_tensor = torch.tensor(np.array(fov))

        if self.args.input_space == 'GR':   # grey, red
            image_tensor_r = image_tensor[0].unsqueeze(0)
            image_tensor_grey = tf.to_tensor(tf.to_grayscale(image))

            image_tensor = torch.cat((image_tensor_r, image_tensor_grey), dim=0)

        # 'mean' and 'std' are acquired by cropped face from sense-time landmark
        if self.args.input_space == 'RGB':
            image_tensor = tf.normalize(image_tensor,
                                        mean=self.image_mean,
                                        std=self.image_std)

        if self.args.n_classes <= 2:  # for visualized binary GT
            target_tensor[target_tensor < 128] = 0
            target_tensor[target_tensor >= 128] = 1

        if self.args.n_classes <= 2:  # for visualized binary FOV
            fov_tensor[fov_tensor < 128] = 0
            fov_tensor[fov_tensor >= 128] = 1
        target_tensor = target_tensor.unsqueeze(0)    # expand 'grey channel' for loss function dependency
        fov_tensor = fov_tensor.unsqueeze(0)          # expand 'grey channel' for loss function dependency

        if hasattr(self.args, 'mask_input_outside_fov') and bool(self.args.mask_input_outside_fov):
            image_tensor = image_tensor * fov_tensor.float()

        return image_tensor, target_tensor, fov_tensor

    def __getitem__(self, index):
        if self.use_fov:
            img_x_tr, img_y_tr, img_z_tr = self.transform(self.img_x[index], self.img_y[index], self.img_z[index])
            img_z_path = self.img_z_path[index]
        else:
            img_x_tr, img_y_tr, img_z_tr = self.transform(self.img_x[index], self.img_y[index], None)
            img_z_path = ''

        return (img_x_tr, self.img_x_path[index]), (img_y_tr, self.img_y_path[index]), (img_z_tr, img_z_path)

    def __len__(self):
        return len(self.img_x_path)

class Image2ImageDataLoader_resize:

    def __init__(self,
                 x_path,
                 y_path,
                 z_path,
                 mode,
                 batch_size=4,
                 num_workers=0,
                 pin_memory=True,
                 **kwargs):

        g = torch.Generator()
        g.manual_seed(3407)

        self.image_loader = Image2ImageLoader_resize(x_path,
                                                     y_path,
                                                     z_path,
                                                     mode=mode,
                                                     **kwargs)

        # use your own data loader
        self.Loader = MultiEpochsDataLoader(self.image_loader,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            shuffle=(not mode == 'validation'),
                                            worker_init_fn=seed_worker,
                                            generator=g,
                                            pin_memory=pin_memory)

    def __len__(self):
        return self.image_loader.__len__()


class Image2ImageLoader_zero_pad(Dataset):

    def __init__(self,
                 x_path,
                 y_path,
                 z_path,
                 mode,
                 **kwargs):

        self.mode = mode
        self.args = kwargs['args']
        self.use_fov = bool(getattr(self.args, 'use_fov', False))
        if self.use_fov and (not z_path or not os.path.isdir(z_path)):
            raise ValueError('use_fov is true, but z_path is missing or invalid.')
        if not self.use_fov:
            z_path = None

        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]

        x_img_name = os.listdir(x_path)
        y_img_name = os.listdir(y_path)
        x_img_name = filter(is_image, x_img_name)
        y_img_name = filter(is_image, y_img_name)
        if self.use_fov:
            z_img_name = os.listdir(z_path)
            z_img_name = filter(is_image, z_img_name)

        self.img_x_path = []
        self.img_y_path = []
        self.img_z_path = []

        x_img_name = sorted(x_img_name)
        y_img_name = sorted(y_img_name)
        if self.use_fov:
            z_img_name = sorted(z_img_name)
            img_paths = zip(x_img_name, y_img_name, z_img_name)
            for item in img_paths:
                # print("****************************")
                self.img_x_path.append(x_path + os.sep + item[0])
                self.img_y_path.append(y_path + os.sep + item[1])
                self.img_z_path.append(z_path + os.sep + item[2])
        else:
            img_paths = zip(x_img_name, y_img_name)
            for item in img_paths:
                self.img_x_path.append(x_path + os.sep + item[0])
                self.img_y_path.append(y_path + os.sep + item[1])

        assert len(self.img_x_path) == len(self.img_y_path), 'Images in directory must have same file indices!!'
        if self.use_fov:
            assert len(self.img_x_path) == len(self.img_z_path), 'Images in directory must have same file indices!!'
       
        print(f'{utils.Colors.LIGHT_RED}Mounting data on memory...{self.__class__.__name__}:{self.mode}{utils.Colors.END}')
        
        self.img_x = []
        self.img_y = []
        self.img_z = []
        for index in range(len(self.img_x_path)):
            x_path = self.img_x_path[index]
            y_path = self.img_y_path[index]
            # print("****************************")
            # print(x_path, y_path, z_path)
            
            self.img_x.append(Image.open(x_path).convert('RGB'))
            self.img_y.append(Image.open(y_path).convert('L'))
            if self.use_fov:
                z_path = self.img_z_path[index]
                self.img_z.append(Image.open(z_path).convert('L'))

    def transform(self, image, target, fov):
        if fov is None:
            fov = Image.new('L', image.size, color=255)
        if self.mode == 'validation':
            image = utils.center_padding(image, [int(self.args.input_size[0]), int(self.args.input_size[1])])
            target = utils.center_padding(target, [int(self.args.input_size[0]), int(self.args.input_size[1])])
            fov = utils.center_padding(fov, [int(self.args.input_size[0]), int(self.args.input_size[1])])
        
        if not self.mode == 'validation':
            random_gen = random.Random()  # thread-safe random

            if (random_gen.random() < 0.8) and self.args.transform_cutmix:
                rand_n = random_gen.randint(0, self.__len__() - 1)     # randomly generates reference image on dataset
                image_refer = Image.open(self.img_x_path[rand_n]).convert('RGB')
                target_refer = Image.open(self.img_y_path[rand_n]).convert('L')
                fov_refer = Image.open(self.img_z_path[rand_n]).convert('L')
                
                image, target, fov = utils.cut_mix(image, target, fov, image_refer, target_refer, fov_refer)

            if (random_gen.random() < 0.8) and self.args.transform_rand_resize:
                rand_h = (random_gen.random() * 1.5) + 0.5  # [0.5, 2.0]
                rand_w = (random_gen.random() * 1.5) + 0.5
                resize_h = int((self.args.input_size[0] * rand_h).__round__())
                resize_w = int((self.args.input_size[1] * rand_w).__round__())

                image = tf.resize(image, [resize_h, resize_w])
                target = tf.resize(target, [resize_h, resize_w], interpolation=InterpolationMode.NEAREST)
                fov = tf.resize(fov, [resize_h, resize_w], interpolation=InterpolationMode.NEAREST)
                
                
            if hasattr(self.args, 'transform_rand_crop'):
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(int(self.args.transform_rand_crop), int(self.args.transform_rand_crop)))
                image = tf.crop(image, i, j, h, w)
                target = tf.crop(target, i, j, h, w)
                fov = tf.crop(fov, i, j, h, w)

            if (random_gen.random() < 0.5) and self.args.transform_hflip:
                image = tf.hflip(image)
                target = tf.hflip(target)
                fov = tf.hflip(fov)

            if (random_gen.random() < 0.8) and self.args.transform_jitter:
                transform = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                image = transform(image)

            if (random_gen.random() < 0.5) and self.args.transform_blur:
                kernel_size = int((random.random() * 10 + 2.5).__round__())    # random kernel size 3 to 11
                if kernel_size % 2 == 0:
                    kernel_size -= 1
                transform = transforms.GaussianBlur(kernel_size=kernel_size)
                image = transform(image)

            # recommend to use at the end.
            if (random_gen.random() < 0.3) and self.args.transform_perspective:
                start_p, end_p = transforms.RandomPerspective.get_params(image.width, image.height, distortion_scale=0.5)
                image = tf.perspective(image, start_p, end_p)
                target = tf.perspective(target, start_p, end_p, interpolation=InterpolationMode.NEAREST)
                fov = tf.perspective(fov, start_p, end_p, interpolation=InterpolationMode.NEAREST)
                
                
        image_tensor = tf.to_tensor(image)
        target_tensor = torch.tensor(np.array(target))
        fov_tensor = torch.tensor(np.array(fov))    

        if self.args.input_space == 'GR':   # grey, red
            image_tensor_r = image_tensor[0].unsqueeze(0)
            image_tensor_grey = tf.to_tensor(tf.to_grayscale(image))

            image_tensor = torch.cat((image_tensor_r, image_tensor_grey), dim=0)

        # 'mean' and 'std' are acquired by cropped face from sense-time landmark
        if self.args.input_space == 'RGB':
            image_tensor = tf.normalize(image_tensor,
                                        mean=self.image_mean,
                                        std=self.image_std)

        if self.args.n_classes <= 2:  # for visualized binary GT
            target_tensor[target_tensor < 128] = 0
            target_tensor[target_tensor >= 128] = 1
            
        if self.args.n_classes <= 2:  # for visualized binary FOV
            fov_tensor[fov_tensor < 128] = 0
            fov_tensor[fov_tensor >= 128] = 1
        target_tensor = target_tensor.unsqueeze(0)    # expand 'grey channel' for loss function dependency
        fov_tensor = fov_tensor.unsqueeze(0)          # expand 'grey channel' for loss function dependency

        # optional: zero out pixels outside FOV to suppress background
        if hasattr(self.args, 'mask_input_outside_fov') and bool(self.args.mask_input_outside_fov):
            image_tensor = image_tensor * fov_tensor.float()

        return image_tensor, target_tensor, fov_tensor

    def __getitem__(self, index):
        if self.use_fov:
            img_x_tr, img_y_tr, img_z_tr = self.transform(self.img_x[index], self.img_y[index], self.img_z[index])
            img_z_path = self.img_z_path[index]
        else:
            img_x_tr, img_y_tr, img_z_tr = self.transform(self.img_x[index], self.img_y[index], None)
            img_z_path = ''

        return (img_x_tr, self.img_x_path[index]), (img_y_tr, self.img_y_path[index]), (img_z_tr, img_z_path)

    def __len__(self):
        return len(self.img_x_path)


class Image2ImageDataLoader_zero_pad:

    def __init__(self,
                 x_path,
                 y_path,
                 z_path,
                 mode,
                 batch_size=4,
                 num_workers=0,
                 pin_memory=True,
                 **kwargs):

        g = torch.Generator()
        g.manual_seed(3407)
        # print("****************************")
        # print(x_path, y_path, z_path)
        self.image_loader = Image2ImageLoader_zero_pad(x_path,
                                                       y_path,
                                                       z_path,
                                                       mode=mode,
                                                       **kwargs)

        # use your own data loader
        self.Loader = MultiEpochsDataLoader(self.image_loader,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            shuffle=(not mode == 'validation'),
                                            worker_init_fn=seed_worker,
                                            generator=g,
                                            pin_memory=pin_memory)

    def __len__(self):
        return self.image_loader.__len__()


class Image2ImageDataLoader_crop(Dataset):
    
    def __init__(self,
                 x_path,
                 y_path,
                 z_path,
                 mode,
                 **kwargs):

        self.mode = mode
        self.args = kwargs['args']

        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]

        x_img_name = os.listdir(x_path)
        y_img_name = os.listdir(y_path)
        z_img_name = os.listdir(z_path)
        x_img_name = filter(is_image, x_img_name)
        y_img_name = filter(is_image, y_img_name)
        z_img_name = filter(is_image, z_img_name)

        self.img_x_path = []
        self.img_y_path = []
        self.img_z_path = []

        x_img_name = sorted(x_img_name)
        y_img_name = sorted(y_img_name)
        z_img_name = sorted(z_img_name)
        img_paths = zip(x_img_name, y_img_name, z_img_name)
        for item in img_paths:
            # print("****************************")
            self.img_x_path.append(x_path + os.sep + item[0])
            self.img_y_path.append(y_path + os.sep + item[1])
            self.img_z_path.append(z_path + os.sep + item[2])

        assert len(self.img_x_path) == len(self.img_y_path), 'Images in directory must have same file indices!!'
        assert len(self.img_x_path) == len(self.img_z_path), 'Images in directory must have same file indices!!'
       
        print(f'{utils.Colors.LIGHT_RED}Mounting data on memory...{self.__class__.__name__}:{self.mode}{utils.Colors.END}')
        
        self.img_x = []
        self.img_y = []
        self.img_z = []
        
        for index in range(len(self.img_x_path)):
            x_path = self.img_x_path[index]
            y_path = self.img_y_path[index]
            z_path = self.img_z_path[index]
            # print("****************************")
            # print(x_path, y_path, z_path)
            
            self.img_x.append(Image.open(x_path).convert('RGB'))
            self.img_y.append(Image.open(y_path).convert('L'))
            self.img_z.append(Image.open(z_path).convert('L'))

    def transform(self, image, target, fov):
        if self.mode == 'validation':
            image = utils.center_padding(image, [int(self.args.input_size[0]), int(self.args.input_size[1])])
            target = utils.center_padding(target, [int(self.args.input_size[0]), int(self.args.input_size[1])])
            fov = utils.center_padding(fov, [int(self.args.input_size[0]), int(self.args.input_size[1])])
        
        if not self.mode == 'validation':
            random_gen = random.Random()  # thread-safe random

            if (random_gen.random() < 0.8) and self.args.transform_cutmix:
                rand_n = random_gen.randint(0, self.__len__() - 1)     # randomly generates reference image on dataset
                image_refer = Image.open(self.img_x_path[rand_n]).convert('RGB')
                target_refer = Image.open(self.img_y_path[rand_n]).convert('L')
                fov_refer = Image.open(self.img_z_path[rand_n]).convert('L')
                
                image, target, fov = utils.cut_mix(image, target, fov, image_refer, target_refer, fov_refer)

            if (random_gen.random() < 0.8) and self.args.transform_rand_resize:
                rand_h = (random_gen.random() * 1.5) + 0.5  # [0.5, 2.0]
                rand_w = (random_gen.random() * 1.5) + 0.5
                resize_h = int((self.args.input_size[0] * rand_h).__round__())
                resize_w = int((self.args.input_size[1] * rand_w).__round__())

                image = tf.resize(image, [resize_h, resize_w])
                target = tf.resize(target, [resize_h, resize_w], interpolation=InterpolationMode.NEAREST)
                fov = tf.resize(fov, [resize_h, resize_w], interpolation=InterpolationMode.NEAREST)
                
                
            if hasattr(self.args, 'transform_rand_crop'):
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(int(self.args.transform_rand_crop), int(self.args.transform_rand_crop)))
                image = tf.crop(image, i, j, h, w)
                target = tf.crop(target, i, j, h, w)
                fov = tf.crop(fov, i, j, h, w)

            if (random_gen.random() < 0.5) and self.args.transform_hflip:
                image = tf.hflip(image)
                target = tf.hflip(target)
                fov = tf.hflip(fov)

            if (random_gen.random() < 0.8) and self.args.transform_jitter:
                transform = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                image = transform(image)

            if (random_gen.random() < 0.5) and self.args.transform_blur:
                kernel_size = int((random.random() * 10 + 2.5).__round__())    # random kernel size 3 to 11
                if kernel_size % 2 == 0:
                    kernel_size -= 1
                transform = transforms.GaussianBlur(kernel_size=kernel_size)
                image = transform(image)

            # recommend to use at the end.
            if (random_gen.random() < 0.3) and self.args.transform_perspective:
                start_p, end_p = transforms.RandomPerspective.get_params(image.width, image.height, distortion_scale=0.5)
                image = tf.perspective(image, start_p, end_p)
                target = tf.perspective(target, start_p, end_p, interpolation=InterpolationMode.NEAREST)
                fov = tf.perspective(fov, start_p, end_p, interpolation=InterpolationMode.NEAREST)
                
                
        image_tensor = tf.to_tensor(image)
        target_tensor = torch.tensor(np.array(target))
        fov_tensor = torch.tensor(np.array(fov))    

        if self.args.input_space == 'GR':   # grey, red
            image_tensor_r = image_tensor[0].unsqueeze(0)
            image_tensor_grey = tf.to_tensor(tf.to_grayscale(image))

            image_tensor = torch.cat((image_tensor_r, image_tensor_grey), dim=0)

        # 'mean' and 'std' are acquired by cropped face from sense-time landmark
        if self.args.input_space == 'RGB':
            image_tensor = tf.normalize(image_tensor,
                                        mean=self.image_mean,
                                        std=self.image_std)

        if self.args.n_classes <= 2:  # for visualized binary GT
            target_tensor[target_tensor < 128] = 0
            target_tensor[target_tensor >= 128] = 1
            
        if self.args.n_classes <= 2:  # for visualized binary FOV
            fov_tensor[fov_tensor < 128] = 0
            fov_tensor[fov_tensor >= 128] = 1
        target_tensor = target_tensor.unsqueeze(0)    # expand 'grey channel' for loss function dependency
        fov_tensor = fov_tensor.unsqueeze(0)          # expand 'grey channel' for loss function dependency

        # optional: zero out pixels outside FOV to suppress background
        if hasattr(self.args, 'mask_input_outside_fov') and bool(self.args.mask_input_outside_fov):
            image_tensor = image_tensor * fov_tensor.float()

        return image_tensor, target_tensor, fov_tensor


class Image2ImageDataLoader_crop:
    
    def __init__(self,
                 x_path,
                 y_path,
                 z_path,
                 mode,
                 batch_size=4, 
                 num_workers=0,
                 pin_memory=True,
                 **kwargs):
        g = torch.Generator()
        g.manual_seed(3407)
        self.image_loader = Image2ImageLoader_crop(x_path,
                                                   y_path,
                                                   z_path,
                                                   mode=mode,
                                                   **kwargs)
        # use your own data loader
        self.Loader = MultiEpochsDataLoader(self.image_loader,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            shuffle=(not mode == 'validation'),
                                            worker_init_fn=seed_worker,
                                            generator=g,
                                            pin_memory=pin_memory)
    def __len__(self):
        return self.image_loader.__len__()