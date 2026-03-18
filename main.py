# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2025-12-15
@Author  :   niuliang 
@Version :   1.0
@Contact :   niouleung@gmail.com
'''

import torch
import os
import argparse
import wandb
import yaml
import numpy as np
import random
import ast

from train import Trainer_seg
from inference import Inferencer
from torch.cuda import is_available
from datetime import datetime


# fix seed for reproducibility
seed = 3407
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)  # raise error if CUDA >= 10.2
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


def conf_to_args(args, **kwargs):
    var = vars(args)

    for key, value in kwargs.items():
        var[key] = value


def _is_auto_value(value):
    if isinstance(value, str):
        return value.strip().lower() == 'auto'
    return value is None


DATASET_PROFILES = {
    'DRIVE': {
        'base_dir': 'DRIVE',
        'train_split': 'train',
        'eval_split': 'val',
        'input_size': [608, 608], ## 565 584
        'transform_rand_crop': 288,
    },
    'STARE': {
        'base_dir': 'STARE',
        'train_split': 'train',
        'eval_split': 'val',
        'input_size': [704, 704],
        'transform_rand_crop': 288,
    },
    'CHASE_DB1': {
        'base_dir': 'CHASE_DB1',
        'train_split': 'train',
        'eval_split': 'val',
        'input_size': [1024, 1024], #[999, 960]
        'transform_rand_crop': 448,  #448
    },
    'HRF': {
        'base_dir': 'HRF',
        'train_split': 'train',
        'eval_split': 'val',
        'input_size': [1344, 1344],
        'transform_rand_crop': 288,
    },

    'OCTA500_3MM': {
        'base_dir': 'OCTA500/3mm',
        'train_split': 'train',
        'eval_split': 'val',
        'input_size': [304, 304],
        'transform_rand_crop': 128,
    },
    'OCTA500_6MM': {
        'base_dir': 'OCTA500/6mm',
        'train_split': 'train',
        'eval_split': 'val',
        'input_size': [400, 400],
        'transform_rand_crop': 128,
    },
    'OCTA500_Tang': {
        'base_dir': 'OCTA500/tang',
        'train_split': 'train',
        'eval_split': 'val',
        'input_size': [400, 400],
        'transform_rand_crop': 128,
    },
    'DAC1': {
        'base_dir': 'DAC1',
        'train_split': 'train',
        'eval_split': 'val',
        'input_size': [320, 320],
        'transform_rand_crop': 128,
    },
    
    'RVD': {
        'base_dir': 'RVD',
        'train_split': 'train',
        'eval_split': 'test',
        'input_size': [1024, 1024],
        'transform_rand_crop': 448,
    },
}


DATASET_NAME_ALIASES = {

    'OCTA500_3MM': 'OCTA500_3MM',
    'OCTA500/3MM': 'OCTA500_3MM',
    'OCTA5003MM': 'OCTA500_3MM',
    'OCTA500_6MM': 'OCTA500_6MM',
    'OCTA500/6MM': 'OCTA500_6MM',
    'OCTA5006MM': 'OCTA500_6MM',
    'DRIVE': 'DRIVE',
    'CHASE_DB1': 'CHASE_DB1',
    'DAC1': 'DAC1',

}


def _normalize_dataset_name(dataset_name):
    if not isinstance(dataset_name, str):
        return None
    normalized = dataset_name.strip()
    if not normalized:
        return None
    if normalized in DATASET_PROFILES:
        return normalized
    return DATASET_NAME_ALIASES.get(normalized.upper())


def _infer_dataset_name_from_path(path_hint: str):
    if not path_hint:
        return None

    lower_path = path_hint.lower()

    if 'drive' in lower_path:
        return 'DRIVE'
    if 'chase_db1' in lower_path:
        return 'CHASE_DB1'
    if 'dac1' in lower_path:
        return 'DAC1'
    if 'octa500' in lower_path and '3mm' in lower_path:
        return 'OCTA500_3MM'
    if 'octa500' in lower_path and '6mm' in lower_path:
        return 'OCTA500_6MM'
    
    return None


def _build_split_paths(data_root, base_dir, split_name):
    split_dir = os.path.join(data_root, base_dir, split_name)
    return {
        'x': os.path.join(split_dir, 'input'),
        'y': os.path.join(split_dir, 'label'),
        'z': os.path.join(split_dir, 'fov'),
    }


def apply_auto_dataset_params(args):
    dataset_name = getattr(args, 'dataset_name', None)
    if isinstance(dataset_name, str) and dataset_name.strip():
        dataset_name = dataset_name.strip().upper()
    else:
        dataset_name = None

    if dataset_name is None:
        path_candidates = []
        for path_key in ['train_x_path', 'val_x_path', 'train_y_path', 'val_y_path']:
            if hasattr(args, path_key):
                path_candidates.append(getattr(args, path_key))

        for path_candidate in path_candidates:
            dataset_name = _infer_dataset_name_from_path(path_candidate)
            if dataset_name is not None:
                break

    if dataset_name is None or dataset_name not in DATASET_PROFILES:
        return

    profile = DATASET_PROFILES[dataset_name]
    data_root = getattr(args, 'data_root', './data')
    train_split = getattr(args, 'train_split', profile['train_split'])
    eval_split = getattr(args, 'eval_split', profile['eval_split'])

    train_paths = _build_split_paths(data_root, profile['base_dir'], train_split)
    eval_paths = _build_split_paths(data_root, profile['base_dir'], eval_split)

    if hasattr(args, 'train_x_path') and _is_auto_value(getattr(args, 'train_x_path')):
        args.train_x_path = train_paths['x']
    if hasattr(args, 'train_y_path') and _is_auto_value(getattr(args, 'train_y_path')):
        args.train_y_path = train_paths['y']
    if hasattr(args, 'train_z_path') and _is_auto_value(getattr(args, 'train_z_path')):
        args.train_z_path = train_paths['z']

    if hasattr(args, 'val_x_path') and _is_auto_value(getattr(args, 'val_x_path')):
        args.val_x_path = eval_paths['x']
    if hasattr(args, 'val_y_path') and _is_auto_value(getattr(args, 'val_y_path')):
        args.val_y_path = eval_paths['y']
    if hasattr(args, 'val_z_path') and _is_auto_value(getattr(args, 'val_z_path')):
        args.val_z_path = eval_paths['z']

    if (not hasattr(args, 'input_size')) or _is_auto_value(getattr(args, 'input_size')):
        args.input_size = profile['input_size']

    if hasattr(args, 'transform_rand_crop') and _is_auto_value(getattr(args, 'transform_rand_crop')):
        args.transform_rand_crop = profile['transform_rand_crop']

    args.dataset_name = dataset_name
    print(
        f"[AUTO-CONFIG] dataset={dataset_name}, input_size={args.input_size}, "
        f"transform_rand_crop={getattr(args, 'transform_rand_crop', 'N/A')}, "
        f"train_x_path={getattr(args, 'train_x_path', 'N/A')}, "
        f"val_x_path={getattr(args, 'val_x_path', 'N/A')}"
    )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default=None, type=str)
    arg, unknown_arg = parser.parse_known_args()

    if arg.config_path is not None:
        with open(arg.config_path, 'rb') as f:
            conf = yaml.load(f.read(), Loader=yaml.Loader)  # load the config file
            conf['config_path'] = arg.config_path
    else:
        # make unrecognized args to dict
        conf = {'config_path': 'configs/sweep_config.yaml'}
        for item in unknown_arg:
            item = item.strip('--')
            key, value = item.split('=')
            if key != 'CUDA_VISIBLE_DEVICES':
                try:
                    if value == 'true' or value == 'false':
                        value = value.title()
                    value = ast.literal_eval(value)
                except ValueError:
                    if value.isalpha(): pass
                except SyntaxError as e:
                    if '/' in value: pass
                    else: raise e
            conf[key] = value

    args = argparse.Namespace()
    conf_to_args(args, **conf)  # pass in keyword args
    apply_auto_dataset_params(args)

    now_time = datetime.now().strftime("%Y-%m-%d %H%M%S")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES

    if args.debug:
        args.wandb = False

    print('Use CUDA :', args.cuda and is_available())
    if args.mode in 'train':
        if args.mode == 'train':
            if args.task == 'segmentation':
                print("**********************************")
                print(args.train_x_path)
                trainer = Trainer_seg(args, now_time)
        else:
            raise Exception('Invalid mode')

        trainer.start_train()

    elif args.mode in 'inference':
        inferencer = Inferencer(args)

        if args.inference_mode == 'segmentation':
            inferencer.start_inference_segmentation()
        else:
            raise ValueError('Please select correct inference_mode !!!')
    else:
        print('No mode supported.')


if __name__ == "__main__":
    main()