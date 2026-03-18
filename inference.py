# -*- encoding: utf-8 -*-
'''
@File    :   inference.py
@Time    :   2025-12-15
@Author  :   niuliang 
@Version :   1.0
@Contact :   niouleung@gmail.com
'''


import torch
import time
import numpy as np
import os

from models import metrics
from models import utils
from models import dataloader as dataloader_hub
from models import model_implements
from train import Trainer_seg
from PIL import Image


class Inferencer:
    def __init__(self, args):
        self.start_time = time.time()
        self.args = args

        use_cuda = self.args.cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        val_z_path = self.args.val_z_path if getattr(self.args, 'use_fov', False) else None
        self.loader_form = self.__init_data_loader(self.args.val_x_path,
                       self.args.val_y_path,
                       val_z_path,
                       batch_size=1,
                       mode='validation')

        self.model = Trainer_seg.init_model(self.args.model_name, self.device, self.args)
        self.model.load_state_dict(torch.load(args.model_path))
        self.model.eval()

        self.metric = self._init_metric(self.args.task, self.args.n_classes)

        self.image_mean = self.loader_form.image_loader.image_mean
        self.image_std = self.loader_form.image_loader.image_std
        self.fn_list = []

    def start_inference_segmentation(self):
        f1_list = []
        acc_list = []
        auc_list = []
        sp_list = []
        sen_list = []
        mcc_list = []
        hd95_list = []
        c_list = []
        a_list = []
        l_list = []
        f_list = []

        for batch_idx, (img, target, fov) in enumerate(self.loader_form.Loader):
            with torch.no_grad():
                x_in, img_id = img
                target, origin_size = target
                fov, _ = fov

                x_in = x_in.to(self.device)
                x_img = x_in
                target = target.long().to(self.device)
                if getattr(self.args, 'use_fov', False):
                    fov = fov.float().to(self.device)

                output = self.model(x_in)

                if isinstance(output, (tuple, list)):
                    outputs = output
                else:
                    outputs = [output]

                for i, out_item in enumerate(outputs):
                    metric_result = self.post_process(out_item, target, fov, x_img, img_id, i)

                # if isinstance(output, tuple) or isinstance(output, list):  # condition for deep supervision
                #     output = output[0]

                # metric_result = self.post_process(output, target, x_img, img_id)
                f1_list.append(metric_result['f1'])
                acc_list.append(metric_result['acc'])
                auc_list.append(metric_result['auc'])
                sp_list.append(metric_result['sp'])
                sen_list.append(metric_result['sen'])
                mcc_list.append(metric_result['mcc'])
                hd95_list.append(metric_result['hd95'])
                c_list.append(metric_result['c'])
                a_list.append(metric_result['a'])
                l_list.append(metric_result['l'])
                f_list.append(metric_result['f'])

        metrics = self.metric.get_results()
        fg_class_iou = [metrics['Class IoU'][i] for i in range(1, self.args.n_classes + 1)]
        mIoU = sum(fg_class_iou) / len(fg_class_iou)

        print('ACC', sum(acc_list) / len(acc_list))
        print('SE', sum(sen_list) / len(sen_list))
        print('SP', sum(sp_list) / len(sp_list))
        print('F1', sum(f1_list) / len(f1_list))
        print('MIoU', mIoU)
        print('HD95', sum(hd95_list) / len(hd95_list))
        print('AUC', sum(auc_list) / len(auc_list))
        print('C', sum(c_list) / len(c_list))
        print('A', sum(a_list) / len(a_list))
        print('L', sum(l_list) / len(l_list))
        print('F', sum(f_list) / len(f_list))

    def post_process(self, output, target, fov, x_img, img_id, idx):
        # reconstruct original image
        x_img = x_img.squeeze(0).data.cpu().numpy()
        x_img = np.transpose(x_img, (1, 2, 0))
        x_img = x_img * np.array(self.image_std)
        x_img = x_img + np.array(self.image_mean)
        x_img = x_img * 255.0
        x_img = x_img.astype(np.uint8)

        # output = utils.remove_center_padding(output)
        # target = utils.remove_center_padding(target)

        output_argmax = torch.where(output > 0.5, 1, 0)
        if getattr(self.args, 'use_fov', False):
            output_masked = (output_argmax[:, 0] * fov.squeeze(1)).cpu()
            target_masked = (target.squeeze(0) * fov.squeeze(1)).cpu().detach().numpy()
        else:
            output_masked = output_argmax[:, 0].cpu()
            target_masked = target.squeeze(0).cpu().detach().numpy()
        self.metric.update(target_masked, output_masked.numpy())

        path, fn = os.path.split(img_id[0])
        img_id, ext = os.path.splitext(fn)
        dir_path, fn = os.path.split(self.args.model_path)
        fn, ext = os.path.splitext(fn)
        save_dir = dir_path + '/' + fn + '/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        img_id = img_id + '_' + str(2 - idx)
        # Image.fromarray(x_img).save(save_dir + img_id + '.png', quality=100)
        output_np = (output_masked.squeeze().numpy() * 255).astype(np.uint8)
        target_np = (target_masked.squeeze() * 255).astype(np.uint8)
        output_img = Image.fromarray(output_np)
        target_img = Image.fromarray(target_np)

        # Export with explicit (width, height) so output size follows args.input_size directly.
        if hasattr(self.args, 'input_size') and len(self.args.input_size) == 2:
            export_w = int(self.args.input_size[0])
            export_h = int(self.args.input_size[1])
            if output_img.size != (export_w, export_h):
                output_img = output_img.resize((export_w, export_h), resample=Image.NEAREST)
                target_img = target_img.resize((export_w, export_h), resample=Image.NEAREST)

        output_img.save(save_dir + img_id + f'_argmax.png', quality=100)
        target_img.save(save_dir + img_id + f'_target.png', quality=100)
        # Image.fromarray(output_heatmap.astype(np.uint8)).save(save_dir + img_id + f'_heatmap_overlay.png', quality=100)

        metric_result = metrics.metrics_np(output_masked[None, :], target_masked, b_auc=True, b_hd95=True)
        print(f'{img_id} \t Done !!')

        return metric_result

    def __init_model(self, model_name):
        if model_name == 'LPLSNet':
            model = model_implements.LPLSNet(n_classes=1, in_channels=self.args.input_channel)
        else:
            raise Exception('No model named', model_name)

        return torch.nn.DataParallel(model)

    def __init_data_loader(self,
                           x_path,
                           y_path,
                           z_path,
                           batch_size,
                           mode):

        if self.args.dataloader == 'Image2Image_zero_pad':
            loader = dataloader_hub.Image2ImageDataLoader_zero_pad(x_path=x_path,
                                                                   y_path=y_path,
                                                                   z_path=z_path,
                                                                   batch_size=batch_size,
                                                                   num_workers=self.args.worker,
                                                                   pin_memory=self.args.pin_memory,
                                                                   mode=mode,
                                                                   args=self.args)
        elif self.args.dataloader == 'Image2Image_resize':
            loader = dataloader_hub.Image2ImageDataLoader_resize(x_path=x_path,
                                                                 y_path=y_path,
                                                                 z_path=z_path,
                                                                 batch_size=batch_size,
                                                                 num_workers=self.args.worker,
                                                                 pin_memory=self.args.pin_memory,
                                                                 mode=mode,
                                                                 args=self.args)
        else:
            raise Exception('No dataloader named', self.args.dataloader)

        return loader

    def _init_metric(self, task_name, num_class):
        if task_name == 'segmentation':
            metric = metrics.StreamSegMetrics_segmentation(num_class + 1)
        else:
            raise Exception('No task named', task_name)

        return metric
