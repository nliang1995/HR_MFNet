# -*- encoding: utf-8 -*-
'''
@File    :   metrics.py
@Time    :   2025-12-15
@Author  :   niuliang 
@Version :   1.0
@Contact :   niouleung@gmail.com
'''


import math
import numpy as np
import cv2
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt, label as ndi_label
try:
    from skimage.morphology import skeletonize
except Exception:
    skeletonize = None

from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, confusion_matrix


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()


class StreamSegMetrics_segmentation(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.metric_dict = {
            "Overall Acc": 0,
            "Mean Acc": 0,
            "FreqW Acc": 0,
            "Mean IoU": 0,
            "Class IoU": 0
        }

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU":
                string += "%s: %f\n" % (k, v)

        # string+='Class IoU:\n'
        # for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean iou
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iou = np.nanmean(iou)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        cls_iou = dict(zip(range(self.n_classes), iou))

        self.metric_dict['Overall Acc'] = acc
        self.metric_dict['Mean Acc'] = acc_cls
        self.metric_dict['FreqW Acc'] = fwavacc
        self.metric_dict['Mean IoU'] = mean_iou
        self.metric_dict['Class IoU'] = cls_iou

        return self.metric_dict

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


def _hd95_binary(pred_binary, gt_binary):
    pred_binary = pred_binary.astype(bool)
    gt_binary = gt_binary.astype(bool)

    if (not np.any(pred_binary)) and (not np.any(gt_binary)):
        return 0.0
    if (not np.any(pred_binary)) or (not np.any(gt_binary)):
        return float('inf')

    pred_border = np.logical_xor(pred_binary, binary_erosion(pred_binary))
    gt_border = np.logical_xor(gt_binary, binary_erosion(gt_binary))

    if not np.any(pred_border):
        pred_border = pred_binary
    if not np.any(gt_border):
        gt_border = gt_binary

    dt_gt = distance_transform_edt(~gt_border)
    dt_pred = distance_transform_edt(~pred_border)

    d_pred_to_gt = dt_gt[pred_border]
    d_gt_to_pred = dt_pred[gt_border]
    all_dist = np.concatenate([d_pred_to_gt, d_gt_to_pred])

    if all_dist.size == 0:
        return 0.0

    return float(np.percentile(all_dist, 95))


def _connectivity_binary(pred_binary, gt_binary):
    pred_binary = pred_binary.astype(bool)
    gt_binary = gt_binary.astype(bool)

    cardinality_gt = int(np.sum(gt_binary))
    if cardinality_gt == 0:
        return 1.0 if np.sum(pred_binary) == 0 else 0.0

    num_components_gt = np.max(ndi_label(gt_binary)[0])
    num_components_pred = np.max(ndi_label(pred_binary)[0])
    return 1.0 - min(1.0, abs(num_components_gt - num_components_pred) / float(cardinality_gt))


def _area_binary(pred_binary, gt_binary, dilation_radius=2):
    pred_binary = pred_binary.astype(bool)
    gt_binary = gt_binary.astype(bool)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * dilation_radius + 1, 2 * dilation_radius + 1),
    )
    pred_dilated = cv2.dilate(pred_binary.astype(np.uint8), kernel).astype(bool)
    gt_dilated = cv2.dilate(gt_binary.astype(np.uint8), kernel).astype(bool)

    intersection_left = np.logical_and(pred_binary, gt_dilated)
    intersection_right = np.logical_and(gt_binary, pred_dilated)
    inter = np.logical_or(intersection_left, intersection_right)
    union = np.logical_or(pred_binary, gt_binary)

    union_sum = np.sum(union)
    if union_sum == 0:
        return 1.0
    return float(np.sum(inter) / union_sum)


def _length_binary(pred_binary, gt_binary, dilation_radius=2):
    pred_binary = pred_binary.astype(bool)
    gt_binary = gt_binary.astype(bool)

    if skeletonize is not None:
        pred_skeleton = skeletonize(pred_binary)
        gt_skeleton = skeletonize(gt_binary)
    else:
        pred_skeleton = pred_binary
        gt_skeleton = gt_binary

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * dilation_radius + 1, 2 * dilation_radius + 1),
    )
    pred_dilated = cv2.dilate(pred_binary.astype(np.uint8), kernel).astype(bool)
    gt_dilated = cv2.dilate(gt_binary.astype(np.uint8), kernel).astype(bool)

    intersection_left = np.logical_and(pred_skeleton, gt_dilated)
    intersection_right = np.logical_and(gt_skeleton, pred_dilated)
    inter = np.logical_or(intersection_left, intersection_right)
    union = np.logical_or(pred_skeleton, gt_skeleton)

    union_sum = np.sum(union)
    if union_sum == 0:
        return 1.0
    return float(np.sum(inter) / union_sum)


def metrics_np(np_res, np_gnd, b_auc=False, b_hd95=False):
    f1m = []
    accm = []
    aucm = []
    spm = []
    sensitivitym = []
    ioum = []
    mccm = []
    hd95m = []
    cm = []
    am = []
    lm = []
    fm = []

    epsilon = 2.22045e-16

    for i in range(np_res.shape[0]):
        label_2d = np.asarray(np_gnd[i, :, :])
        pred_2d = np.asarray(np_res[i, :, :])
        label = label_2d.flatten()
        pred = pred_2d.flatten()

        y_pred = np.zeros_like(pred)
        y_pred[pred > 0.5] = 1

        try:
            # Explicitly specify binary labels to ensure a 2x2 confusion matrix shape
            tn, fp, fn, tp = confusion_matrix(y_true=label, y_pred=y_pred, labels=[0, 1]).ravel()
        except ValueError:
            tn, fp, fn, tp = 0, 0, 0, 0
        accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
        sensitivity = tp / (tp + fn + epsilon)  # Recall
        specificity = tn / (tn + fp + epsilon)
        precision = tp / (tp + fp + epsilon)
        f1_score = (2 * sensitivity * precision) / (sensitivity + precision + epsilon)
        iou = tp / (tp + fp + fn + epsilon)

        tp_tmp, tn_tmp, fp_tmp, fn_tmp = tp / 1000, tn / 1000, fp / 1000, fn / 1000     # to prevent overflowing
        mcc = (tp_tmp * tn_tmp - fp_tmp * fn_tmp) / math.sqrt((tp_tmp + fp_tmp) * (tp_tmp + fn_tmp) * (tn_tmp + fp_tmp) * (tn_tmp + fn_tmp) + epsilon)  # Matthews correlation coefficient

        f1m.append(f1_score)
        accm.append(accuracy)
        spm.append(specificity)
        sensitivitym.append(sensitivity)
        ioum.append(iou)
        mccm.append(mcc)
        if b_auc:
            auc = roc_auc_score(sorted(label), sorted(y_pred))
            aucm.append(auc)
        if b_hd95:
            y_pred_2d = y_pred.reshape(label_2d.shape)
            hd95 = _hd95_binary(y_pred_2d, label_2d)
            c_val = _connectivity_binary(y_pred_2d, label_2d)
            a_val = _area_binary(y_pred_2d, label_2d)
            l_val = _length_binary(y_pred_2d, label_2d)

            hd95m.append(hd95)
            cm.append(c_val)
            am.append(a_val)
            lm.append(l_val)
            fm.append(c_val * a_val * l_val)

    output = dict()
    output['f1'] = np.array(f1m).mean()
    output['acc'] = np.array(accm).mean()
    output['sp'] = np.array(spm).mean()
    output['sen'] = np.array(sensitivitym).mean()
    output['iou'] = np.array(ioum).mean()
    output['mcc'] = np.array(mccm).mean()

    if b_auc:
        output['auc'] = np.array(aucm).mean()
    if b_hd95:
        output['hd95'] = np.array(hd95m).mean()
        output['hd'] = np.array(hd95m).mean()
        output['c'] = np.array(cm).mean()
        output['a'] = np.array(am).mean()
        output['l'] = np.array(lm).mean()
        output['f'] = np.array(fm).mean() if len(fm) > 0 else 0.0

    return output
