# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import numpy as np
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
from functools import lru_cache
import torch
from torch.distributions.weibull import Weibull
from torch.distributions.transforms import AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution
from fvcore.common.file_io import PathManager

from detectron2.data import MetadataCatalog
from detectron2.utils import comm

from .evaluator import DatasetEvaluator

np.set_printoptions(threshold=sys.maxsize)

class PascalVOCDetectionEvaluator(DatasetEvaluator):
    """
    Evaluate Pascal VOC style AP for Pascal VOC dataset.
    It contains a synchronization, therefore has to be called from all ranks.

    Note that the concept of AP can be implemented in different ways and may not
    produce identical results. This class mimics the implementation of the official
    Pascal VOC Matlab API, and should produce similar but not identical results to the
    official API.
    """

    def __init__(self, dataset_name, cfg=None):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)
        self._anno_file_template = os.path.join(meta.dirname, "Annotations", "{}.xml")
        self._image_set_path = os.path.join(meta.dirname, "ImageSets", "Main", meta.split + ".txt")
        self._class_names = meta.thing_classes
        assert meta.year in [2007, 2012], meta.year
        self._is_2007 = False
        # self._is_2007 = meta.year == 2007
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        if cfg is not None:
            self.prev_intro_cls = cfg.OWOD.PREV_INTRODUCED_CLS
            self.curr_intro_cls = cfg.OWOD.CUR_INTRODUCED_CLS
            self.total_num_class = cfg.MODEL.ROI_HEADS.NUM_CLASSES
            self.unknown_class_index = self.total_num_class - 1
            self.num_seen_classes = self.prev_intro_cls + self.curr_intro_cls
            self.known_classes = self._class_names[:self.num_seen_classes]

            param_save_location = os.path.join(cfg.OUTPUT_DIR,'energy_dist_' + str(self.num_seen_classes) + '.pkl')
            self.energy_distribution_loaded = False
            if os.path.isfile(param_save_location) and os.access(param_save_location, os.R_OK):
                self._logger.info('Loading energy distribution from ' + param_save_location)
                params = torch.load(param_save_location)
                unknown = params[0]
                known = params[1]
                self.unk_dist = self.create_distribution(unknown['scale_unk'], unknown['shape_unk'], unknown['shift_unk'])
                self.known_dist = self.create_distribution(known['scale_known'], known['shape_known'], known['shift_known'])
                self.energy_distribution_loaded = True
            else:
                self._logger.info('Energy distribution is not found at ' + param_save_location)


    def create_distribution(self, scale, shape, shift):
        wd = Weibull(scale=scale, concentration=shape)
        transforms = AffineTransform(loc=shift, scale=1.)
        weibull = TransformedDistribution(wd, transforms)
        return weibull

    def compute_prob(self, x, distribution):
        eps_radius = 0.5
        num_eval_points = 100
        start_x = x - eps_radius
        end_x = x + eps_radius
        step = (end_x - start_x) / num_eval_points
        dx = torch.linspace(x - eps_radius, x + eps_radius, num_eval_points)
        pdf = distribution.log_prob(dx).exp()
        prob = torch.sum(pdf * step)
        return prob

    def reset(self):
        self._predictions = defaultdict(list)  # class name -> list of prediction strings

    def update_label_based_on_energy(self, logits, classes):
        if not self.energy_distribution_loaded:
            return classes
        else:
            cls = classes
            lse = torch.logsumexp(logits[:, :self.num_seen_classes], dim=1)
            for i, energy in enumerate(lse):
                p_unk = self.compute_prob(energy, self.unk_dist)
                p_known = self.compute_prob(energy, self.known_dist)
                if torch.isnan(p_unk) or torch.isnan(p_known):
                    continue
                if p_unk <= p_known:
                    if cls[i] == self.unknown_class_index:
                        cls[i] = -100
                else:
                    if cls[i] != self.unknown_class_index:
                        cls[i] = self.unknown_class_index
            return cls


    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            logits = instances.logits
            classes = self.update_label_based_on_energy(logits, classes)
            for box, score, cls in zip(boxes, scores, classes):
                if cls == -100:
                    continue
                xmin, ymin, xmax, ymax = box
                # The inverse of data loading logic in `datasets/pascal_voc.py`
                xmin += 1
                ymin += 1
                self._predictions[cls].append(
                    f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                )

    def compute_avg_precision_at_many_recall_level_for_unk(self, precisions, recalls):
        precs = {}
        for r in range(1, 10):
            r = r/10
            p = self.compute_avg_precision_at_a_recall_level_for_unk(precisions, recalls, recall_level=r)
            precs[r] = p
        return precs

    def compute_avg_precision_at_a_recall_level_for_unk(self, precisions, recalls, recall_level=0.5):
        precs = {}
        for iou, recall in recalls.items():
            prec = []
            for cls_id, rec in enumerate(recall):
                if cls_id == self.unknown_class_index and len(rec)>0:
                    p = precisions[iou][cls_id][min(range(len(rec)), key=lambda i: abs(rec[i] - recall_level))]
                    prec.append(p)
            if len(prec) > 0:
                precs[iou] = np.mean(prec)
            else:
                precs[iou] = 0
        return precs

    def compute_WI_at_many_recall_level(self, recalls, tp_plus_fp_cs, fp_os):
        wi_at_recall = {}
        for r in range(1, 10):
            r = r/10
            wi = self.compute_WI_at_a_recall_level(recalls, tp_plus_fp_cs, fp_os, recall_level=r)
            wi_at_recall[r] = wi
        return wi_at_recall

    def compute_WI_at_a_recall_level(self, recalls, tp_plus_fp_cs, fp_os, recall_level=0.5):
        wi_at_iou = {}
        for iou, recall in recalls.items():
            tp_plus_fps = []
            fps = []
            for cls_id, rec in enumerate(recall):
                if cls_id in range(self.num_seen_classes) and len(rec) > 0:
                    index = min(range(len(rec)), key=lambda i: abs(rec[i] - recall_level))
                    tp_plus_fp = tp_plus_fp_cs[iou][cls_id][index]
                    tp_plus_fps.append(tp_plus_fp)
                    fp = fp_os[iou][cls_id][index]
                    fps.append(fp)
            if len(tp_plus_fps) > 0:
                wi_at_iou[iou] = np.mean(fps) / np.mean(tp_plus_fps)
            else:
                wi_at_iou[iou] = 0
        return wi_at_iou

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        self._logger.info(
            "Evaluating {} using {} metric. "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name, 2007 if self._is_2007 else 2012
            )
        )

        with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")

            aps = defaultdict(list)  # iou -> ap per class
            recs = defaultdict(list)
            precs = defaultdict(list)
            all_recs = defaultdict(list)
            all_precs = defaultdict(list)
            unk_det_as_knowns = defaultdict(list)
            num_unks = defaultdict(list)
            tp_plus_fp_cs = defaultdict(list)
            fp_os = defaultdict(list)

            for cls_id, cls_name in enumerate(self._class_names):
                lines = predictions.get(cls_id, [""])
                self._logger.info(cls_name + " has " + str(len(lines)) + " predictions.")
                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))

                # for thresh in range(50, 100, 5):
                thresh = 50
                rec, prec, ap, unk_det_as_known, num_unk, tp_plus_fp_closed_set, fp_open_set = voc_eval(
                    res_file_template,
                    self._anno_file_template,
                    self._image_set_path,
                    cls_name,
                    ovthresh=thresh / 100.0,
                    use_07_metric=self._is_2007,
                    known_classes=self.known_classes
                )
                aps[thresh].append(ap * 100)
                unk_det_as_knowns[thresh].append(unk_det_as_known)
                num_unks[thresh].append(num_unk)
                all_precs[thresh].append(prec)
                all_recs[thresh].append(rec)
                tp_plus_fp_cs[thresh].append(tp_plus_fp_closed_set)
                fp_os[thresh].append(fp_open_set)
                try:
                    recs[thresh].append(rec[-1] * 100)
                    precs[thresh].append(prec[-1] * 100)
                except:
                    recs[thresh].append(0)
                    precs[thresh].append(0)

        wi = self.compute_WI_at_many_recall_level(all_recs, tp_plus_fp_cs, fp_os)
        self._logger.info('Wilderness Impact: ' + str(wi))

        avg_precision_unk = self.compute_avg_precision_at_many_recall_level_for_unk(all_precs, all_recs)
        self._logger.info('avg_precision: ' + str(avg_precision_unk))

        ret = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50]}

        total_num_unk_det_as_known = {iou: np.sum(x) for iou, x in unk_det_as_knowns.items()}
        total_num_unk = num_unks[50][0]
        self._logger.info('Absolute OSE (total_num_unk_det_as_known): ' + str(total_num_unk_det_as_known))
        self._logger.info('total_num_unk ' + str(total_num_unk))

        # Extra logging of class-wise APs
        avg_precs = list(np.mean([x for _, x in aps.items()], axis=0))
        self._logger.info(self._class_names)
        # self._logger.info("AP__: " + str(['%.1f' % x for x in avg_precs]))
        self._logger.info("AP50: " + str(['%.1f' % x for x in aps[50]]))
        self._logger.info("Precisions50: " + str(['%.1f' % x for x in precs[50]]))
        self._logger.info("Recall50: " + str(['%.1f' % x for x in recs[50]]))
        # self._logger.info("AP75: " + str(['%.1f' % x for x in aps[75]]))

        if self.prev_intro_cls > 0:
            # self._logger.info("\nPrev class AP__: " + str(np.mean(avg_precs[:self.prev_intro_cls])))
            self._logger.info("Prev class AP50: " + str(np.mean(aps[50][:self.prev_intro_cls])))
            self._logger.info("Prev class Precisions50: " + str(np.mean(precs[50][:self.prev_intro_cls])))
            self._logger.info("Prev class Recall50: " + str(np.mean(recs[50][:self.prev_intro_cls])))

            # self._logger.info("Prev class AP75: " + str(np.mean(aps[75][:self.prev_intro_cls])))

        # self._logger.info("\nCurrent class AP__: " + str(np.mean(avg_precs[self.prev_intro_cls:self.curr_intro_cls])))
        self._logger.info("Current class AP50: " + str(np.mean(aps[50][self.prev_intro_cls:self.prev_intro_cls + self.curr_intro_cls])))
        self._logger.info("Current class Precisions50: " + str(np.mean(precs[50][self.prev_intro_cls:self.prev_intro_cls + self.curr_intro_cls])))
        self._logger.info("Current class Recall50: " + str(np.mean(recs[50][self.prev_intro_cls:self.prev_intro_cls + self.curr_intro_cls])))
        # self._logger.info("Current class AP75: " + str(np.mean(aps[75][self.prev_intro_cls:self.curr_intro_cls])))

        # self._logger.info("\nKnown AP__: " + str(np.mean(avg_precs[:self.prev_intro_cls + self.curr_intro_cls])))
        self._logger.info("Known AP50: " + str(np.mean(aps[50][:self.prev_intro_cls + self.curr_intro_cls])))
        self._logger.info("Known Precisions50: " + str(np.mean(precs[50][:self.prev_intro_cls + self.curr_intro_cls])))
        self._logger.info("Known Recall50: " + str(np.mean(recs[50][:self.prev_intro_cls + self.curr_intro_cls])))
        # self._logger.info("Known AP75: " + str(np.mean(aps[75][:self.prev_intro_cls + self.curr_intro_cls])))

        # self._logger.info("\nUnknown AP__: " + str(avg_precs[-1]))
        self._logger.info("Unknown AP50: " + str(aps[50][-1]))
        self._logger.info("Unknown Precisions50: " + str(precs[50][-1]))
        self._logger.info("Unknown Recall50: " + str(recs[50][-1]))
        # self._logger.info("Unknown AP75: " + str(aps[75][-1]))

        # self._logger.info("R__: " + str(['%.1f' % x for x in list(np.mean([x for _, x in recs.items()], axis=0))]))
        # self._logger.info("R50: " + str(['%.1f' % x for x in recs[50]]))
        # self._logger.info("R75: " + str(['%.1f' % x for x in recs[75]]))
        #
        # self._logger.info("P__: " + str(['%.1f' % x for x in list(np.mean([x for _, x in precs.items()], axis=0))]))
        # self._logger.info("P50: " + str(['%.1f' % x for x in precs[50]]))
        # self._logger.info("P75: " + str(['%.1f' % x for x in precs[75]]))

        return ret


##############################################################################
#
# Below code is modified from
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

"""Python implementation of the PASCAL VOC devkit's AP evaluation code."""


@lru_cache(maxsize=None)
def parse_rec(filename, known_classes):
    """Parse a PASCAL VOC xml file."""
    VOC_CLASS_NAMES_COCOFIED = [
        "airplane", "dining table", "motorcycle",
        "potted plant", "couch", "tv"
    ]
    BASE_VOC_CLASS_NAMES = [
        "aeroplane", "diningtable", "motorbike",
        "pottedplant", "sofa", "tvmonitor"
    ]
    try:
        with PathManager.open(filename) as f:
            tree = ET.parse(f)
    except:
        logger = logging.getLogger(__name__)
        logger.info('Not able to load: ' + filename + '. Continuing without aboarting...')
        return None

    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        cls_name = obj.find("name").text
        if cls_name in VOC_CLASS_NAMES_COCOFIED:
            cls_name = BASE_VOC_CLASS_NAMES[VOC_CLASS_NAMES_COCOFIED.index(cls_name)]
        if cls_name not in known_classes:
            cls_name = 'unknown'
        obj_struct["name"] = cls_name
        # obj_struct["pose"] = obj.find("pose").text
        # obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False, known_classes=None):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name

    # first load gt
    # read list of images
    with PathManager.open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    imagenames_filtered = []
    # load annots
    recs = {}
    for imagename in imagenames:
        rec = parse_rec(annopath.format(imagename), tuple(known_classes))
        if rec is not None:
            recs[imagename] = rec
            imagenames_filtered.append(imagename)

    imagenames = imagenames_filtered

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        # difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    # if 'unknown' not in classname:
    #     return tp, fp, 0

    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    # plot_pr_curve(prec, rec, classname+'.png')
    ap = voc_ap(rec, prec, use_07_metric)

    # print('tp: ' + str(tp[-1]))
    # print('fp: ' + str(fp[-1]))
    # print('tp: ')
    # print(tp)
    # print('fp: ')
    # print(fp)

    '''
    Computing Absolute Open-Set Error (A-OSE) and Wilderness Impact (WI)
                                    ===========    
    Absolute OSE = # of unknown objects classified as known objects of class 'classname'
    WI = FP_openset / (TP_closed_set + FP_closed_set)
    '''
    logger = logging.getLogger(__name__)

    # Finding GT of unknown objects
    unknown_class_recs = {}
    n_unk = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == 'unknown']
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        det = [False] * len(R)
        n_unk = n_unk + sum(~difficult)
        unknown_class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    if classname == 'unknown':
        return rec, prec, ap, 0, n_unk, None, None

    # Go down each detection and see if it has an overlap with an unknown object.
    # If so, it is an unknown object that was classified as known.
    is_unk = np.zeros(nd)
    for d in range(nd):
        R = unknown_class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            is_unk[d] = 1.0

    is_unk_sum = np.sum(is_unk)
    # OSE = is_unk / n_unk
    # logger.info('Number of unknowns detected knowns (for class '+ classname + ') is ' + str(is_unk))
    # logger.info("Num of unknown instances: " + str(n_unk))
    # logger.info('OSE: ' + str(OSE))

    tp_plus_fp_closed_set = tp+fp
    fp_open_set = np.cumsum(is_unk)

    return rec, prec, ap, is_unk_sum, n_unk, tp_plus_fp_closed_set, fp_open_set


def plot_pr_curve(precision, recall, filename, base_path='/home/fk1/workspace/OWOD/output/plots/'):
    fig, ax = plt.subplots()
    ax.step(recall, precision, color='r', alpha=0.99, where='post')
    ax.fill_between(recall, precision, alpha=0.2, color='b', step='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.savefig(base_path + filename)

    # print(precision)
    # print(recall)
