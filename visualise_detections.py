import cv2
import os
import torch
from torch.distributions.weibull import Weibull
from torch.distributions.transforms import AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


def create_distribution(scale, shape, shift):
    wd = Weibull(scale=scale, concentration=shape)
    transforms = AffineTransform(loc=shift, scale=1.)
    weibull = TransformedDistribution(wd, transforms)
    return weibull


def compute_prob(x, distribution):
    eps_radius = 0.5
    num_eval_points = 100
    start_x = x - eps_radius
    end_x = x + eps_radius
    step = (end_x - start_x) / num_eval_points
    dx = torch.linspace(x - eps_radius, x + eps_radius, num_eval_points)
    pdf = distribution.log_prob(dx).exp()
    prob = torch.sum(pdf * step)
    return prob


def update_label_based_on_energy(logits, classes, unk_dist, known_dist):
    unknown_class_index = 80
    cls = classes
    lse = torch.logsumexp(logits[:, :5], dim=1)
    for i, energy in enumerate(lse):
        p_unk = compute_prob(energy, unk_dist)
        p_known = compute_prob(energy, known_dist)
        # print(str(p_unk) + '  --  ' + str(p_known))
        if torch.isnan(p_unk) or torch.isnan(p_known):
            continue
        if p_unk > p_known:
            cls[i] = unknown_class_index
    return cls

# Get image
fnum = '348006'
file_name = '000000' + fnum
im = cv2.imread("/home/fk1/workspace/OWOD/datasets/VOC2007/JPEGImages/" + file_name + ".jpg")
# model = '/home/fk1/workspace/OWOD/output/old/t1_20_class/model_0009999.pth'
# model = '/home/fk1/workspace/OWOD/output/t1_THRESHOLD_AUTOLABEL_UNK/model_final.pth'
# model = '/home/fk1/workspace/OWOD/output/t1_clustering_with_save/model_final.pth'
# model = '/home/fk1/workspace/OWOD/output/t2_ft/model_final.pth'
# model = '/home/fk1/workspace/OWOD/output/t3_ft/model_final.pth'
model = '/home/fk1/workspace/OWOD/output/t4_ft/model_final.pth'
cfg_file = '/home/fk1/workspace/OWOD/configs/OWOD/t1/t1_test.yaml'


# Get the configuration ready
cfg = get_cfg()
cfg.merge_from_file(cfg_file)
cfg.MODEL.WEIGHTS = model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.61
# cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.8
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4

# POSITIVE_FRACTION: 0.25
# NMS_THRESH_TEST: 0.5
# SCORE_THRESH_TEST: 0.05
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 21

predictor = DefaultPredictor(cfg)
outputs = predictor(im)

print('Before' + str(outputs["instances"].pred_classes))

param_save_location = os.path.join('/home/fk1/workspace/OWOD/output/t1_clustering_val/energy_dist_' + str(20) + '.pkl')
params = torch.load(param_save_location)
unknown = params[0]
known = params[1]
unk_dist = create_distribution(unknown['scale_unk'], unknown['shape_unk'], unknown['shift_unk'])
known_dist = create_distribution(known['scale_known'], known['shape_known'], known['shift_known'])

instances = outputs["instances"].to(torch.device("cpu"))
dev =instances.pred_classes.get_device()
classes = instances.pred_classes.tolist()
logits = instances.logits
classes = update_label_based_on_energy(logits, classes, unk_dist, known_dist)
classes = torch.IntTensor(classes).to(torch.device("cuda"))
outputs["instances"].pred_classes = classes
print(classes)
print('After' + str(outputs["instances"].pred_classes))


v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
img = v.get_image()[:, :, ::-1]
cv2.imwrite('output_' + file_name + '.jpg', img)

