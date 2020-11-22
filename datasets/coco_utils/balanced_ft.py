import itertools
import random
import os
import xml.etree.ElementTree as ET
from fvcore.common.file_io import PathManager

from detectron2.utils.store_non_list import Store

VOC_CLASS_NAMES_COCOFIED = [
    "airplane",  "dining table", "motorcycle",
    "potted plant", "couch", "tv"
]

BASE_VOC_CLASS_NAMES = [
    "aeroplane", "diningtable", "motorbike",
    "pottedplant",  "sofa", "tvmonitor"
]

VOC_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

T2_CLASS_NAMES = [
    "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "microwave", "oven", "toaster", "sink", "refrigerator"
]

T3_CLASS_NAMES = [
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake"
]

T4_CLASS_NAMES = [
    "bed", "toilet", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl"
]

UNK_CLASS = ["unknown"]

# Change this accodingly for each task t*
known_classes = list(itertools.chain(VOC_CLASS_NAMES, T2_CLASS_NAMES))
train_files = ['/home/fk1/workspace/OWOD/datasets/VOC2007/ImageSets/Main/t2_train.txt','/home/fk1/workspace/OWOD/datasets/VOC2007/ImageSets/Main/t1_train.txt']

# known_classes = list(itertools.chain(VOC_CLASS_NAMES))
# train_files = ['/home/fk1/workspace/OWOD/datasets/VOC2007/ImageSets/Main/train.txt']
annotation_location = '/home/fk1/workspace/OWOD/datasets/VOC2007/Annotations'

items_per_class = 20
dest_file = '/home/fk1/workspace/OWOD/datasets/VOC2007/ImageSets/Main/t2_ft_' + str(items_per_class) + '.txt'

file_names = []
for tf in train_files:
    with open(tf, mode="r") as myFile:
        file_names.extend(myFile.readlines())

random.shuffle(file_names)

image_store = Store(len(known_classes), items_per_class)

current_min_item_count = 0

for fileid in file_names:
    fileid = fileid.strip()
    anno_file = os.path.join(annotation_location, fileid + ".xml")

    with PathManager.open(anno_file) as f:
        tree = ET.parse(f)

    for obj in tree.findall("object"):
        cls = obj.find("name").text
        if cls in VOC_CLASS_NAMES_COCOFIED:
            cls = BASE_VOC_CLASS_NAMES[VOC_CLASS_NAMES_COCOFIED.index(cls)]
        if cls in known_classes:
            image_store.add((fileid,), (known_classes.index(cls),))

    current_min_item_count = min([len(items) for items in image_store.retrieve(-1)])
    print(current_min_item_count)
    if current_min_item_count == items_per_class:
        break

filtered_file_names = []
for items in image_store.retrieve(-1):
    filtered_file_names.extend(items)

print(image_store)
print(len(filtered_file_names))
print(len(set(filtered_file_names)))

filtered_file_names = set(filtered_file_names)
filtered_file_names = map(lambda x: x + '\n', filtered_file_names)

with open(dest_file, mode="w") as myFile:
    myFile.writelines(filtered_file_names)

print('Saved to file: ' + dest_file)
