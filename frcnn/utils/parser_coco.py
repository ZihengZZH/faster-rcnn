import os
import cv2
import json
from tqdm import tqdm


def get_data(input_path):
    """
    Get Data from MSCOCO Dataset
    ---
    dataset should contain information on width/height/bndboxes/names
    """
    all_imgs = []
    classes_count = {}
    class_mapping = {}

    # parsing Flag
    visualise = False

    # MSCOCO directory
    data_path = input_path

    print('Parsing annotation files')
    annot_path = os.path.join(data_path, 'annotations_bbox')
    imgs_path = os.path.join(data_path, 'images')

    # images directory (train, val, trainval, test)
    imgsets_path_trainval = os.path.join(data_path, 'images', 'trainval.txt')
    imgsets_path_train = os.path.join(data_path, 'images', 'train.txt')
    imgsets_path_val = os.path.join(data_path, 'images', 'val.txt')
    imgsets_path_test = os.path.join(data_path, 'images', 'test.txt')

    trainval_files = []
    train_files = []
    val_files = []
    test_files = []

    with open(imgsets_path_trainval) as f:
        for line in f:
            trainval_files.append(line.strip())

    with open(imgsets_path_train) as f:
        for line in f:
            train_files.append(line.strip())

    with open(imgsets_path_val) as f:
        for line in f:
            val_files.append(line.strip())

    # test-set (default) not included in MSCOCO
    if os.path.isfile(imgsets_path_test):
        with open(imgsets_path_test) as f:
            for line in f:
                test_files.append(line.strip())

    # annotation read
    annots_train = json.load(open(os.path.join(annot_path, 'bbox_train2017.json'), 'r'))
    annots_val = json.load(open(os.path.join(annot_path, 'bbox_val2017.json'), 'r'))
    annots = dict()
    annots['train'] = annots_train
    annots['val'] = annots_val

    for part in ['train', 'val']:
        annots_keys = tqdm(annots[part].keys())
        for img_name in annots_keys:
            annots_keys.set_description("Processing %s" % img_name)
            for bbox in annots[part][img_name]:
                class_name = bbox['label'].replace(' ', '')
                all_imgs.append({
                    "filepath": os.path.join(data_path, 'images', '%s2017' % part, "%s.jpg" % img_name),
                    "width": None,
                    "height": None,
                    "bboxes": [{
                        "class": class_name,
                        "x1": bbox['bbox']['x1'],
                        "y1": bbox['bbox']['x2'],
                        "x2": bbox['bbox']['y1'],
                        "y2": bbox['bbox']['y2'],
                        "difficult": False
                    }],
                    "image_id": img_name,
                    "imageset": part
                })
                if class_name not in classes_count:
                    classes_count[class_name] = 1
                else:
                    classes_count[class_name] += 1
                if class_name not in class_mapping:
                    class_mapping[class_name] = len(class_mapping)

    # visualise bounding boxes
    if visualise:
        img = cv2.imread(annotation_data['filepath'])
        for bbox in annotation_data['bboxes']:
            cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 0, 255))
        cv2.imshow('img', img)
        print(annotation_data['imageset'])
        cv2.waitKey(0)

    return all_imgs, classes_count, class_mapping
