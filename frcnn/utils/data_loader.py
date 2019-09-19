from .parser_pascal import get_data as get_data_voc
from .parser_coco import get_data as get_data_coco
import copy


def merge_count(count1, count2):
    count_all = dict()
    for key in count1.keys():
        count_all[key] = count1[key]
    for key in count2.keys():
        if key in count_all.keys():
            count_all[key] += count2[key]
        else:
            count_all[key] = count2[key]
    return count_all

def merge_mapping(mapping1, mapping2):
    mapping_all = dict()
    for key in mapping1.keys():
        mapping_all[key] = mapping1[key]
    for key in mapping2.keys():
        if key in mapping_all.keys():
            continue
        else:
            mapping_all[key] = len(mapping_all)
    return mapping_all 
        
def get_data(train_path_voc, train_path_coco):
    all_imgs_voc, class_count_voc, class_mapping_voc = get_data_voc(train_path_voc)
    all_imgs_coco, class_count_coco, class_mapping_coco = get_data_coco(train_path_coco)
    all_imgs = copy.deepcopy(all_imgs_voc)
    all_imgs.extend(all_imgs_coco)
    assert len(all_imgs) == len(all_imgs_voc) + len(all_imgs_coco)
    class_count = merge_count(class_count_voc, class_count_coco)
    class_mapping = merge_mapping(class_mapping_voc, class_mapping_coco)
    return all_imgs, class_count, class_mapping