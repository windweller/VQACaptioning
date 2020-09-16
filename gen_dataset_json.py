"""
We generate the data/dataset_coco.json style file here
For data processing

We save under ./data/coco_fc_2017

All they need are:
{
    "images": [ {'filepath', 'filename'} ]
}
img['filepath'], img['filename']
'filename': 'COCO_val2014_000000184613.jpg'
'filepath': 'val2014'

Other arguments:
'split': 'val'
cocoid': 184613
"""

from tqdm import tqdm
import glob
import json

# TODO: add "train" into this...
if __name__ == '__main__':
    # val_2017_dir = "./data/mscoco/val2017/"
    # dataset_dic = {"images": [], "dataset": "coco"}
    # for name in tqdm(glob.glob(val_2017_dir + '*.jpg')):
    #     obj = {}
    #     filename = name.replace(val_2017_dir, "")
    #     obj['filename'] = filename
    #     obj['filepath'] = 'val2017'
    #     obj['cocoid'] = int(filename.strip(".jpg"))
    #     dataset_dic['images'].append(obj)
    #
    # json.dump(dataset_dic, open("./data/dataset_coco_2017val.json", 'w'))

    val_2017_dir = "./data/mscoco/train2017/"
    dataset_dic = {"images": [], "dataset": "coco"}
    for name in tqdm(glob.glob(val_2017_dir + '*.jpg')):
        obj = {}
        filename = name.replace(val_2017_dir, "")
        obj['filename'] = filename
        obj['filepath'] = 'train2017'
        obj['cocoid'] = int(filename.strip(".jpg"))
        dataset_dic['images'].append(obj)

    json.dump(dataset_dic, open("./data/dataset_coco_2017train.json", 'w'))