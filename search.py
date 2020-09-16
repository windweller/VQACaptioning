"""
Provide a search interface to retrieve a good partition of images
(Or to build our partition)
"""

import json
import random
import numpy as np
from PIL import Image
from os.path import join as pjoin

from whoosh import index
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser

from tqdm import tqdm

FC_FEAT_SIM = "./data/cocotalk_fc_feat_similarity"
BU_FEAT_SIM = "./data/cocobu_fc_feat_similarity"


class ImageDataset(object):
    def __init__(self, whoosh_index_dir=None):
        karpathy_split = json.load(open("./data/dataset_coco.json"))

        img_ids_train = []
        img_ids_val = []
        img_ids_test = []
        for img_dic in karpathy_split['images']:
            split = img_dic['split']
            image_id = img_dic['filename'].split('COCO_{}_'.format(img_dic['filepath']))[1].strip(".jpg")
            if split == 'test':
                img_ids_test.append(image_id)
            elif split == 'train':
                img_ids_train.append(image_id)
            elif split == 'val':
                img_ids_val.append(image_id)
            elif split == 'restval':
                img_ids_train.append(image_id)

        img_id_to_gold_captions = {}
        for img_dic in karpathy_split['images']:
            split = img_dic['split']
            image_id = img_dic['filename'].split('COCO_{}_'.format(img_dic['filepath']))[1].strip(".jpg")
            img_id_to_gold_captions[int(image_id)] = [img_dic['sentences'][i]['raw'] for i in
                                                      range(len(img_dic['sentences']))]

        self.img_id_to_gold_captions = img_id_to_gold_captions

        self.schema = Schema(image_id=NUMERIC(stored=True), content=TEXT)  # path=ID(stored=True)
        if whoosh_index_dir is None:
            os.makedirs("whoosh_indexdir", exist_ok=True)
            self.ix = create_in("whoosh_indexdir", self.schema)
        else:
            self.ix = index.open_dir(whoosh_index_dir)

        self.whoosh_index_dir = whoosh_index_dir
        self.writer = self.ix.writer()

        self.query_parser = QueryParser("content", self.ix.schema)

        self.val_test_id_to_partition = json.load(
            open("/home/anie/PragmaticVQA/data/vqa/val_test_image_to_partitions.json"))
        self.good_val_test_ids = json.load(open("./data/high_quality_val_test_ids.json"))

    def get_partition(self, img_id):
        return self.val_test_id_to_partition[str(img_id)]

    def sample_good_partition(self):
        img_id = random.choice(self.good_val_test_ids)
        return img_id, self.get_partition(img_id)

    def get_a_good_list_of_partitions(self):
        return ['524638',
                '114634',
                '112988',
                '242365',
                '426172',
                '473754',
                '439651',
                '449156',
                '244575',
                '94865',
                '432234',
                '538465',
                '396691',
                '58329',
                '454102',
                '67569',
                '319350',
                '270738',
                '309160',
                '472621',
                '255315',
                '295945',
                '226988',
                '174672',
                '419371',
                '531299',
                '444302',
                '211604',
                '355228',
                '195269',
                '206831',
                '55466',
                '248980',
                '423951',
                '312341',
                '419228',
                '37670',
                '553075',
                '545407',
                '71357',
                '392928',
                '384012',
                '447770',
                '305695',
                '221610',
                '185210',
                '229132',
                '67255',
                '391006',
                '149304']

    def get_caption(self, img_id):
        return self.img_id_to_gold_captions[img_id]

    def init_caption_search(self):
        if self.whoosh_index_dir is not None:
            print("Index is ready")
            return

        print("Indexing captions")
        for img_id, captions in tqdm(self.img_id_to_gold_captions.items()):
            self.writer.add_document(image_id=img_id, content=" ".join(captions))
        self.writer.commit()

    def search(self, text, return_captions=False, limit=10):
        with self.ix.searcher() as searcher:
            query = self.query_parser.parse(text)
            results = searcher.search(query, limit=limit)

            if return_captions:
                all_entries = [(res['image_id'], res['content']) for res in results]
                return all_entries
            else:
                image_ids = [res['image_id'] for res in results]
                return image_ids


image_val_test_root_path = "/mnt/fs5/anie/VQACaptioning/data/mscoco/val2014/COCO_val2014_"
image_train_root_path = "/mnt/fs5/anie/VQACaptioning/data/mscoco/train2014/COCO_train2014_"

import os


def grab_image_ids(dir_path):
    root, dirs, files = list(os.walk(dir_path))[0]
    img_ids = set()
    for file in files:
        img_ids.add(file.split('_')[0])

    return img_ids, files


def get_right_img_id(img_id):
    c_list = [c for c in img_id]
    zero_pad = ['0' for _ in range(12 - len(c_list))]
    return ''.join(zero_pad + c_list)


def display_image(img_id, shrink_prop=0.3):
    img_id = str(img_id)
    try:
        img = Image.open(image_val_test_root_path + get_right_img_id(img_id) + '.jpg')  # filename=
    except:
        img = Image.open(image_train_root_path + get_right_img_id(img_id) + '.jpg')  # filename=

    w, h = img.size
    return img.resize((int(w * shrink_prop), int(h * shrink_prop)))


def retrieve_img(sim_dir, cell, k, threshold=None, reverse=True):
    """
    :param sim_dir: root directory
    :param cell: a list of images ids, should be integer, no zero-padding
    :param k: retrieve k-nearest for each image
    :param threshold: not coded in
    :return:
    """
    # we could add a distance threshold for retrieval
    # currently not doing it

    # we retrieve as a set
    similar_images = set()

    for img_id in cell:
        # both are sorted in descending order
        # {"distance": [606.717529296875, 500.06451416015625, 491.666015625, 479.6894226074219 ...}
        dic = json.load(open(pjoin(sim_dir, "{}.json".format(img_id))))
        dist_list, index_list = dic['distance'], dic['index']
        if reverse:
            index_list = index_list[::-1]
            dist_list = dist_list[::-1]
        closest_k_list = index_list[:k]  # closest k items
        closest_k_dist_list = dist_list[:k]

        similar_images.update(closest_k_list)

    return list(similar_images)


def retrieve_img_fc(cell, k, threshold=None, reverse=True):
    return retrieve_img(FC_FEAT_SIM, cell, k, threshold, reverse)


def retrieve_img_bu(cell, k, threshold=None, reverse=True):
    return retrieve_img(BU_FEAT_SIM, cell, k, threshold, reverse)
