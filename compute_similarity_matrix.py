"""
Compute similarity matrix
"""

import glob
from tqdm import tqdm
import faiss
import numpy as np
from os.path import join as pjoin
import random
import json

random.seed(153242)

def load_np_array(root_dir, cache_freq=5000):
    names = glob.glob(pjoin(root_dir, '*.npy'))
    img_ids = []
    print("loading all numpy arrays into memory")

    large_np_array = []

    with tqdm(total=len(names)) as pbar:
        for i in range(0, len(names), cache_freq):
            np_list = []
            for name in names[i:i+cache_freq]:
                a = np.load(name)
                np_list.append(a)
                img_ids.append(int(name.replace(root_dir, "").strip(".npy")))
                pbar.update(1)
            np_array = np.vstack(np_list)
            large_np_array.append(np_array)

    return np.vstack(large_np_array), np.array(img_ids)

def build_faiss_index(np_array):
    d = np_array.shape[1]
    # Inner-product cosine distance
    index = faiss.IndexFlatIP(d)  # build the index
    #print(index.is_trained)
    index.add(np_array)  # add vectors to the index
    #print(index.ntotal)
    return index

import os
from os.path import join as pjoin

def build_similarity_list(out_dir, np_array, image_id_array, index, k=50, cache_freq=100):
    os.makedirs(out_dir, exist_ok=True)
    # since it's parallelizable, we do 500 images at a time
    with tqdm(total=image_id_array.shape[0]) as pbar:
        for i in range(0, image_id_array.shape[0], cache_freq):
            search_q = np_array[i:i+cache_freq]
            dist_mat, idx_mat = index.search(search_q, k)
            img_ids = image_id_array[i:i + cache_freq]
            for j, img_id in enumerate(img_ids):
                with open(pjoin(out_dir, "{}.json".format(img_id)), 'w') as f:
                    true_similar_ids = image_id_array[idx_mat[j]]
                    json.dump({"distance": dist_mat[j].tolist(), "index": true_similar_ids.tolist()}, f)
                pbar.update(1)

if __name__ == '__main__':
    np_array, img_id_list = load_np_array("./cocotalk_fc/")
    print("array assembled")
    index = build_faiss_index(np_array)
    print("index built!")
    build_similarity_list("./cocotalk_fc_feat_similarity", np_array, img_id_list, index)
