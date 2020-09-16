"""
We build faiss index here
"""

import glob
from tqdm import tqdm
import faiss
import numpy as np
from os.path import join as pjoin
import random

random.seed(153242)

# Not a satisfying way because it converts index to a string only
def serialize_index(index):
    """ convert an index to a numpy uint8 array  """
    writer = faiss.VectorIOWriter()
    faiss.write_index(index, writer)
    return faiss.vector_to_array(writer.data)

def deserialize_index(data):
    reader = faiss.VectorIOReader()
    faiss.copy_array_to_vector(data, reader.data)
    return faiss.read_index(reader)

def load_np_array(root_dir, limit=5000):
    np_list = []
    image_ids = []
    names = glob.glob(pjoin(root_dir, '*.npy'))
    random.shuffle(names)
    for name in tqdm(names[:limit]):
        a = np.load(name)
        np_list.append(a)
        image_ids.append(int(name.replace(root_dir, "").strip(".npy")))

    np_array = np.vstack(np_list)
    return np_array, np.array(image_ids)

def build_faiss_index(np_array):
    d = np_array.shape[1]
    index = faiss.IndexFlatIP(d)  # build the index
    #print(index.is_trained)
    index.add(np_array)  # add vectors to the index
    #print(index.ntotal)
    return index

def search_img(root_dir, cell, index, image_id_list, k):
    """
    :param cell: a list of images
    :param index: faiss_index
    :return: a list of images that are similar
    """
    # load image np array
    # index.search(query, k=10).
    # returns a tuple: (distance_matrix of [query_size, k], index matrix of [query_size, k]
    cell_np = []
    for img_id in cell:
        cell_np.append(np.load(pjoin(root_dir, str(img_id) + '.npy')))
    cell_np = np.vstack(cell_np)

    dist_mat, idx_mat = index.search(cell_np, k)
    # TODO: current we don't have a threshold of rejection; we probably should
    # TODO: have a rejection threshold here
    # TODO: i.e., any distance > threshold, we don't include
    indices = idx_mat.flatten()

    # this index list is likely NOT disjoint, so we need to make that happen
    # need to reverse because somehow that's the case?
    indices = np.array(list(set(indices.tolist())), dtype=int)[::-1]

    return image_id_list[indices].tolist()

from os.path import join as pjoin
import json

def retrieve_img(sim_dir, cell, k, threshold=None):
    # we could add a distance threshold for retrieval
    # currently not doing it

    # we retrieve as a set
    similar_images = set()
    for img_id in cell:
        # both are sorted in descending order
        # {"distance": [606.717529296875, 500.06451416015625, 491.666015625, 479.6894226074219 ...}
        dic = json.load(open(pjoin(sim_dir, "{}.json".format(img_id))))
        dist_list, index_list = dic['distance'], dic['index']
        closest_k_list = index_list[-k:]  # closest k items
        similar_images.update(closest_k_list)

    return list(similar_images)

if __name__ == '__main__':

    # d = 64  # dimension
    # nb = 100000  # database size
    # nq = 10000  # nb of queries
    # np.random.seed(1234)  # make reproducible
    # xb = np.random.random((nb, d)).astype('float32')
    # xb[:, 0] += np.arange(nb) / 1000.
    # xq = np.random.random((nq, d)).astype('float32')
    # xq[:, 0] += np.arange(nq) / 1000.

    np_array = load_np_array("./data/")



    # index = faiss.IndexFlatL2(d)  # build the index
    # print(index.is_trained)
    # index.add(xb)  # add vectors to the index
    # print(index.ntotal)



    # quantization seems so slow!!

    # nlist = 100
    # m = 8  # number of subquantizers
    # k = 4
    # quantizer = faiss.IndexFlatL2(d)  # this remains the same
    # index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
    # # 8 specifies that each sub-vector is encoded as 8 bits
    # index.train(xb)
    # index.add(xb)
    # D, I = index.search(xb[:5], k)  # sanity check
    # print(I)
    # print(D)
    # index.nprobe = 10  # make comparable with experiment above
    # D, I = index.search(xq, k)  # search
    # print(I[-5:])
