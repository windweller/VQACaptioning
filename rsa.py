"""
Currently we don't need a seperate module...I think
We can have rsa.py and rsa_utils.py on the main dir

For RSA,
We get logprob of the original image quite easily and seperately
then we evaluate those beams on a batch of other images
"""

import os
import random
import copy
import json
import pickle
from dataloader import *
import models
import torch
import torch.nn.functional as F
from os.path import join as pjoin
from tqdm import tqdm
import misc.utils as utils
from build_faiss import load_np_array, build_faiss_index, retrieve_img

from lm import TransformerLM

from scipy.stats import entropy

class VQADistractorDataset(object):

    def __init__(self, root_path, data_loader=None, max_cap_per_cell=5, max_num_cells=5, cell_select_strategy='none',
                 seed=1234, argstring=None, expand_size=0, expand_img_feat_dataset="./data/cocotalk_fc_feat_similarity/",
                 expand_threshold=30, vqa_model_root_dir="/mnt/fs5/anie/VQACaptioning/data/vqa_model_data/", silent=True, limit=5000,
                 eval_batch_size=8):
        """
        We are implementing the following strategy for distractors:
        1. Select within negated cells
        2. Select randomly on the full dataset (probably just within batch)
        
        max_num_cells * max_cap_per_cell = num of distractors we get

        We need to design this to have two modes:
        1. Interactive mode (where we can examine/visualize result)
            - We build off of this actually
        2. Automatic mode (where we get to plug into RSAModel and run eval_utils.py)
            - Automatic part is supported by the RSAModel forward() function
        
        Arguments:
            root_path {[type]} -- root_path
            dataloader {[type]} -- this strictly refers to `loader.dataset`
        
        Keyword Arguments:
            max_num_cells {int} -- if the partition has too many cells, we only consider max_num_cells of them
            max_cap_per_cell {int} -- [description] (default: {5})
            cell_select_strategy {str} -- when exceed max_cap_per_cell, we can pick randomly, according to similarity, etc. (default: {"none"}, means
                                          we don't change the list order)
                                          options: ['none', 'shuffle', 'similarity', etc.]
            argstring {str} -- configurations to load dataloader
            expand_size {int} --  this determines how many images to pull from `expand_img_dataset`
            expand_threshold {int} -- if there are more than `expand_threshold` number of images in partition,
                                       we would not expand the partition.
                                      Note: currently we are loading from the dataset itself.
            limit {int} -- we only load 5000 numpy arrays in
        """

        if data_loader is not None:
            self.data_loader = data_loader
            self.dataset = self.data_loader.dataset
        else:
            self.data_loader = self.initialize_dataloader(argstring)
            self.dataset = self.data_loader.dataset

        self.train_image_to_partitions = json.load(open(root_path + 'train_image_to_partitions.json'))
        self.val_test_image_to_partitions = json.load(open(root_path + 'val_test_image_to_partitions.json'))

        self.max_cap_per_cell = max_cap_per_cell
        self.max_num_cells = max_num_cells
        self.cell_select_strategy = cell_select_strategy

        self.image_id_to_loader_idx = {}

        self.silent = silent

        for idx, em in enumerate(self.dataset.info['images']):
            self.image_id_to_loader_idx[em['id']] = idx

        random.seed(seed)

        # considering expansion
        self.limit = limit
        self.expand_size = expand_size
        self.expand_threshold = expand_threshold
        self.expand_img_feat_dataset = expand_img_feat_dataset
        self.vqa_model_root_dir = vqa_model_root_dir
        self.faiss_index, self.vqa_demo = None, None
        self.eval_batch_size = eval_batch_size
        if expand_size > 0:
            # np_array, self.np_image_ids = load_np_array(expand_img_feat_dataset, limit=self.limit)
            # self.faiss_index = build_faiss_index(np_array)

            from vqa import PythiaDemo, VQAHelper
            print("initializing VQA model")
            self.vqa_demo = PythiaDemo(vqa_model_root_dir)
            self.vqa_helper = VQAHelper(self.vqa_demo, self.eval_batch_size)

    def set_expand_size(self, expand_size):
        if self.expand_size == 0 and expand_size > 0 and self.vqa_demo is None:
            # np_array, self.np_image_ids = load_np_array(self.expand_img_feat_dataset, limit=self.limit)
            # self.faiss_index = build_faiss_index(np_array)

            from vqa import PythiaDemo, VQAHelper
            print("initializing VQA model")
            self.vqa_demo = PythiaDemo(self.vqa_model_root_dir)
            self.vqa_helper = VQAHelper(self.vqa_demo, self.eval_batch_size)

        self.expand_size = expand_size

    def initialize_dataloader(self, argstring=None):
        import opts
        import argparse
        import misc.utils as utils

        # Input arguments and options
        parser = argparse.ArgumentParser()
        # Input paths
        parser.add_argument('--model', type=str, default='',
                            help='path to model to evaluate')
        parser.add_argument('--cnn_model', type=str, default='resnet101',
                            help='resnet101, resnet152')
        parser.add_argument('--infos_path', type=str, default='',
                            help='path to infos to evaluate')
        parser.add_argument('--only_lang_eval', type=int, default=0,
                            help='lang eval on saved results')
        parser.add_argument('--force', type=int, default=0,
                            help='force to evaluate no matter if there are results available')
        opts.add_eval_options(parser)
        opts.add_diversity_opts(parser)
        if argstring is None:
            opt = parser.parse_args(
                "--dump_images 0 --num_images 5000 --model ./data/bottomup/trans_nsc/model-best.pth --infos_path ./data/bottomup/trans_nsc/infos_trans_nsc-best.pkl --language_eval 1 --sample_method bs --beam_size 5".split())
        else:
            opt = parser.parse_args(argstring.split())

        with open(opt.infos_path, 'rb') as f:
            infos = utils.pickle_load(f)

        # override and collect parameters
        replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
        ignore = ['start_from']

        for k in vars(infos['opt']).keys():
            if k in replace:
                setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
            elif k not in ignore:
                if not k in vars(opt):
                    vars(opt).update({k: vars(infos['opt'])[k]})  # copy over options from model

        opt.batch_size = 5

        loader = DataLoader(opt)
        return loader

    def expand_qud(self):
        # randomly sample 20 images, condition on QuD, ask it
        pass

    def get_distractors_by_random(self, img_id, split='test'):
        """
        This is the "baseline" wrong approach, where we randomly pick one..
        """
        raise NotImplemented

    def get_distractors_by_similarity(self, img_id):
        """
        This is the "baseline" common approach, where we just find the image in training set that's closest to
        the test image.
        """
        raise NotImplemented

    def get_top_k_from_cell(self, cell):
        if self.cell_select_strategy == 'none':
            return cell[:self.max_cap_per_cell]
        elif self.cell_select_strategy == 'shuffle':
            random.shuffle(cell)
            return cell[:self.max_cap_per_cell]
        elif self.cell_select_strategy == 'similarity':
            raise NotImplemented
        else:
            raise NotImplemented

    def get_cells_by_partition(self, img_id, split='test', qud_id=-1):
        """
        [(57870,
            [('What are the chairs made off?', {'metal': [404545], 'wood': [57870]}),
            ('Is this a dinner setting?', {'no': [576593], 'yes': [57870]})
        ...]

        This function needs to be modified multiple times (based on what we want to get out...)
        So don't waste too much time...only add things in the future

        This is our desired approach, we get from cells that don't contain this image.

        For now we discard answers...but we might be able to pass it along in the future...
        
        For now, we are not batching anything.

        Decisions:
        1. Do we sample just from one cell?
        2. Do we merge all images from all negated cells in parttion and sample from that? (current)

        Could be: "soccer" vs. "football", "baseball", ...
        
        img_id: int
        split: train/val/test, refer to Karpathy split
        
        returns a list of partitions
        in the format of [(qud1, distractors_for_partition1, similar_for_partition1),
                          (qud2, distractors_for_partition2, similar_for_partition2), ...]

        Warning: similar_for_partition2 after removing the original target image, this can be empty!!

        In the outside, we use `img_list[qud_id][1]` for distractors
        `img_list[qud_id][2]` for similar images
        So in general we are fine!
        """

        # Step 1: look up the partitions
        if split != 'train':
            partitions = self.val_test_image_to_partitions[str(img_id)]
        else:
            partitions = self.train_image_to_partitions[str(img_id)]

        # Step 1.5: populate distractor cells
        # Funny thing is that we search based on distractor images, not target images!
        # and keep ones that don't have the same answer as the target

        if self.expand_size > 0:
            distractors_by_partition = []

            # we populate all partitions!
            for i, par in enumerate(partitions):
                negated_cells = []
                same_cells = []
                qud = par[0]
                for ans, cell in par[1].items():
                    if img_id in cell:
                        target_answer = ans
                        # can't do in-place remove, let's just generate a new list
                        new_cell = [c for c in cell if c != img_id]
                        same_cells.append((ans, new_cell))
                    else:
                        negated_cells.append((ans, cell))

                overall_distractor_ids = []
                for ans, cell in negated_cells[:self.max_num_cells]:
                    selected_imgs = self.get_top_k_from_cell(cell)
                    overall_distractor_ids.extend(selected_imgs)

                overall_similar_ids = []
                for ans, cell in same_cells[:self.max_num_cells]:
                    selected_imgs = self.get_top_k_from_cell(cell)
                    overall_similar_ids.extend(selected_imgs)

                assert img_id not in overall_distractor_ids
                assert img_id not in overall_similar_ids

                # if we specify qud_id, this only gets executed once!
                # otherwise we populate everything
                if qud_id != -1 and i == qud_id:
                    if not self.silent:
                        print("before expansion we have {} distractor images".format(len(overall_distractor_ids)))

                    # now we populate!!
                    # TODO: simplify the logic here...run by Reuben!!! He can tell you if this logic is crazy
                    if len(overall_distractor_ids) < self.expand_threshold:
                        k = self.expand_size // len(overall_distractor_ids) + 1  # if it's 5 // 20, it will be 0, we still add 1 to it
                        # this is ok! if it has a lot of distractors, due to "conflict", we won't expand much
                        # if it has very few distractors like 1 or 2, then this can get a lot of images!
                        # this is the result that we want.
                        new_img_ids = retrieve_img(self.expand_img_feat_dataset, overall_distractor_ids, k)
                        if img_id in new_img_ids:
                            new_img_ids.remove(img_id)  # can't contain target image!
                        negated_new_img_ids, _, sim_new_img_ids, _ = self.vqa_helper.filter_img_ids_by_answers(qud, target_answer, new_img_ids, self.silent)
                        overall_distractor_ids.extend(negated_new_img_ids)
                        overall_similar_ids.extend(sim_new_img_ids)

                        # however, this is not good because we need to find one similar to original image...
                        # to populate the same image cells

                        # we only search for k, not adding that many...
                        # TODO: this is not the best idea...
                        # TODO: the end result is that distractors have way more images than "similar images".
                        # TODO: this shows that presupposition model or smart filtering is important
                        # new_img_ids = search_img(self.expand_img_feat_dataset, [img_id], self.faiss_index,
                        #                          self.np_image_ids, k)
                        new_img_ids = retrieve_img(self.expand_img_feat_dataset, [img_id], k)

                        if img_id in new_img_ids:
                            new_img_ids.remove(img_id)  # can't contain target image!
                        negated_new_img_ids, _, sim_new_img_ids, _ = self.vqa_helper.filter_img_ids_by_answers(qud, target_answer, new_img_ids, self.silent)
                        overall_distractor_ids.extend(negated_new_img_ids)
                        overall_similar_ids.extend(sim_new_img_ids)

                    if not self.silent:
                        print("after expansion we have {} distractor images".format(len(overall_distractor_ids)))

                distractors_by_partition.append((qud, overall_distractor_ids, overall_similar_ids))

            return distractors_by_partition

        # Step 2: retrieve each image
        # with a cap, exceed the cap, we shuffle the cell list, pick first k
        # or we can do a strategy (in the future)

        distractors_by_partition = []

        for par in partitions:
            # par: ('What are the chairs made off?', {'metal': [404545], 'wood': [57870]})
            # get negated cells
            negated_cells = []
            same_cells = []
            qud = par[0]
            for ans, cell in par[1].items():
                if img_id not in cell:
                    negated_cells.append((ans, cell))
                else:
                    new_cell = [c for c in cell if c != img_id]
                    same_cells.append((ans, new_cell))

            overall_distractor_ids = []
            for ans, cell in negated_cells[:self.max_num_cells]:
                selected_imgs = self.get_top_k_from_cell(cell)
                overall_distractor_ids.extend(selected_imgs)

            overall_similar_ids = []
            for ans, cell in same_cells[:self.max_num_cells]:
                selected_imgs = self.get_top_k_from_cell(cell)
                overall_similar_ids.extend(selected_imgs)

            assert img_id not in overall_distractor_ids
            assert img_id not in overall_similar_ids

            distractors_by_partition.append((qud, overall_distractor_ids, overall_similar_ids))

        return distractors_by_partition

    def map_img_id_to_idx(self, img_id):
        # we don't need to worry about counter and wrapped

        # elem = (self._index_list[self.iter_counter], self.iter_counter+1, wrapped)
        return (self.image_id_to_loader_idx[img_id], 0, False)

    def get_batch(self, list_img_ids, split='test'):
        """
        This is used to retrieve distractor images and turn them into a batch!

        :param list_img_ids: a list of image ids, integer; we turn them into a batch
        :param split: this is used to get `it_max`, how many images in the split
        :return:

        We return something that can be handled by model automatically

        {'fc_feats': tensor([[0.]]),
         'att_feats': tensor([[[6.0150e+00, 4.4883e-01, 4.1887e-02,  ..., 0.0000e+00,
                   1.0197e-01, 7.9792e+00],
                  [1.4278e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
                   3.9953e-01, 0.0000e+00],
                  [5.9261e-01, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
                   5.5667e-01, 5.3477e-03],
                  ...,
                  [2.6863e+00, 1.1994e-01, 0.0000e+00,  ..., 0.0000e+00,
                   0.0000e+00, 0.0000e+00],
                  [4.2054e+00, 3.4663e+00, 0.0000e+00,  ..., 0.0000e+00,
                   1.5952e-01, 0.0000e+00],
                  [3.5983e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
                   1.6066e-02, 4.9341e-03]]]),
         'att_masks': None,
         'labels': tensor([[ 0,  1, 38, 39,  1, 40,  6, 41, 42, 43,  1, 44,  0,  0,  0,  0,  0,  0],
                 [ 0,  1, 38, 43,  1, 45, 46, 47, 44,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                 [ 0,  1, 38, 39,  1, 48, 40, 43,  1, 45, 47, 44,  0,  0,  0,  0,  0,  0],
                 [ 0, 49, 35,  1, 38, 50, 35, 43,  1, 46, 44,  0,  0,  0,  0,  0,  0,  0],
                 [ 0,  1, 38, 51,  1, 44,  3, 14, 16, 17,  1, 52, 53,  0,  0,  0,  0,  0]]),
         'masks': tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
                 [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                 [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
                 [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                 [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.]]),
         'gts': (array([[ 1, 38, 39,  1, 40,  6, 41, 42, 43,  1, 44,  0,  0,  0,  0,  0],
                 [ 1, 38, 43,  1, 45, 46, 47, 44,  0,  0,  0,  0,  0,  0,  0,  0],
                 [ 1, 38, 39,  1, 48, 40, 43,  1, 45, 47, 44,  0,  0,  0,  0,  0],
                 [49, 35,  1, 38, 50, 35, 43,  1, 46, 44,  0,  0,  0,  0,  0,  0],
                 [ 1, 38, 51,  1, 44,  3, 14, 16, 17,  1, 52, 53,  0,  0,  0,  0]],
                dtype=uint32),),
         'bounds': {'it_pos_now': 0, 'it_max': 5000, 'wrapped': False},
         'infos': ({'ix': 1,
           'id': 522418,
           'file_path': 'val2014/COCO_val2014_000000522418.jpg'},)}
        """

        batch = [self.dataset[self.map_img_id_to_idx(img_id)] for img_id in list_img_ids]

        return self.dataset.collate_func(batch, split)


def load_model(argstring=None, split='test', batch_size=5, beam_size=5):
    import opts
    import argparse
    import misc.utils as utils

    # Input arguments and options
    parser = argparse.ArgumentParser()
    # Input paths
    parser.add_argument('--model', type=str, default='',
                        help='path to model to evaluate')
    parser.add_argument('--cnn_model', type=str, default='resnet101',
                        help='resnet101, resnet152')
    parser.add_argument('--infos_path', type=str, default='',
                        help='path to infos to evaluate')
    parser.add_argument('--only_lang_eval', type=int, default=0,
                        help='lang eval on saved results')
    parser.add_argument('--force', type=int, default=0,
                        help='force to evaluate no matter if there are results available')
    opts.add_eval_options(parser)
    opts.add_diversity_opts(parser)
    if argstring is None:
        opt = parser.parse_args(
            "--dump_images 0 --num_images 5000 --model ./data/bottomup/trans_nsc/model-best.pth --infos_path ./data/bottomup/trans_nsc/infos_trans_nsc-best.pkl --language_eval 1 --sample_method bs --beam_size {}".format(
                beam_size).split())
    else:
        opt = parser.parse_args(argstring.split())

    with open(opt.infos_path, 'rb') as f:
        infos = utils.pickle_load(f)

    # override and collect parameters
    replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
    ignore = ['start_from']

    for k in vars(infos['opt']).keys():
        if k in replace:
            setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
        elif k not in ignore:
            if not k in vars(opt):
                vars(opt).update({k: vars(infos['opt'])[k]})  # copy over options from model

    vocab = infos['vocab']  # ix -> word mapping
    opt.batch_size = batch_size

    # Setup the model
    opt.vocab = vocab
    model = models.setup(opt)
    del opt.vocab  # why is this deleting vocab? But well, it's what they do...
    model.load_state_dict(torch.load(opt.model))
    model.cuda()
    if split != 'train':
        model.eval()
    # crit = utils.LanguageModelCriterion()

    loader = DataLoader(opt)
    loader.reset_iterator(split)

    return model, loader, opt

def load_lm_model(model_name='log_transformer_lm', split='test', batch_size=5, beam_size=1):
    # TODO: note that with self-critical loss, it may or may not make sense to train LM on such loss.
    import opts
    import argparse
    import misc.utils as utils

    # Input arguments and options
    parser = argparse.ArgumentParser()
    # Input paths
    parser.add_argument('--model', type=str, default='',
                        help='path to model to evaluate')
    parser.add_argument('--cnn_model', type=str, default='resnet101',
                        help='resnet101, resnet152')
    parser.add_argument('--infos_path', type=str, default='',
                        help='path to infos to evaluate')
    parser.add_argument('--only_lang_eval', type=int, default=0,
                        help='lang eval on saved results')
    parser.add_argument('--force', type=int, default=0,
                        help='force to evaluate no matter if there are results available')
    opts.add_eval_options(parser)
    opts.add_diversity_opts(parser)
    if model_name == 'log_transformer_lm':
        opt = parser.parse_args(
            "--dump_images 0 --num_images 5000 --model ./log_transformer_lm/model-best.pth --infos_path ./log_transformer_lm/infos_transformer_lm.pkl --language_eval 1 --sample_method bs --beam_size {}".format(
                beam_size).split())
    else:
        raise Exception("LM Model not trained yet")

    with open(opt.infos_path, 'rb') as f:
        infos = utils.pickle_load(f)

    # override and collect parameters
    replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
    ignore = ['start_from']

    for k in vars(infos['opt']).keys():
        if k in replace:
            setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
        elif k not in ignore:
            if not k in vars(opt):
                vars(opt).update({k: vars(infos['opt'])[k]})  # copy over options from model

    vocab = infos['vocab']  # ix -> word mapping
    opt.batch_size = batch_size

    # Setup the model
    opt.vocab = vocab
    # model = models.setup(opt)
    model = TransformerLM(opt)
    del opt.vocab  # why is this deleting vocab? But well, it's what they do...
    model.load_state_dict(torch.load(opt.model))
    model.cuda()
    if split != 'train':
        model.eval()
    # crit = utils.LanguageModelCriterion()

    # loader = DataLoader(opt)
    # loader.reset_iterator(split)

    return model, opt


def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


class RSA(object):
    """
    RSA through matrix normalization
    Given a literal matrix of log-prob
        c1  c2  c3
    i   -5  -6  -20
    i'  -5  -9  -20
    i'' -10 -11 -20

    RSA has three cases:
    Case 1: If a sequence (C) has high prob for i, but high also in i', i'', the prob is relatively down-weighted
    Case 2: If a sequence (C) has low prob for i, but low also in i', i'', the prob is then relatively up-weighted (higher than original)
    Case 3: If a seuqnce (C) has high prob for i, but low for i', i'', the prob is relatively up-weighted
    (But this is hard to say due to the final row normalization)

    use logsumexp() to compute normalization constant

    Normalization/division in log-space is just a substraction

    Column normalization means: -5 - logsumexp([-5, -5, -10])
    (Add together a column)

    Row normalization means: -5 - logsumexp([-5, -6, -7])
    (Add together a row)

    We can compute RSA through the following steps:
    Step 1: add image prior: + log P(i) to the row
    Step 2: Column normalize
    - Pragmatic Listener L1: L1(i|c) \propto S0(c|i) P(i)
    Step 3: Multiply the full matrix by rationality parameter (0, infty), when rationality=1, no changes (similar to temperature)
    Step 4: add speaker prior: + log P(c_t|i, c_<t) (basically add the original literal matrix) (very easy)
            OR add a unconditioned speaker prior: + log P(c) (through a language model, like KenLM)
    Step 5: Row normalization
    - Pragmatic Speaker S1: S1(c|i) \propto L1(i|c) p(c), where p(c) can be S0

    The reason for additions is e^{\alpha log L1(i|c) + log p(i)}, where \alpha is rationality parameter
    """

    def __init__(self, lm_root_path="./data/lm/"):
        # can be used to add KenLM language model
        # The "gigaword" one takes too long to load
        self.kenlm_model = None
        self.kenlm_model_name = ""
        self.lm_root_path = lm_root_path

    def build_literal_matrix(self, orig_logprob, distractor_logprob):
        """
        :param orig_logprob: [n_sample]
        :param distractor_logprob: [num_distractors, n_sample]
        :return: We put orig_logprob as the FIRST row
                [num_distractors+1 , n_sample]
        """
        return torch.cat([orig_logprob.unsqueeze(0), distractor_logprob], dim=0)

    def compute_pragmatic_speaker(self, literal_matrix,
                                  rationality=1.0, speaker_prior=False, lm_logprobsf=None):
        """
        Do the normalization over logprob matrix

        literal_matrix: [num_distractor_images+1, captions]
        So row normalization correspond to

        :param literal_matrix: should be [I, C]  (num_images, num_captions)
                               Or [I, Vocab] (num_images, vocab_size)
        :param speaker_prior: turn on, we default to adding literal matrix
        :param speaker_prior_lm_mat: [I, Vocab] (a grammar weighting for previous tokens)

        :return:
               A re-weighted matrix [I, C/Vocab]
        """
        # step 1
        pass
        # step 2
        norm_const = logsumexp(literal_matrix, dim=0, keepdim=True)  # without keepdim, we get [num_captions]
        pragmatic_listener_matrix = literal_matrix - norm_const
        # step 3
        pragmatic_listener_matrix *= rationality
        # step 4
        if speaker_prior:
            # we add speaker prior
            # this needs to be a LM with shared vocabulary
            if lm_logprobsf is not None:
                # we had an error of adding [25, vocab] and [5, vocab]
                # I don't know where the 5 came from, but for successful broadcasting, we just add first row
                pragmatic_listener_matrix += lm_logprobsf[0]  # the broadcast should be along the image dimension...hopefully
                # lm_model = self.load_lm(speaker_prior_lm)
                # need to
                # lm_model.score()
                # This won't work for incremental, because:
                # It can't return a [I, Vocab] matrix. LM's vocab is disjoint with this vocab
            else:
                pragmatic_listener_matrix += literal_matrix
        # step 5
        norm_const = logsumexp(pragmatic_listener_matrix, dim=1, keepdim=True)  # row normalization
        pragmatic_speaker_matrix = pragmatic_listener_matrix - norm_const

        return pragmatic_speaker_matrix

    # def compute_entropy(self, prob_mat, dim, keepdim=True):
    #     prob_mat *= -torch.exp(prob_mat)
    #     return prob_mat.sum(dim, keepdim=keepdim)

    def compute_entropy(self, prob_mat, dim, keepdim=True):
        return -torch.sum(prob_mat * torch.exp(prob_mat), dim=dim, keepdim=keepdim)

    def compute_pragmatic_speaker_w_similarity(self, literal_matrix, num_similar_images,
                                               rationality=1.0, speaker_prior=False, lm_logprobsf=None,
                                               entropy_penalty_alpha=0.0):
        """
        TODO: keep in mind that unlike distractor_matrix, similarity_matrix might be empty!!!!
        :param literal_matrix: 0th row is the target distribution, [1:num_similar_images+1] is the similar matrix
                               [num_similar_images+1:] is the distractor matrix
        :param speaker_prior:
        :param speaker_prior_lm_mat:
        :param entropy_penalty_alpha: maybe be 0 to 1? Try out 0.1 first...
        :return:
        """
        from IPython.core.debugger import set_trace

        # step 1
        pass
        # step 2
        s0_mat = literal_matrix
        prior = s0_mat.clone()[0]
        #         norm_const = logsumexp(s0_mat, dim=0, keepdim=True)  # without keepdim, we get [num_captions]
        l1_mat = s0_mat - logsumexp(s0_mat, dim=0, keepdim=True)

        # step 3
        # pragmatic_listener_matrix *= rationality
        # step 5: QuD-RSA S1
        # 0). Compute entropy H[P(v|i, q(i)=q(i'))]; normalize "vertically" on
        # more formally, Entropy-RSA formalized the notion of finding the intersection instead of unions of the images in cell
        #         same_cell_norm = logsumexp(l1_mat[:num_similar_images + 1], dim=0, keepdim=True)
        same_cell_prob_mat = l1_mat[:num_similar_images + 1] - logsumexp(l1_mat[:num_similar_images + 1], dim=0)
        #         same_cell_norm
        entropy = self.compute_entropy(same_cell_prob_mat, 0, keepdim=True)  # (1, |V|)
        utility_2 = entropy

        # 1). Sum over similar images with target image (vertically)
        # [target_image, [similar_images], [distractor_images]]
        utility_1 = logsumexp(l1_mat[:num_similar_images + 1], dim=0, keepdim=True)  # [1, |V|]
        #         utility_1 = l1_mat[0]
        # This tradeoff may or may not be the best way...we are adding log-probability with entropy

        utility = (1 - entropy_penalty_alpha) * utility_1 + entropy_penalty_alpha * utility_2

        #         l1_mat[:num_similar_images + 1] = utility

        s1 = utility * rationality

        # set_trace()

        # apply rationality
        if speaker_prior:
            if lm_logprobsf is None:
                s1 += prior
            else:
                s1 += lm_logprobsf[0]  # lm rows are all the same  # here is two rows summation

        #         else: s1_mat = utility

        # step 4
        #         if speaker_prior:
        #             # we add speaker prior
        #             # this needs to be a LM with shared vocabulary
        #             if speaker_prior_lm_mat is not None:
        #                 pass
        #                 # lm_model = self.load_lm(speaker_prior_lm)
        #                 # lm_model.score()
        #                 # This won't work for incremental, because:
        #                 # It can't return a [I, Vocab] matrix. LM's vocab is disjoint with this vocab
        #             else:
        #                 l1_mat += s0_mat

        #     after this all rows will be the same
        # 2). Normalize each row in the normal manner

        #         set_trace()

        #         norm_const = logsumexp(l1_mat, dim=1, keepdim=True)  # row normalization
        #         s1_mat = s1_mat - logsumexp(s1_mat, dim=1, keepdim=True)

        # sys.exit(0)

        # set_trace()

        # remove this, change the take first row away
        # s1 = s1.repeat(11, 1)
        #         print(s1.shape)

        #         print((s1 - logsumexp(s1, dim=1, keepdim=True)).shape)
        return s1 - logsumexp(s1, dim=1, keepdim=True)

    def compute_pragmatic_speaker_w_similarity_old(self, literal_matrix, num_similar_images,
                                               rationality=1.0, speaker_prior=False, speaker_prior_lm_mat=None,
                                               entropy_penalty_alpha=0.0):
        """
        TODO: keep in mind that unlike distractor_matrix, similarity_matrix might be empty!!!!
        :param literal_matrix: 0th row is the target distribution, [1:num_similar_images+1] is the similar matrix
                               [num_similar_images+1:] is the distractor matrix
        :param speaker_prior:
        :param speaker_prior_lm_mat:
        :param entropy_penalty_alpha: maybe be 0 to 1? Try out 0.1 first...
        :return:
        """
        from IPython.core.debugger import set_trace

        # step 1
        pass
        # step 2
        s0_mat = literal_matrix
        norm_const = logsumexp(literal_matrix, dim=0, keepdim=True)  # without keepdim, we get [num_captions]
        l1_mat = literal_matrix - norm_const

        # step 3
        # pragmatic_listener_matrix *= rationality
        # step 5: QuD-RSA S1
        # 0). Compute entropy H[P(v|i, q(i)=q(i'))]; normalize "vertically" on
        # more formally, Entropy-RSA formalized the notion of finding the intersection instead of unions of the images in cell
        same_cell_norm = logsumexp(l1_mat[:num_similar_images + 1], dim=0, keepdim=True)
        same_cell_prob_mat = l1_mat[:num_similar_images + 1] - same_cell_norm
        utility_2 = self.compute_entropy(same_cell_prob_mat, 0, keepdim=True)  # (1, |V|)

        # 1). Sum over similar images with target image (vertically)
        # [target_image, [similar_images], [distractor_images]]
        utility_1 = logsumexp(l1_mat[:num_similar_images + 1], dim=0, keepdim=True)  # [1, |V|]

        # This tradeoff may or may not be the best way...we are adding log-probability with entropy

        utility = (1 - entropy_penalty_alpha) * utility_1 + entropy_penalty_alpha * utility_2

        l1_mat[:num_similar_images + 1] = utility

        # apply rationality
        l1_mat *= rationality

        # step 4
        if speaker_prior:
            # we add speaker prior
            # this needs to be a LM with shared vocabulary
            if speaker_prior_lm_mat is not None:
                pass
                # lm_model = self.load_lm(speaker_prior_lm)
                # lm_model.score()
                # This won't work for incremental, because:
                # It can't return a [I, Vocab] matrix. LM's vocab is disjoint with this vocab
            else:
                l1_mat += s0_mat

        #     after this all rows will be the same
        # 2). Normalize each row in the normal manner

        # set_trace()

        norm_const = logsumexp(l1_mat, dim=1, keepdim=True)  # row normalization
        s1_mat = l1_mat - norm_const

        # sys.exit(0)

        # set_trace()

        return s1_mat


class FullRSAModel(RSA):
    """
    Currently this only works under "EVAL"!
    It has a bunch of torch.no_grad() in there
    FullRSAModel might not be used for training

    Try to sample and return more...
    """

    def __init__(self, model, loader, opt, distractor_dataset):
        super().__init__()

        self.model = model
        self.loader = loader
        self.opt = opt
        self.distractor_dataset = distractor_dataset

        eval_kwargs = vars(opt)

        self.verbose = eval_kwargs.get('verbose', True)
        self.verbose_beam = eval_kwargs.get('verbose_beam', 1)
        self.verbose_loss = eval_kwargs.get('verbose_loss', 1)
        self.num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
        self.split = eval_kwargs.get('split', 'val')
        self.lang_eval = eval_kwargs.get('language_eval', 0)
        self.dataset = eval_kwargs.get('dataset', 'coco')
        self.beam_size = eval_kwargs.get('beam_size', 1)
        self.sample_n = eval_kwargs.get('sample_n', 1)  # this one will be ignored and updated later on

        self.batch_size = opt.batch_size
        self.max_length = eval_kwargs.get('max_length', 20)

        self.vocab_size = opt.vocab_size

        remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
        os.environ["REMOVE_BAD_ENDINGS"] = str(
            remove_bad_endings)  # Use this nasty way to make other code clean since it's a global configuration

        self.eval_kwargs = eval_kwargs

    def sample_data(self, split):
        """
        Sample a batch from loader
        Loader is the same used for training.

        :return:
        """
        data = self.loader.get_batch(split)
        return data

    def prep_data(self, data):
        """
        This is not the same as model._prepare_feature()
        This only loads data into CUDA

        :param data: whatever is returned by sample_data()
        :return:
        """
        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'],
               data['att_feats'],
               data['att_masks']]
        # this might cause issue...CUDA memory in this program is not quite freed
        tmp = [_.cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, att_masks = tmp
        return fc_feats, att_feats, att_masks

    def semantic_speaker(self, data, sample_n=1):
        """
        The goal of semantic speaker is that given an image, it can sample literal captions (S0)

        :return:
        """
        assert sample_n == self.beam_size or sample_n == 1, "sample_n needs to be 1 or beam size"

        fc_feats, att_feats, att_masks = self.prep_data(data)
        with torch.no_grad():
            tmp_eval_kwargs = self.eval_kwargs.copy()
            tmp_eval_kwargs.update({'sample_n': sample_n})
            seq, seq_logprobs = self.model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
            seq = seq.data
            # entropy = - (F.softmax(seq_logprobs, dim=2) * seq_logprobs).sum(2).sum(1) / ((seq > 0).float().sum(1) + 1)
            # perplexity = - seq_logprobs.gather(2, seq.unsqueeze(2)).squeeze(2).sum(1) / ((seq > 0).float().sum(1) + 1)

        if sample_n == 1:
            # so originally this would have been
            # torch.Size([25, 20])
            # torch.Size([25, 20, 9488])
            # but now we always reshape into 3D
            return seq.reshape([-1, 1, self.max_length]), seq_logprobs.reshape(
                [-1, 1, self.max_length, self.vocab_size + 1])
        else:
            return seq.reshape([-1, sample_n, self.max_length]), seq_logprobs.reshape(
                [-1, sample_n, self.max_length, self.vocab_size + 1])

    def iterate_through_qud(self, seq, seq_logprobs, data, idx, sample_n, split='test', rationality=1.,
                            save_dir=""):
        """

        :param seq:
        :param seq_logprobs:
        :param data:
        :param idx:
        :param sample_n: determined by the sample_n we pass into semantic_speaker()
        :param split:
        :param rationality:
        :param save_dir:
        :return:
        """
        # remember to add torch.no_grad() on the outside!
        # the frequency of file writing is a bit too high, can change this...

        # img_info = data['infos'][idx]
        # img_id = img_info['id']
        # qud_to_imgs_list = self.distractor_dataset.get_distractors_by_partition(img_id, split=split)

        img_id, qud_to_imgs_list = self.get_qud_to_imgs_list(data, idx, split)

        # once we get QUD, we can iterate through
        for i, qud_to_imgs in enumerate(qud_to_imgs_list):
            qud = qud_to_imgs[0]
            if len(qud_to_imgs[1]) == 0:
                continue

            pragmatic_matrix, samples_captions = self.pragmatic_speaker(seq, seq_logprobs, data, idx, qud_to_imgs,
                                                                        sample_n, split, rationality)
            pragmatic_matrix = pragmatic_matrix.cpu().numpy()

            target_id = img_id
            distractor_ids = qud_to_imgs[1]

            if save_dir != "":
                # we save JSON that has (qud, target_id, distractor_ids, samples_captions)
                # samples_captions always corresponds to the example
                # named "imgid_qud_0.json", and numpy for the matrix: "imgid_qud_0.npy"
                file_name = "{}_qud_{}".format(img_id, i)
                json.dump([qud, target_id, distractor_ids, samples_captions],
                          open(pjoin(save_dir, file_name + ".json"), 'w'))
                np.save(pjoin(save_dir, file_name), pragmatic_matrix)

        # we do not return anything?

    def get_qud_to_imgs_list(self, data, idx, split):
        img_info = data['infos'][idx]
        img_id = img_info['id']
        qud_to_imgs_list = self.distractor_dataset.get_cells_by_partition(img_id, split=split)

        return img_id, qud_to_imgs_list

    def pragmatic_speaker(self, seq, seq_logprobs, data, idx, qud_to_imgs, sample_n, split='test', rationality=1.):
        """

        TODO: consider unifying the API of this and semantic_speaker. You can always and should always call semantic_speaker() inside this

        Use `get_qud_to_imgs_list` to get relevant parts and then call this function...

        Unlike semantic speaker, this function does not support batched operation

        It should be samples from one data point.

        We base this function on model._sample() method (since it's easier)

        TODO: If we want partitions to be useful, we add a rule: generate literal captions for both
        TODO: distractors and target, and if the literal captions are already too different, for performance boost
        TODO: we should not include this partition
        TODO: we can do this by checking the similarity between generated captions

        :param seq: (num_samples, max_length)
        :param seq_logprobs: (num_samples, max_length, vocab_size)
        :param data: original data sent into semantic speaker, we use it to grab image id
        :param idx: the index that we use to grab that image from data as well as seq, seq_log_prob
        :param qud_to_imgs: we ask the outside to directly pass in qud_to_imgs:
                    ('What position is this man playing?', [337038, 518843, 211172])
                    This can be obtained by self.distractor_dataset.get_distractors_by_partition(img_id, split=split)
                    Then index in: qud_to_imgs_list[1][1] means we take the 2nd QuD, get the list of images
        :return:

        TODO: could try to verify this pipeline again, and write some unit test...
        """
        samples = seq[idx]  # this will be [sample_n, self.max_length]
        samples_logprob = seq_logprobs[idx].gather(2, seq[idx].unsqueeze(2)).squeeze(
            2)  # this will be [sample_n, self.max_length]

        samples_captions = self.decode_sequence(samples)

        # [sample_n]
        sample_logprob_for_nsamples = samples_logprob.sum(1)

        # Inner loop is deisnged for ONE QUD

        distractor_data = self.distractor_dataset.get_batch(qud_to_imgs[1])  # we batch over distractor data

        fc_feats, att_feats, att_masks = self.prep_data(distractor_data)

        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self.model._prepare_feature(fc_feats, att_feats, att_masks)

        other_seq = fc_feats.new_zeros((batch_size, sample_n, self.model.seq_length), dtype=torch.long)
        other_seqLogprobs = fc_feats.new_zeros(batch_size, sample_n, self.model.seq_length, self.model.vocab_size + 1)

        for n in range(sample_n):
            state = self.model.init_hidden(batch_size)  # * sample_n
            for t in range(self.model.seq_length + 1):
                if t == 0:  # input <bos>
                    it = fc_feats.new_zeros(batch_size, dtype=torch.long)

                logprobs, state = self.model.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks,
                                                                state, output_logsoftmax=True)

                # greedy sample from model.sample_next_word()
                # it, sampleLogprobs = model.sample_next_word(logprobs, 'greedy', temperature)
                #     sampleLogprobs, it = torch.max(logprobs.data, 1)
                #     it = it.view(-1).long()

                # but what we do is that we pick out the logprob from our original sequence
                single_token_it = seq[idx][n][t]
                it = single_token_it.expand(batch_size)

                if t == 0:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                it = it * unfinished.type_as(it)
                other_seq[:, n, t] = it
                other_seqLogprobs[:, n, t] = logprobs
                # quit loop if all sequences have finished
                if unfinished.sum() == 0:
                    break

        # [num_distractor_images, n_samples]
        # n_samples come from
        distractor_seqLogprobs_nsamples = other_seqLogprobs.gather(3, other_seq.unsqueeze(3)).squeeze(-1).sum(-1)

        # we build the literal matrix
        # [num_distractor_images+1, captions]
        literal_matrix = self.build_literal_matrix(sample_logprob_for_nsamples, distractor_seqLogprobs_nsamples)

        # now we can do RSA
        pragmatic_matrix = self.compute_pragmatic_speaker(literal_matrix, rationality)

        # first row corresponds to the re-weighted log-prob for each caption
        return pragmatic_matrix, samples_captions

    def decode_sequence(self, seq):
        return self.model.decode_sequence(seq)


def generate_full_rsa_pragmatic_caption_dist(argstring=None, beam_size=10, n_sample=10, num=500, split='test',
                                             rationality=1.0,
                                             save_dir=""):
    """
    Two loops:
    Use batched data to iterate through test set
    Then go through batched data one by one (using `idx` argument)
    To generate pragmatic distribution for each data point (using `iterate_qud`)

    :param rsa_model:
    :param distractor_dataset:
    :param num:
    :param split:
    :return:
    """
    assert beam_size == n_sample or n_sample == 1

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        rsa_dataset = VQADistractorDataset("/home/anie/PragmaticVQA/data/vqa/")
        model, loader, opt = load_model(argstring, beam_size=beam_size)
        loader.reset_iterator("test")

        rsa_model = FullRSAModel(model, loader, opt, rsa_dataset)
        batch_size = rsa_model.batch_size

        for _ in tqdm(range(int(num / batch_size))):

            data = loader.get_batch('test')
            # reshaped into [batch_size, n_sample, max_length] and [batch_size, n_sample, max_length, vocab_size]
            seq, seq_logprobs = rsa_model.semantic_speaker(data, n_sample)

            for idx in range(batch_size):
                # returns nothing...
                rsa_model.iterate_through_qud(seq, seq_logprobs, data, idx, n_sample, split, rationality, save_dir)

            # break out
            # if data['bounds']['it_pos_now'] >= num:
            #     break


class IncRSAModel(RSA):
    """
    IncRSAModel, we might be able to do n-gram reweight, instead of weighting by words/characters
    We can weight by P((d, e)|(a, b, c)) vs. P((d, e)|(a', b', c'))
    -------
    We will be careful for

    """

    def __init__(self, model, loader, opt, distractor_dataset, lm_model=None):
        super().__init__()

        self.model = model
        self.loader = loader
        self.opt = opt
        self.distractor_dataset = distractor_dataset
        self.lm_model = lm_model

        eval_kwargs = vars(opt)

        self.verbose = eval_kwargs.get('verbose', True)
        self.verbose_beam = eval_kwargs.get('verbose_beam', 1)
        self.verbose_loss = eval_kwargs.get('verbose_loss', 1)
        self.num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
        self.split = eval_kwargs.get('split', 'val')
        self.lang_eval = eval_kwargs.get('language_eval', 0)
        self.dataset = eval_kwargs.get('dataset', 'coco')
        self.beam_size = eval_kwargs.get('beam_size', 1)
        self.sample_n = eval_kwargs.get('sample_n', 1)  # this one will be ignored and updated later on

        self.batch_size = opt.batch_size
        self.max_length = eval_kwargs.get('max_length', 20)

        self.vocab_size = opt.vocab_size

        remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
        os.environ["REMOVE_BAD_ENDINGS"] = str(
            remove_bad_endings)  # Use this nasty way to make other code clean since it's a global configuration

        self.eval_kwargs = eval_kwargs

    def sample_data(self, split):
        """
        Sample a batch from loader
        Loader is the same used for training.

        :return:
        """
        data = self.loader.get_batch(split)
        return data

    def prep_data(self, data):
        """
        This is not the same as model._prepare_feature()
        This only loads data into CUDA

        :param data: whatever is returned by sample_data()
        :return:
        """
        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'],
               data['att_feats'],
               data['att_masks']]
        # this might cause issue...CUDA memory in this program is not quite freed
        tmp = [_.cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, att_masks = tmp
        return fc_feats, att_feats, att_masks

    # TODO: expand here; allow taking in direct list
    # TODO: if they are both None, then we are fine
    def rsa_aug_beam_search(self, img_id, split, qud_id, rationality, speaker_prior, speaker_prior_lm, delta_rsa,
                            entropy_penalty_alpha, dis_img_list, sim_img_list,
                            init_state, init_logprobs, init_lm_state, init_lm_logprobs, *args, **kwargs):
        """
        Original: CaptionModel.beam_search()

        IncRSA operation manual:
        At time step t,
        Assume I generated {w_0, w_1, …, w_t-1} for the gold image i, and I have a distractor image d, my conditional model is p:
        now my goal is to get the p(w_t | w_<t, i).
        So I evaluate on distractor image p(w_t | w_<t, d) , and build the matrix from there.

        So to translate into batch-based processing.

        We just augment the probability, and let beam search take care of the rest, including adding diversity.

        If the goal is that we need the original log-prob, we could just evaluate it ourselves, no need to add to this method
        (which is already too complex)

        So this returns RSA-augmented logprob!

        This function is actually very well-written (because you have access to all of your past choices)

        :param img_id: from data['info']
        :param split: 'train' 'valid' 'test'
        :param qud_id: just a number. Need to rerun this multiple times for each partition.
        :param init_state: [beam_size]
        :param init_logprobs: [beam_size, vocab_size]
        :param speaker_prior_lm: turn on language modeling
        :param args:
        :param kwargs:
        :return:
        """

        # function computes the similarity score to be augmented
        def add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash):
            # prev_choice I think is the choice of other beam groups?
            # beam_seq_table[prev_choice] gives a group list
            # bdash is the size of the beam group
            local_time = t - divm
            unaug_logprobsf = logprobsf.clone()
            for prev_choice in range(divm):
                prev_decisions = beam_seq_table[prev_choice][local_time]
                for sub_beam in range(bdash):
                    for prev_labels in range(bdash):
                        # here it's updating the value directly
                        logprobsf[sub_beam][prev_decisions[prev_labels]] = logprobsf[sub_beam][prev_decisions[
                            prev_labels]] - diversity_lambda
            return unaug_logprobsf

        # similar to add diversity, we augment the logprob!
        # the goal is to construct |N| x |V| matrix
        def add_rsa(beam_seq_table, logprobsf, t, divm, bdash, dist_logprobs_list, lm_logprobsf=None):
            """
            The goal is to build the matrix |N+1| x |V|

            So |N| x |V| is dist_logprobs_list! It already has it!

            We iterate through bdash (sub_beam size) because we are running RSA at every time step of each beam

            Step 1: assemble these two into our desired matrix
            Step 2: do RSA stuff
            Step 3: do in-place replacement?

            :param beam_seq_table:
            :param logprobsf: [beam_size // group_size, vocab_size], bdash = beam_size // group_size
            :param t: time
            :param divm: group_size
            :param bdash: the size of the beam group
            :param dist_logprobs_list: index by group_size (bdash), need to iterate through them
            :param dist_state_list: same as above
            :return:
            """
            # return logprobsf
            for sub_beam in range(bdash):
                # print(logprobsf.shape)  # torch.Size([3, 9488])
                # print(logprobsf[sub_beam].shape) # torch.Size([9488])
                # print(dist_logprobs_list[sub_beam].shape)  # torch.Size([5, 9488])
                literal_matrix = self.build_literal_matrix(logprobsf[sub_beam], dist_logprobs_list[sub_beam])
                # pragmatic_matrix = self.compute_pragmatic_speaker(literal_matrix, rationality=4.0, speaker_prior=True) # True, remove speaker prior for now
                if num_similar_images == 0:
                    pragmatic_matrix = self.compute_pragmatic_speaker(literal_matrix,
                                                                  rationality=rationality, speaker_prior=speaker_prior,
                                                                  lm_logprobsf=lm_logprobsf)
                    pragmatic_array = pragmatic_matrix[0]
                else:
                    pragmatic_array = self.compute_pragmatic_speaker_w_similarity(literal_matrix, num_similar_images,
                                                                      rationality=rationality, speaker_prior=speaker_prior,
                                                                      entropy_penalty_alpha=entropy_penalty_alpha,
                                                                      lm_logprobsf=lm_logprobsf)  # pragmatic_matrix =
                logprobsf[sub_beam] = pragmatic_array  # pragmatic_matrix[0]  # take the first row as the augmented prob

            # we are not adding state/it here. We do it at the end of the beam_search for the target image
            return logprobsf

        # does one step of classical beam search
        def beam_step(logprobsf, unaug_logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            # INPUTS:
            # logprobsf: probabilities augmented after diversity
            # beam_size: obvious
            # t        : time instant
            # beam_seq : tensor contanining the beamspragmatic_speaker
            # beam_seq_logprobs: tensor contanining the beam logprobs
            # beam_logprobs_sum: tensor contanining joint logprobs
            # OUPUTS:
            # beam_seq : tensor containing the word indices of the decoded captions
            # beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            # beam_logprobs_sum : joint log-probability of each beam

            ys, ix = torch.sort(logprobsf, 1, True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols):  # for each column (word, essentially)
                for q in range(rows):  # for each beam expansion
                    # compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q, c].item()
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    # local_unaug_logprob = unaug_logprobsf[q,ix[q,c]]
                    candidates.append({'c': ix[q, c], 'q': q, 'p': candidate_logprob, 'r': unaug_logprobsf[q]})
            candidates = sorted(candidates, key=lambda x: -x['p'])

            new_state = [_.clone() for _ in state]
            # beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1:
                # we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                # fork beam index q into index vix
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                # rearrange recurrent states
                for state_ix in range(len(new_state)):
                    #  copy over state in previous beam q to new beam at vix
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']]  # dimension one is time step
                # append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c']  # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r']  # the raw logprob here
                beam_logprobs_sum[vix] = v['p']  # the new (sum) logprob along this beam
            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates

        # Start diverse_beam_search
        opt = kwargs['opt']
        temperature = opt.get('temperature', 1)  # This should not affect beam search, but will affect dbs
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)
        suppress_UNK = opt.get('suppress_UNK', 0)
        length_penalty = utils.penalty_builder(opt.get('length_penalty', ''))
        bdash = beam_size // group_size  # beam per group

        # ==== Inc RSA Initialization =====
        # grab distractor images, run to get init_state, init_logprobs
        # tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, opt=opt

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        qud_to_imgs_list = self.distractor_dataset.get_cells_by_partition(img_id, split=split, qud_id=qud_id)

        # compile a list of images
        # dis_img_list, sim_img_list
        if dis_img_list is None:
            dis_img_list = qud_to_imgs_list[qud_id][1]
        if sim_img_list is None:
            sim_img_list = qud_to_imgs_list[qud_id][2]

        # print(dis_img_list)
        # print(sim_img_list)

        if delta_rsa:
            num_similar_images = len(sim_img_list)  # qud_to_imgs_list[qud_id][2]
            # imgs_list = qud_to_imgs_list[qud_id][2] + qud_to_imgs_list[qud_id][1]
            imgs_list = sim_img_list + dis_img_list # qud_to_imgs_list[qud_id][2] + qud_to_imgs_list[qud_id][1]
        else:
            num_similar_images = 0
            # imgs_list = qud_to_imgs_list[qud_id][1]  # get the list of distractor images
            imgs_list = dis_img_list # qud_to_imgs_list[qud_id][1]  # get the list of distractor images

        distractor_data = self.distractor_dataset.get_batch(imgs_list)
        fc_feats, att_feats, att_masks = self.prep_data(distractor_data)

        distractor_batch_size = fc_feats.size(0)  # this batch_size should be distractor set size
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self.model._prepare_feature(fc_feats, att_feats, att_masks)

        # for each "beam", we have 5 distractor image log-prob
        # [beam_size x num_distractor_images x vocab_size]
        dist_logprobs_list = []
        # [beam_size, num_distractor_images, ?]
        dist_state_list = []

        # for each distractor image, we repeat it beam_size
        # this is not very efficient, but easy to process
        for k in range(beam_size):
            # we only run the FIRST timestep
            it = fc_feats.new_zeros([distractor_batch_size], dtype=torch.long)
            dist_state = self.model.init_hidden(distractor_batch_size)  # we don't need n_sample I think

            dist_logprobs, dist_state = self.model.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats,
                                                                      p_att_masks,
                                                                      dist_state, output_logsoftmax=True)
            dist_logprobs_list.append(dist_logprobs)
            dist_state_list.append(dist_state)

        # so the chunking worked!
        # divm => index on groups
        # we can do dist_state_list_sub_groups[divm][i] and i on each beam
        # each element retrieved will be 5, 5 distractor images

        dist_logprobs_list_sub_groups = list(chunks(dist_logprobs_list, bdash))
        dist_state_list_sub_groups = list(chunks(dist_state_list, bdash))

        # print(dist_state_list_sub_groups[0])
        # print(dist_state_list_sub_groups[1])
        # each one is beam_size of 3 on the length of list
        # each bdash is of 5 distractor images, they need to match what's been generated by the true model

        if 'Transformer' not in self.model.__class__.__name__:
            raise Exception("Due to our RSA distractor implementation, it won't work with LSTM/RNN based architecture")

        # ===== Inc RSA initialization ends =====

        # INITIALIZATIONS
        beam_seq_table = [torch.LongTensor(self.model.seq_length, bdash).zero_() for _ in range(group_size)]
        beam_seq_logprobs_table = [torch.FloatTensor(self.model.seq_length, bdash, self.model.vocab_size + 1).zero_()
                                   for _ in
                                   range(group_size)]
        beam_logprobs_sum_table = [torch.zeros(bdash) for _ in range(group_size)]

        # logprobs # logprobs predicted in last time step, shape (beam_size, vocab_size+1)
        done_beams_table = [[] for _ in range(group_size)]
        # state_table = [list(torch.unbind(_)) for _ in torch.stack(init_state).chunk(group_size, 2)]
        state_table = list(zip(*[_.chunk(group_size, 1) for _ in init_state]))
        logprobs_table = list(init_logprobs.chunk(group_size, 0))
        # END INIT

        # ====== Inc RSA Language Model initialization starts =======

        # So since the LM for all distractor images will be the SAME, we treat it normally
        # No need to break into distractor chunks, just keep the same table as the overall loop

        # lm_beam_seq_table = [torch.LongTensor(self.model.seq_length, bdash).zero_() for _ in range(group_size)]
        # lm_beam_seq_logprobs_table = [torch.FloatTensor(self.model.seq_length, bdash, self.model.vocab_size + 1).zero_()
        #                            for _ in
        #                            range(group_size)]

        if self.lm_model and speaker_prior_lm:
            lm_state_table = list(zip(*[_.chunk(group_size, 1) for _ in init_lm_state]))
            lm_logprobs_table = list(init_lm_logprobs.chunk(group_size, 0))

        # ====== Inc RSA Language Model initialization ends =======

        # Chunk elements in the args
        # this is done because features were prepped in beam_size, it needs to be divided into sub-groups
        args = list(args)
        if self.__class__.__name__ == 'AttEnsemble':
            args = [[_.chunk(group_size) if _ is not None else [None] * group_size for _ in args_] for args_ in
                    args]  # arg_name, model_name, group_name
            args = [[[args[j][i][k] for i in range(len(self.model.models))] for j in range(len(args))] for k in
                    range(group_size)]  # group_name, arg_name, model_name
        else:
            args = [_.chunk(group_size) if _ is not None else [None] * group_size for _ in args]
            args = [[args[i][j] for i in range(len(args))] for j in range(group_size)]

        for t in range(self.model.seq_length + group_size - 1):
            for divm in range(group_size):
                # for group_size = 2, then t=0, divm=0,1, when t=0, divm=1, this condition is false, so no update on logprobs_table[1]
                # local_time = t - divm
                # when t = 1, divm=0, 1, then divm=0 updates as normal (local_time=1), divm=1, update for first time, local_time=0
                # We need to just work with this iteration strategy...we should be agnostic
                if t >= divm and t <= self.model.seq_length + divm - 1:
                    # add diversity
                    logprobsf = logprobs_table[divm].float()  # get the correct beam group's table
                    # [beam_size // group_size, vocab]

                    # before a lot of "tricks" to modify logprob were added
                    # we do RSA re-weighting, so that IncRSA can be combined with
                    # other techniques for decoding, also not affected by things like -inf log-prob

                    # So we are calculating Inc-RSA within the beam group!
                    # It's the "diversity" score that pushes DBS to choose something different
                    # Once DBS sequence diverges between groups, RSA will be different!

                    # we change it here and really don't have to worry about anything else.
                    # this function is in-place...

                    # get language model stuff
                    lm_logprobsf = lm_logprobs_table[divm].float() if speaker_prior_lm and self.lm_model else None

                    logprobsf = add_rsa(beam_seq_table, logprobsf, t, divm, bdash, dist_logprobs_list_sub_groups[divm], lm_logprobsf)

                    # suppress previous word
                    if decoding_constraint and t - divm > 0:
                        logprobsf.scatter_(1, beam_seq_table[divm][t - divm - 1].unsqueeze(1).cuda(), float('-inf'))
                    if remove_bad_endings and t - divm > 0:
                        logprobsf[torch.from_numpy(
                            np.isin(beam_seq_table[divm][t - divm - 1].cpu().numpy(),
                                    self.model.bad_endings_ix)), 0] = float(
                            '-inf')
                    # suppress UNK tokens in the decoding
                    if suppress_UNK and hasattr(self, 'vocab') and self.model.vocab[
                        str(logprobsf.size(1) - 1)] == 'UNK':
                        logprobsf[:, logprobsf.size(1) - 1] = logprobsf[:, logprobsf.size(1) - 1] - 1000

                    # diversity is added here
                    # the function directly modifies the logprobsf values and hence, we need to return
                    # the unaugmented ones for sorting the candidates in the end. # for historical
                    # reasons :-)
                    unaug_logprobsf = add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash)

                    # does not notice any difference for state_table[divm]
                    # before and after...

                    # print("before...")
                    # print(state_table[divm])

                    # infer new beams
                    beam_seq_table[divm], \
                    beam_seq_logprobs_table[divm], \
                    beam_logprobs_sum_table[divm], \
                    state_table[divm], \
                    candidates_divm = beam_step(logprobsf,
                                                unaug_logprobsf,
                                                bdash,
                                                t - divm,
                                                beam_seq_table[divm],
                                                beam_seq_logprobs_table[divm],
                                                beam_logprobs_sum_table[divm],
                                                state_table[divm])

                    # print("after re-arranging...")
                    # print(state_table[divm])

                    # if time's up... or if end token is reached then copy beams
                    # unaug_p is our Inc-RSA changed distribution
                    for vix in range(bdash):
                        if beam_seq_table[divm][t - divm, vix] == 0 or t == self.model.seq_length + divm - 1:
                            final_beam = {
                                'seq': beam_seq_table[divm][:, vix].clone(),
                                'logps': beam_seq_logprobs_table[divm][:, vix].clone(),
                                'unaug_p': beam_seq_logprobs_table[divm][:, vix].sum().item(),
                                'p': beam_logprobs_sum_table[divm][vix].item()
                            }
                            final_beam['p'] = length_penalty(t - divm + 1, final_beam['p'])
                            done_beams_table[divm].append(final_beam)
                            # don't continue beams from finished sequences
                            beam_logprobs_sum_table[divm][vix] = -1000

                    # move the current group one step forward in time
                    # logprobs_table[divm] gets replaced, not the same as before
                    it = beam_seq_table[divm][t - divm]
                    logprobs_table[divm], state_table[divm] = self.model.get_logprobs_state(it.cuda(), *(
                            args[divm] + [state_table[divm]]))
                    logprobs_table[divm] = F.log_softmax(logprobs_table[divm] / temperature, dim=-1)

                    # ======= IncRSA Language Model prepare for next iteration for current group ======

                    # image doesn't matter, so here we are fine
                    # are we going to do temperature? Sure...
                    if self.lm_model and speaker_prior_lm:
                        lm_logprobs_table[divm], lm_state_table[divm] = self.lm_model.get_logprobs_state(it.cuda(), *(
                                                                                        args[divm] + [lm_state_table[divm]]))
                        lm_logprobs_table[divm] = F.log_softmax(lm_logprobs_table[divm] / temperature, dim=-1)

                    # ======= IncRSA prepare for next iteration for current group ======
                    """
                    Step 1: `it` has [sub_beam_size=bdash], we take each beam out
                    Step 2: Expand each beam's choice, send it through all distractors
                    """
                    # p_fc_feats, ..., etc. are distractor features fixed
                    # the "it" is different
                    # dist_state_list_sub_groups[divm]: [sub_group_size, number_of_distractor_image]

                    # it.shape: torch.Size([3])
                    # same as bdash

                    for sub_beam in range(bdash):
                        beam_it = it[sub_beam].expand(distractor_batch_size)
                        # we iterate over sub_group_size, update number_of_distractor_images as a batch
                        dist_logprobs_list_sub_groups[divm][sub_beam], dist_state_list_sub_groups[divm][
                            sub_beam] = self.model.get_logprobs_state(beam_it.cuda(),
                                                                      p_fc_feats, p_att_feats, pp_att_feats,
                                                                      p_att_masks,
                                                                      dist_state_list_sub_groups[divm][sub_beam],
                                                                      output_logsoftmax=True)

        # all beams are sorted by their log-probabilities
        done_beams_table = [sorted(done_beams_table[i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)]
        done_beams = sum(done_beams_table, [])
        return done_beams

    def sample_beam(self, split, data, qud_id,
                    rationality, speaker_prior, speaker_prior_lm, delta_rsa, entropy_penalty_alpha,
                    dis_img_list, sim_img_list,
                    fc_feats, att_feats, att_masks=None, opt={}):
        """
        This generates the output...the main loop

        This is NOT where RSA happens.

        :param split: BAD API design!!! Clearly you are not thinking this through....
        :param data: this is what returned after loader.get_batch() or rsa_model.sample()
                     It has image id!!

        :return:
        """
        beam_size = opt.get('beam_size', 5)
        sample_n = opt.get('sample_n', 5)
        # when sample_n == beam_size then each beam is a sample.
        assert sample_n == 1 or sample_n == beam_size, 'when beam search, sample_n == 1 or beam search'
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self.model._prepare_feature(fc_feats, att_feats, att_masks)

        # we keep sample_n this way because the whole code is written in this logic
        seq = fc_feats.new_zeros((batch_size * sample_n, self.model.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.model.seq_length, self.vocab_size + 1)

        # In the end, what we do is STILL beam search...
        # So our goal is still
        self.done_beams = [[] for _ in range(batch_size)]

        # iterate through the batch
        for k in range(batch_size):
            state = self.model.init_hidden(beam_size)
            # in here, batch_size is replaced with beam size
            tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks = self.model.repeat_tensors(beam_size,
                   p_fc_feats[k:k + 1], p_att_feats[k:k + 1], pp_att_feats[k:k + 1], p_att_masks[k:k + 1] if att_masks is not None else None)

            # print(p_fc_feats[k:k + 1].shape) # torch.Size([1, 1])
            # print(tmp_fc_feats.shape) # torch.Size([5, 1])

            if speaker_prior_lm and self.lm_model:
                lm_state = self.lm_model.init_hidden(beam_size)

            # Run one time step!
            for t in range(1):
                if t == 0:  # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, state = self.model.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats,
                                                                tmp_att_masks, state)

                if speaker_prior_lm and self.lm_model:
                    lm_logprobs, lm_state = self.lm_model.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats,
                                                                tmp_att_masks, lm_state)
                else:
                    lm_logprobs, lm_state = None, None

            # do beam search
            # logprobs: [beam_size, vocab_size]
            img_info = data['infos'][k]
            img_id = img_info['id']
            self.done_beams[k] = self.rsa_aug_beam_search(img_id, split, qud_id,
                                                          rationality, speaker_prior, speaker_prior_lm, delta_rsa,
                                                          entropy_penalty_alpha, dis_img_list, sim_img_list,
                                                          state, logprobs, lm_state, lm_logprobs, tmp_fc_feats,
                                                          tmp_att_feats, tmp_p_att_feats,
                                                          tmp_att_masks, opt=opt)

            # just return all sampled beams when sample_n == beam_size
            if sample_n == beam_size:
                for _n in range(sample_n):
                    seq[k * sample_n + _n, :] = self.done_beams[k][_n]['seq']
                    seqLogprobs[k * sample_n + _n, :] = self.done_beams[k][_n]['logps']
            else:
                seq[k, :] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
                seqLogprobs[k, :] = self.done_beams[k][0]['logps']
            # return the samples and their log likelihoods
        return seq, seqLogprobs

    def semantic_speaker(self, data, sample_n, group_size):
        """
        The goal of semantic speaker is that given an image, it can sample literal captions (S0)

        :param data: what's returned by loader.get_batch() or rsa_model.sample_data()
        :return:
        """
        assert sample_n == self.beam_size or sample_n == 1, "sample_n needs to be 1 or beam size"

        fc_feats, att_feats, att_masks = self.prep_data(data)
        with torch.no_grad():
            tmp_eval_kwargs = self.eval_kwargs.copy()
            tmp_eval_kwargs.update({'sample_n': sample_n})
            tmp_eval_kwargs.update({'group_size': group_size})
            seq, seq_logprobs = self.model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
            seq = seq.data
            # entropy = - (F.softmax(seq_logprobs, dim=2) * seq_logprobs).sum(2).sum(1) / ((seq > 0).float().sum(1) + 1)
            # perplexity = - seq_logprobs.gather(2, seq.unsqueeze(2)).squeeze(2).sum(1) / ((seq > 0).float().sum(1) + 1)

        if sample_n == 1:
            # so originally this would have been
            # torch.Size([25, 20])
            # torch.Size([25, 20, 9488])
            # but now we always reshape into 3D
            return seq.reshape([-1, 1, self.max_length]), seq_logprobs.reshape(
                [-1, 1, self.max_length, self.vocab_size + 1])
        else:
            return seq.reshape([-1, sample_n, self.max_length]), seq_logprobs.reshape(
                [-1, sample_n, self.max_length, self.vocab_size + 1])

    def pragmatic_speaker(self, data, sample_n, group_size, split, qud_id,
                          rationality=4.0, speaker_prior=True, speaker_prior_lm=None, delta_rsa=False,
                          entropy_penalty_alpha=0.0, dis_img_list=None, sim_img_list=None):
        """
        We can test if pragmatic speaker behaves the same as semantic speaker if we remove the RSA
        probability augmentation.
        :param data:
        :param sample_n:
        :return:
        """
        self.eval_kwargs['sample_n'] = sample_n
        self.eval_kwargs['group_size'] = group_size

        assert sample_n == self.beam_size or sample_n == 1, "sample_n needs to be 1 or beam size"

        fc_feats, att_feats, att_masks = self.prep_data(data)

        with torch.no_grad():
            tmp_eval_kwargs = self.eval_kwargs.copy()
            tmp_eval_kwargs.update({'sample_n': sample_n})

            seq, seq_logprobs = self.sample_beam(split, data, qud_id,
                                                 rationality, speaker_prior, speaker_prior_lm, delta_rsa,
                                                 entropy_penalty_alpha, dis_img_list, sim_img_list,
                                                 fc_feats, att_feats, att_masks,
                                                         opt=tmp_eval_kwargs)
            seq = seq.data

        if sample_n == 1:
            # so originally this would have been
            # torch.Size([25, 20])
            # torch.Size([25, 20, 9488])
            # but now we always reshape into 3D
            return seq.reshape([-1, 1, self.max_length]), seq_logprobs.reshape(
                [-1, 1, self.max_length, self.vocab_size + 1])
        else:
            return seq.reshape([-1, sample_n, self.max_length]), seq_logprobs.reshape(
                [-1, sample_n, self.max_length, self.vocab_size + 1])

    def train(self):
        # override training
        # in order for this to work, we need greedy decoding ready.
        # still do probabilistic re-weighting, recomputing the log-prob
        pass

    def decode_sequence(self, seq):
        return self.model.decode_sequence(seq)


if __name__ == '__main__':
    pass

    # TODO: write a "test" for Full-utterance RSA using "Pitcher" image.

    # Generate re-ranking results for full-utterance RSA
    # Might need to rerun some checks to make sure it actually is ok and works fine
    ## generate_full_rsa_pragmatic_caption_dist(None, 10, 10, 100, 'test', save_dir="/mnt/fs5/anie/VQACaptioning/gen_results/trans_nsc_beam_search_10")

    # ====== Test 2 =======
    # Inc RSAcompute_entropy
    model, loader, opt = load_model(None, batch_size=2, beam_size=5)
    distractor_dataset = VQADistractorDataset("/home/anie/PragmaticVQA/data/vqa/", loader)
    rsa_model = IncRSAModel(model, loader, opt, distractor_dataset)
    data = rsa_model.sample_data('test')
    #fc_feats, att_feats, att_masks = rsa_model.prep_data(data)
    #opt.sample_n = 6

    # this is ugly. We need to determine qud_id from outside and pass in
    # meaning we retrieve twice...well, but the retrieval is fast...so maybe not too bad
    # qud_id = 0
    # opt.group_size = 2
    # rsa_model.sample_beam('test', data, qud_id, fc_feats, att_feats, att_masks, vars(opt))

    # ==== Test 3 =======
    # in this test, both should return the same result, before we add RSA augmentation
    # print(rsa_model.semantic_speaker(data, sample_n=6, group_size=2)[0])
    # print(rsa_model.pragmatic_speaker(data, 6, 2, 'test', 0)[0])

    # ==== Test 4 ======
    # No sub-groups, no dbs, we test if it works
    # print(rsa_model.semantic_speaker(data, sample_n=5, group_size=1)[0])
    # print(rsa_model.pragmatic_speaker(data, 5, 1, 'test', 0)[0])

    # choose first example from the batch of 2
    print(rsa_model.decode_sequence(rsa_model.semantic_speaker(data, sample_n=5, group_size=1)[0][0]))
    print(rsa_model.decode_sequence(rsa_model.pragmatic_speaker(data, 5, 1, 'test', 0)[0][0]))

    # ==== Test 5 ====
    # maybe add_rsa,