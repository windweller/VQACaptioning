"""
Pythia VQA model
"""

import sys

sys.path.append('/mnt/fs5/anie/VQACaptioning/data/vqa_model_data/pythia')
sys.path.append('/mnt/fs5/anie/VQACaptioning/data/vqa_model_data/vqa-maskrcnn-benchmark')

import yaml
import cv2
import torch
import requests
import numpy as np
import gc
import torch.nn.functional as F
import pandas as pd

import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from IPython.display import display, HTML, clear_output
from ipywidgets import widgets, Layout
from io import BytesIO

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict

from pythia.utils.configuration import ConfigNode
from pythia.tasks.processors import VocabProcessor, VQAAnswerProcessor
from pythia.models.pythia import Pythia
from pythia.common.registry import registry
from pythia.common.sample import Sample, SampleList


def get_right_img_id(img_id):
    c_list = [c for c in img_id]
    zero_pad = ['0' for _ in range(12 - len(c_list))]
    return ''.join(zero_pad + c_list)


class PythiaDemo:
    TARGET_IMAGE_SIZE = [448, 448]
    CHANNEL_MEAN = [0.485, 0.456, 0.406]
    CHANNEL_STD = [0.229, 0.224, 0.225]

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self._init_processors()
        self.pythia_model = self._build_pythia_model()
        self.detection_model = self._build_detection_model()
        self.resnet_model = self._build_resnet_model()

    def _init_processors(self):
        with open(self.root_dir + "pythia.yaml") as f:
            config = yaml.load(f)

        config = ConfigNode(config)
        # Remove warning
        config.training_parameters.evalai_inference = True
        registry.register("config", config)

        self.config = config

        vqa_config = config.task_attributes.vqa.dataset_attributes.vqa2
        text_processor_config = vqa_config.processors.text_processor
        answer_processor_config = vqa_config.processors.answer_processor

        text_processor_config.params.vocab.vocab_file = self.root_dir + "vocabulary_100k.txt"
        answer_processor_config.params.vocab_file = self.root_dir + "answers_vqa.txt"
        # Add preprocessor as that will needed when we are getting questions from user
        self.text_processor = VocabProcessor(text_processor_config.params)
        self.answer_processor = VQAAnswerProcessor(answer_processor_config.params)

        registry.register("vqa2_text_processor", self.text_processor)
        registry.register("vqa2_answer_processor", self.answer_processor)
        registry.register("vqa2_num_final_outputs",
                          self.answer_processor.get_vocab_size())

    def _build_pythia_model(self):
        state_dict = torch.load(self.root_dir + 'pythia.pth')
        model_config = self.config.model_attributes.pythia
        model_config.model_data_dir = self.root_dir  # "/content/"
        model = Pythia(model_config)
        model.build()
        model.init_losses_and_metrics()

        if list(state_dict.keys())[0].startswith('module') and \
                not hasattr(model, 'module'):
            state_dict = self._multi_gpu_state_to_single(state_dict)

        model.load_state_dict(state_dict)
        model.to("cuda")
        model.eval()

        return model

    def _build_resnet_model(self):
        self.data_transforms = transforms.Compose([
            transforms.Resize(self.TARGET_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(self.CHANNEL_MEAN, self.CHANNEL_STD),
        ])
        resnet152 = models.resnet152(pretrained=True)
        resnet152.eval()
        modules = list(resnet152.children())[:-2]
        self.resnet152_model = torch.nn.Sequential(*modules)
        self.resnet152_model.to("cuda")

    def _multi_gpu_state_to_single(self, state_dict):
        new_sd = {}
        for k, v in state_dict.items():
            if not k.startswith('module.'):
                raise TypeError("Not a multiple GPU state of dict")
            k1 = k[7:]
            new_sd[k1] = v
        return new_sd

    def predict(self, url, question):
        with torch.no_grad():
            detectron_features = self.get_detectron_features(url)
            resnet_features = self.get_resnet_features(url)

            sample = Sample()

            processed_text = self.text_processor({"text": question})
            sample.text = processed_text["text"]
            sample.text_len = len(processed_text["tokens"])

            sample.image_feature_0 = detectron_features
            sample.image_info_0 = Sample({
                "max_features": torch.tensor(100, dtype=torch.long)
            })

            sample.image_feature_1 = resnet_features

            sample_list = SampleList([sample])
            sample_list = sample_list.to("cuda")

            scores = self.pythia_model(sample_list)["scores"]
            scores = torch.nn.functional.softmax(scores, dim=1)
            actual, indices = scores.topk(5, dim=1)

            top_indices = indices[0]
            top_scores = actual[0]

            probs = []
            answers = []

            for idx, score in enumerate(top_scores):
                probs.append(score.item())
                answers.append(
                    self.answer_processor.idx2word(top_indices[idx].item())
                )

        gc.collect()
        torch.cuda.empty_cache()

        return probs, answers

    def _build_detection_model(self):

        cfg.merge_from_file(self.root_dir + 'detectron_model.yaml')
        cfg.freeze()

        model = build_detection_model(cfg)
        checkpoint = torch.load(self.root_dir + 'detectron_model.pth',
                                map_location=torch.device("cpu"))

        load_state_dict(model, checkpoint.pop("model"))

        model.to("cuda")
        model.eval()
        return model

    def get_actual_image(self, image_path):
        if image_path.startswith('http'):
            path = requests.get(image_path, stream=True).raw
        else:
            path = image_path

        return path

    def _image_transform(self, image_path):
        path = self.get_actual_image(image_path)

        img = Image.open(path)
        im = np.array(img).astype(np.float32)
        im = im[:, :, ::-1]
        im -= np.array([102.9801, 115.9465, 122.7717])
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(800) / float(im_size_min)
        # Prevent the biggest axis from being more than max_size
        if np.round(im_scale * im_size_max) > 1333:
            im_scale = float(1333) / float(im_size_max)
        im = cv2.resize(
            im,
            None,
            None,
            fx=im_scale,
            fy=im_scale,
            interpolation=cv2.INTER_LINEAR
        )
        img = torch.from_numpy(im).permute(2, 0, 1)
        return img, im_scale

    def _process_feature_extraction(self, output,
                                    im_scales,
                                    feat_name='fc6',
                                    conf_thresh=0.2):
        batch_size = len(output[0]["proposals"])
        n_boxes_per_image = [len(_) for _ in output[0]["proposals"]]
        score_list = output[0]["scores"].split(n_boxes_per_image)
        score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
        feats = output[0][feat_name].split(n_boxes_per_image)
        cur_device = score_list[0].device

        feat_list = []

        for i in range(batch_size):
            dets = output[0]["proposals"][i].bbox / im_scales[i]
            scores = score_list[i]

            max_conf = torch.zeros((scores.shape[0])).to(cur_device)

            for cls_ind in range(1, scores.shape[1]):
                cls_scores = scores[:, cls_ind]
                keep = nms(dets, cls_scores, 0.5)
                max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                             cls_scores[keep],
                                             max_conf[keep])

            keep_boxes = torch.argsort(max_conf, descending=True)[:100]
            feat_list.append(feats[i][keep_boxes])
        return feat_list

    def masked_unk_softmax(self, x, dim, mask_idx):
        x1 = F.softmax(x, dim=dim)
        x1[:, mask_idx] = 0
        x1_sum = torch.sum(x1, dim=1, keepdim=True)
        y = x1 / x1_sum
        return y

    def get_resnet_features(self, image_path):
        path = self.get_actual_image(image_path)
        img = Image.open(path).convert("RGB")
        img_transform = self.data_transforms(img)

        if img_transform.shape[0] == 1:
            img_transform = img_transform.expand(3, -1, -1)
        img_transform = img_transform.unsqueeze(0).to("cuda")

        features = self.resnet152_model(img_transform).permute(0, 2, 3, 1)
        features = features.view(196, 2048)
        return features

    def get_detectron_features(self, image_path):
        im, im_scale = self._image_transform(image_path)
        img_tensor, im_scales = [im], [im_scale]
        current_img_list = to_image_list(img_tensor, size_divisible=32)
        current_img_list = current_img_list.to('cuda')
        with torch.no_grad():
            output = self.detection_model(current_img_list)
        feat_list = self._process_feature_extraction(output, im_scales,
                                                     'fc6', 0.2)
        return feat_list[0]


# TODO: add a batch image processing method
# TODO: this is kinda important after seeing how f*cking slow this model is!!!!!

import glob
from os.path import join as pjoin
from tqdm import tqdm


class VQAHelper(object):

    def __init__(self, vqa_demo, eval_batch_size=8):
        # eval_batch_size=8 is tested on TITAN X 12GB memory
        # Masked-RCNN takes a lot of memory

        image_val_test_root_path = "/mnt/fs5/anie/VQACaptioning/data/mscoco/val2014/"
        image_train_root_path = "/mnt/fs5/anie/VQACaptioning/data/mscoco/train2014/"

        # we collect all file names in here...
        image_id_to_file_names = {}

        for root_path in [image_train_root_path, image_val_test_root_path]:
            for name in glob.glob(root_path + '*.jpg'):
                img_id = int(name.replace(root_path, "").strip(".jpg").split('_')[-1])
                image_id_to_file_names[img_id] = name

        self.image_id_to_file_names = image_id_to_file_names
        self.vqa_demo = vqa_demo
        self.eval_batch_size = eval_batch_size

    def filter_img_ids_by_answers(self, qud, target_answer, img_ids, silent=True):
        """
        return a tuple: [(img_id, ans), ...]
        :param img_ids:
        :return:
        """
        distractor_img_ids = []
        distractor_answers = []
        similar_img_ids = []
        similar_answers = []

        # this is for Masked-RCNN
        assert len(img_ids) < 32 or len(img_ids) % 32 == 0

        image_paths = [self.image_id_to_file_names[img_id] for img_id in img_ids]

        # we can only process 8 images at a time
        # at TITAN X: 12GB memory
        for i in tqdm(range(0, len(img_ids), self.eval_batch_size), disable=silent):
            curr_image_paths = image_paths[i:i+self.eval_batch_size]
            curr_img_ids = img_ids[i:i+self.eval_batch_size]
            batch_scores, batch_answers = self.predict(curr_image_paths, qud)
            batch_top_ans = [ans[0] for ans in batch_answers]  # batch size

            for i, top_ans in enumerate(batch_top_ans):
                if top_ans != target_answer:
                    distractor_img_ids.append(curr_img_ids[i])
                    distractor_answers.append(top_ans)
                else:
                    similar_img_ids.append(curr_img_ids[i])
                    similar_answers.append(top_ans)

        # if silent:
        #     for img_id in img_ids:
        #         img_path = self.image_id_to_file_names[img_id]
        #         scores, answers = self.vqa_demo.predict(img_path, qud)
        #         top_ans = answers[0]
        #         if top_ans != target_answer:
        #             distractor_img_ids.append(img_id)
        #             distractor_answers.append(top_ans)
        #         else:
        #             similar_img_ids.append(img_id)
        #             similar_answers.append(top_ans)
        # else:
        #     for img_id in tqdm(img_ids):
        #         img_path = self.image_id_to_file_names[img_id]
        #         scores, answers = self.vqa_demo.predict(img_path, qud)
        #         top_ans = answers[0]
        #         if top_ans != target_answer:
        #             distractor_img_ids.append(img_id)
        #             distractor_answers.append(top_ans)
        #         else:
        #             similar_img_ids.append(img_id)
        #             similar_answers.append(top_ans)

        return distractor_img_ids, distractor_answers, similar_img_ids, similar_answers

    def get_detectron_features(self, image_paths):
        img_tensor, im_scales = [], []
        for img_path in image_paths:
            im, im_scale = self.vqa_demo._image_transform(img_path)
            img_tensor.append(im)
            im_scales.append(im_scale)

        current_img_list = to_image_list(img_tensor, size_divisible=32)
        current_img_list = current_img_list.to('cuda')
        with torch.no_grad():
            output = self.vqa_demo.detection_model(current_img_list)
        feat_list = self.vqa_demo._process_feature_extraction(output, im_scales,
                                                              'fc6', 0.2)
        # length is the number of images we process
        return feat_list

    def get_resnet_features(self, image_paths):
        # we return [batch_size, 196, 2048]
        # originally, the demo returns [196, 2048]
        img_transforms = []
        for image_path in image_paths:
            path = self.vqa_demo.get_actual_image(image_path)
            img = Image.open(path).convert("RGB")
            img_transform = self.vqa_demo.data_transforms(img)
            # this line is for non-RGB images (not 3 channels)
            if img_transform.shape[0] == 1:
                img_transform = img_transform.expand(3, -1, -1)
            img_transforms.append(img_transform)
        img_transforms = torch.stack(img_transforms)
        img_transforms = img_transforms.to("cuda")

        features = self.vqa_demo.resnet152_model(img_transforms).permute(0, 2, 3, 1)
        features = features.view(-1, 196, 2048)

        return features

    def predict(self, img_paths, qud):
        """
        We enable batch prediction here
        :return:
        """
        with torch.no_grad():
            detectron_features = self.get_detectron_features(img_paths)  # a list of image features
            resnet_features = self.get_resnet_features(img_paths)  # [batch_size, 196, 2048]

            sample_list = []
            for i in range(len(detectron_features)):
                sample = Sample()
                processed_text = self.vqa_demo.text_processor({"text": qud})
                sample.text = processed_text["text"]
                sample.text_len = len(processed_text["tokens"])

                sample.image_feature_0 = detectron_features[i]
                sample.image_info_0 = Sample({
                    "max_features": torch.tensor(100, dtype=torch.long)
                })
                sample.image_feature_1 = resnet_features[i]
                sample_list.append(sample)

            sample_list = SampleList(sample_list)
            sample_list = sample_list.to("cuda")

            scores = self.vqa_demo.pythia_model(sample_list)["scores"]
            scores = torch.nn.functional.softmax(scores, dim=1)
            actual, indices = scores.topk(5, dim=1)

            batch_probs = []
            batch_answers = []

            for i in range(scores.shape[0]):
                top_indices = indices[i]
                top_scores = actual[i]

                probs = []
                answers = []

                for idx, score in enumerate(top_scores):
                    probs.append(score.item())
                    answers.append(
                        self.vqa_demo.answer_processor.idx2word(top_indices[idx].item())
                    )
                batch_probs.append(probs)
                batch_answers.append(answers)

        ## if the memory becomes an issue, we then clear this
        # gc.collect()
        # torch.cuda.empty_cache()

        # list is of batch_size
        # [[ans_1, ans_2], [ans_1, ans2]]
        return batch_probs, batch_answers
