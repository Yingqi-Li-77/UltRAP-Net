"""
Date: April-1-2025
Author: Yingqi
This code is to construct the dataset of UltRAP-Net
"""
import random
import torch
import torch.utils.data as data
import os
import cv2
import numpy as np


def get_random_group(path_folder):
    group_list = os.listdir(path_folder)
    random_group_idx = random.choice((group_list))
    return random_group_idx


def get_fixed_idx(group_size, num_set):
    random_index = torch.arange(group_size)[torch.randperm(group_size)[:num_set]]
    return random_index


class MixImgDataset(data.Dataset):
    """
    This dataset: mix all the training data
    1. given a sequence name
    2. within the sequence, random select the combination index --> determine the label at the same time
    3. according to the combination index, read the corresponding images
    """

    def __init__(self, path_folder, num_set, seq_name, args):
        super(MixImgDataset, self).__init__()
        self.path_folder = path_folder
        self.num_set = num_set
        self.seq_name = seq_name
        self.seq_name_list, self.seq_length = self.get_seq_length()
        self.args = args

    def get_all_seq_names(self):
        all_seq = os.listdir(os.path.join(self.path_folder, self.seq_name))
        return all_seq

    def get_seq_length(self):
        seq_name_list = os.listdir(os.path.join(self.path_folder, self.seq_name))
        seq_length = len(seq_name_list)
        return seq_name_list, seq_length

    def get_random_idx(self, group_size):
        random_index = random.sample(np.arange(group_size).tolist(), self.num_set)
        return random_index

    def __getitem__(self, item):
        # 1. center the input image
        left_corner_x_label = self.args.left_corner_x_label
        left_corner_y_label = self.args.left_corner_y_label
        height_label = self.args.height_label
        width_label = self.args.width_label

        left_corner_x_input = self.args.left_corner_x_input
        left_corner_y_input = self.args.left_corner_x_input
        height_input = self.args.height_input
        width_input = self.args.width_input

        # 2. random select the input index
        selected_index = self.get_random_idx(self.seq_length)

        # 3. get the corresponding label
        label_name = os.path.splitext(self.seq_name)[0]
        label_path = os.path.join(self.args.label_data_path, label_name + '.png')
        label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label_img = label_img[left_corner_y_label:left_corner_y_label + width_label,
                    left_corner_x_label:left_corner_x_label + height_label]
        label_img = torch.tensor(cv2.resize(label_img, (512, 512)), device=self.args.device, dtype=torch.float32) / 255.

        # 4. get the input sets
        input_img = []
        for img_idx in selected_index:
            img_i = cv2.imread(os.path.join(self.path_folder, self.seq_name, self.seq_name_list[img_idx]),
                               cv2.IMREAD_GRAYSCALE)
            img_i = img_i[left_corner_y_input:left_corner_y_input + width_input,
                    left_corner_x_input:left_corner_x_input + height_input]
            img_i = torch.tensor(cv2.resize(img_i, (512, 512)), device=self.args.device, dtype=torch.float32) / 255.
            input_img.append(img_i)
        input_img = torch.stack(input_img)
        return input_img, label_img[None, :, :]

    def __len__(self):
        return 600


class MixImgTestDataset(data.Dataset):
    """
    This dataset: mix all the training data
    1. random choose the sequence
    2. within the sequence, random select the combination index --> determine the label at the same time
    3. according to the combination index, read the corresponding images
    """

    def __init__(self, path_folder, num_set, seq_name, args, fixed_idx=None):
        super(MixImgTestDataset, self).__init__()
        self.path_folder = path_folder
        self.num_set = num_set
        self.seq_name = seq_name
        self.seq_idx = 0
        self.fixed_idx = fixed_idx
        self.seq_name_list, self.seq_length = self.get_seq_length()
        self.args = args

    def get_seq_length(self):
        seq_name_list = os.listdir(os.path.join(self.path_folder, self.seq_name))
        seq_length = len(seq_name_list)
        return seq_name_list, seq_length

    def get_random_idx(self, group_size):
        random_index = random.sample(np.arange(group_size).tolist(), self.num_set)
        return random_index

    def __getitem__(self, item):
        # 1. center the input image
        left_corner_x_label = self.args.left_corner_x_label
        left_corner_y_label = self.args.left_corner_y_label
        height_label = self.args.height_label
        width_label = self.args.width_label

        left_corner_x_input = self.args.left_corner_x_input
        left_corner_y_input = self.args.left_corner_x_input
        height_input = self.args.height_input
        width_input = self.args.width_input

        # 2. random select the input index
        selected_index = self.fixed_idx

        # 3. get the corresponding label
        label_name = os.path.splitext(self.seq_name)[0]
        label_path = os.path.join(self.args.label_data_path, label_name + '.png')
        label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label_img = label_img[left_corner_y_label:left_corner_y_label + width_label,
                    left_corner_x_label:left_corner_x_label + height_label]
        label_img = torch.tensor(cv2.resize(label_img, (512, 512)), device=self.args.device, dtype=torch.float32) / 255.

        # 4. get the input sets
        input_img = []
        for img_idx in selected_index:
            img_i = cv2.imread(os.path.join(self.path_folder, self.seq_name, self.seq_name_list[img_idx]),
                               cv2.IMREAD_GRAYSCALE)
            img_i = img_i[left_corner_y_input:left_corner_y_input + width_input,
                    left_corner_x_input:left_corner_x_input + height_input]
            img_i = torch.tensor(cv2.resize(img_i, (512, 512)), device=self.args.device, dtype=torch.float32) / 255.
            input_img.append(img_i)
        input_img = torch.stack(input_img)

        return input_img, label_img[None, :, :]

    def __len__(self):
        return 20
