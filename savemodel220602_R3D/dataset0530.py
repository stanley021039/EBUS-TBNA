import os
import cv2
import random
from tqdm import tqdm
from math import floor
import torch
import pandas as pd
import numpy as np
import matplotlib.patches as patches
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
import time
from utils import E_build, UD_npy_bulid

# crop_size(x1, x2, y1, y2)
cropping_size = {'r1': (748, 1452, 160, 736),
                 'r2': (716, 1484, 160, 840),
                 'r3': (684, 1516, 160, 992)}

E_cropping_size = {'Er1': (1010, 1610, 160, 695)}

de_id_size = {'de_id': (300, 1700, 125, 1100),
              'r1': (448, 1152, 35, 611),
              'r2': (416, 1184, 35, 715),
              'r3': (384, 1216, 35, 867)}


class Dataset3D(Dataset):  # numpy type data
    def __init__(self, video_dir, UD_dir, excel_dir, sample_rate, time_steps, number_classes, dataset_split, doppler=False, elastography=False, E_dir=None, augmentation=None, UD_reconstruct=False, E_reconstruct=False, crop_size='r1', E_crop_size='Er1', resize=None, E_resize=None):
        self.exc = pd.read_excel(excel_dir)
        self.isD = doppler
        self.isE = elastography
        self.video_dir = video_dir
        self.E_dir = E_dir
        self.UD_dir = UD_dir
        self.sample_rate = sample_rate
        self.time_steps = time_steps
        self.number_classes = number_classes
        self.augmentation = augmentation
        self.UD_reconstruct = UD_reconstruct
        self.E_reconstruct = E_reconstruct
        self.dataset_split = dataset_split
        self.crop_size = crop_size
        self.E_crop_size = E_crop_size
        self.LNs_exc = self.exc[self.exc['dataset'].isin(dataset_split)]
        self.slice_length = sample_rate * time_steps
        self.E_filename = []
        self.filename = []
        self.all_filename = []
        self.resize = resize
        self.E_resize = E_resize
        x1, x2, y1, y2 = cropping_size[self.crop_size]
        Ex1, Ex2, Ey1, Ey2 = E_cropping_size[self.E_crop_size]
        B_count_UD = 0
        B_count_LN = 0
        M_count_UD = 0
        M_count_LN = 0

        for root, dirs, files in os.walk(video_dir):  # all video name in data set
            for file in files:
                if '.mp4' in file:
                    self.all_filename.append(file)

        if self.E_reconstruct:
            E_build(self.LNs_exc, E_savedir=E_dir, vid_dir=video_dir)

        for folder in os.listdir(E_dir):  # all image name in elastic data set
            for LN in os.listdir(os.path.join(E_dir, folder)):
                for img in os.listdir(os.path.join(E_dir, folder, LN)):
                    self.E_filename.append((folder, LN, img))

        if self.UD_reconstruct:  # construct UD clips (npy)
            UD_npy_bulid(self.LNs_exc, UD_savedir=UD_dir, vid_dir=video_dir, cropping_size=cropping_size, time_steps=self.time_steps, sample_rate=self.sample_rate, size=self.crop_size, resize=self.resize)

        for (i, LN) in self.LNs_exc.iterrows():  # construct input list
            cls = str(LN[5])
            patient_folder = str(LN[7])
            video_folder = str(LN[8])
            if cls == "benign" or cls == "Benign":
                B_count_LN += 1
            else:
                M_count_LN += 1

            ln_npy_dir = os.path.join(UD_dir, patient_folder, video_folder)

            for clip in os.listdir(ln_npy_dir):
                clip_path = os.path.join(ln_npy_dir, clip)
                start_time, graph, percent = str(clip).replace('.npy', '').split('_')
                self.filename.append((video_folder, clip_path, start_time, graph, percent, cls))  # path, no., Stime, type, percentage, class

                if cls == "benign" or cls == "Benign":  # data count
                    B_count_UD += 1
                else:
                    M_count_UD += 1

        print(self.dataset_split, 'clips amount:', len(self.filename))
        print('-Benign lesions:', B_count_LN)
        print('-Malignant lesions:', M_count_LN)
        print('-Benign slices:', B_count_UD)
        print('-Malignant slices:', M_count_UD)

    def __getitem__(self, index):
        lesion_name, clip_npy_path, start_time, video_type, percentage, tumor_class = self.filename[index]
        _, _, folder, LN, clip_name = clip_npy_path.split('/')

        if tumor_class == "Benign" or tumor_class == "benign":
            label_list = 0
        else:
            label_list = 1

        x1, x2, y1, y2 = cropping_size[self.crop_size]
        if not self.resize:
            video = np.zeros((self.time_steps, y2 - y1, x2 - x1, 3))  # t, h, w, c
        else:
            video = np.zeros((self.time_steps, self.resize[0], self.resize[1], 3))  # t, h, w, c

        data_np = np.load(clip_npy_path) / 255.
        for i in range(self.time_steps):
            video[i] = cv2.resize(data_np[i], (video.shape[2], video.shape[1]))

        label = torch.tensor(label_list).float()
        video = torch.from_numpy(video).float()

        if self.isE:
            e_image = np.zeros(video.shape[1:])
            E_list = [E for E in self.E_filename
                      if E[0] == folder and E[1] == LN]

            if random.random() <= 0.8 and len(E_list) != 0:
                E_choice = random.choice(E_list)
                E_path = os.path.join(self.E_dir, E_choice[0], E_choice[1], E_choice[2])
            else:
                E_path = None

            if E_path is not None:
                e_image = cv2.resize(cv2.imread(E_path) / 255., (video.shape[2], video.shape[1]))
                if self.augmentation is not None:
                    img_list = self.augmentation([video, e_image])
                    video, e_image = img_list[0], img_list[1]
                else:
                    if self.augmentation is not None:
                        img_list = self.augmentation([video])
                        video = img_list[0]
                e_image = torch.from_numpy(e_image).float()
            else:
                if self.augmentation is not None:
                    img_list = self.augmentation([video])
                    video = img_list[0]
        if video_type == 'ultrasound':
            graph = torch.from_numpy(np.array([float(percentage), 1-float(percentage)])).float()
        else:
            graph = torch.from_numpy(np.array([1-float(percentage), float(percentage)])).float()
        if self.isE:
            return video, graph, label, e_image, lesion_name, start_time
        else:
            return video, graph, label, lesion_name, start_time

    def __len__(self):
        return len(self.filename)


