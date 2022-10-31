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
from utils import E_build, UD_npy_bulid, UD_2Dimg_bulid, label_smoothing
from scipy import ndimage

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# crop_size(x1, x2, y1, y2)
cropping_size = {'r1': (748, 1452, 160, 736),
                 'r2': (716, 1484, 160, 840),
                 'r3': (684, 1516, 160, 992)}

E_cropping_size = {'Er1': (1010, 1610, 160, 695)}

de_id_size = {'de_id': (300, 1700, 125, 1100),
              'r1': (448, 1152, 35, 611),
              'r2': (416, 1184, 35, 715),
              'r3': (384, 1216, 35, 867)}

# augmentations
def HorizontalFlip(img):
    return np.flip(img, 2)

def GaussianBlur(img, sigma=0.5):
    return ndimage.gaussian_filter(img, sigma)

def GaussianNoise(img, mean=0., std=0.1):
    if img.ndim == 3:
        noise = np.random.rand(img.shape[0], img.shape[1], img.shape[2])
    else:
        noise = np.random.rand(img.shape[0], img.shape[1], img.shape[2], img.shape[3])
    return img + noise * std + mean


class Dataset3D(Dataset):  # numpy type data
    def __init__(self, video_dir, excel_dir, UD_dir, E_dir=None, doppler=False, elastography=False, sample_rate=0.25,
                 time_steps=12, number_classes=1, dataset_split=['train'], augmentation=None, UD_reconstruct=False,
                 E_reconstruct=False, crop_size='r1', E_crop_size='Er1', resize=None, E_resize=None, UD_shape='CHWT',
                 E_shape='CHW', label_smooth=False, E_top=0):
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
        self.E_filedict = {}
        self.filename = []
        self.all_filename = []
        self.resize = resize
        self.E_resize = E_resize
        self.UD_shape = UD_shape
        self.E_shape = E_shape
        self.label_smooth = label_smooth
        self.E_top = E_top
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
            E_build(self.LNs_exc, E_savedir=E_dir, vid_dir=video_dir, resize=resize)

        for folder in os.listdir(E_dir):  # all image name in elastic data set
            for LN in os.listdir(os.path.join(E_dir, folder)):
                E_top_list = []
                self.E_filedict[LN] = []
                for img in os.listdir(os.path.join(E_dir, folder, LN)):
                    if E_top != 0 and len(E_top_list) >= E_top:
                        if float(img.split('_')[1].split('.')[0]) > min(E_top_list):
                            index = E_top_list.index(min(E_top_list))
                            self.E_filedict[LN][index] = os.path.join(self.E_dir, folder, LN, img)
                            E_top_list[index] = float(img.split('_')[1].split('.')[0])
                        else:
                            continue
                    else:
                        # print(LN, img)
                        self.E_filedict[LN].append(os.path.join(self.E_dir, folder, LN, img))
                        E_top_list.append(float(img.split('_')[1].split('.')[0]))
                self.E_filedict[LN].append('')

        if self.UD_reconstruct:  # construct UD clips (npy)
            UD_npy_bulid(self.LNs_exc, UD_savedir=UD_dir, vid_dir=video_dir, cropping_size=cropping_size,
                         time_steps=self.time_steps, sample_rate=self.sample_rate, size=self.crop_size,
                         resize=self.resize)

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
                start_time, graph, percent = str(clip).replace('.npy', '').split('_')
                if not self.isD:
                    if graph != 'grayscale' or percent != '1':
                        continue
                clip_path = os.path.join(ln_npy_dir, clip)
                if E_top == 0:
                    self.filename.append((video_folder, clip_path, self.E_filedict[video_folder], start_time, graph, percent,
                                          cls))  # path, no., Stime, type, percentage, class
                else:
                    for E_path in self.E_filedict[video_folder]:
                        self.filename.append(
                            (video_folder, clip_path, [E_path], start_time, graph, percent,
                             cls))  # path, no., Stime, type, percentage, class

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
        lesion_name, clip_npy_path, E_list, start_time, video_type, percentage, tumor_class = self.filename[index]
        _, _, _, folder, LN, clip_name = clip_npy_path.split('/')
        aug_dict = {'GaussianBlur': HorizontalFlip, 'AddGaussianNoise': GaussianNoise, 'RandomHorizontalFlip': HorizontalFlip}

        UD_transpose_dict = {'C': 3, 'H': 1, 'W': 2, 'T': 0}
        E_transpose_dict = {'C': 2, 'H': 0, 'W': 1}

        if tumor_class == "Benign" or tumor_class == "benign":
            label = 0
        else:
            label = 1

        label = torch.tensor(label).float()
        if self.label_smooth:
            label = label_smoothing(label)

        if video_type == 'grayscale':
            graph = torch.from_numpy(np.array([float(percentage), 1 - float(percentage)])).float()
        else:
            graph = torch.from_numpy(np.array([1 - float(percentage), float(percentage)])).float()

        x1, x2, y1, y2 = cropping_size[self.crop_size]
        if not self.resize:
            video = np.zeros((self.time_steps, y2 - y1, x2 - x1, 3))  # t, h, w, c
        else:
            video = np.zeros((self.time_steps, self.resize[0], self.resize[1], 3))  # t, h, w, c

        data_np = np.load(clip_npy_path) / 255.
        if data_np.shape == video.shape:
            video = data_np
        else:
            for i in range(self.time_steps):
                video[i] = cv2.resize(data_np[i], (video.shape[2], video.shape[1]))

        if self.isE:
            E_path = random.choice(E_list)
            if E_path:
                e_image = cv2.resize(cv2.imread(E_path) / 255., (video.shape[2], video.shape[1]))
            else:
                e_image = np.zeros(video.shape[1:])

            if self.augmentation is not None:
                for (aug, probability) in self.augmentation:
                    if random.random() < probability:
                        video = aug_dict[aug](video)
                        e_image = aug_dict[aug](e_image)

            E_transpose = (
                E_transpose_dict[self.E_shape[0]], E_transpose_dict[self.E_shape[1]], E_transpose_dict[self.E_shape[2]])
            e_image = torch.from_numpy(np.transpose(e_image.copy(), E_transpose)).float()
            UD_transpose = (
                UD_transpose_dict[self.UD_shape[0]], UD_transpose_dict[self.UD_shape[1]],
                UD_transpose_dict[self.UD_shape[2]],
                UD_transpose_dict[self.UD_shape[3]])
            video = torch.from_numpy(np.transpose(video.copy(), UD_transpose)).float()
            return video, graph, label, e_image, lesion_name, start_time

        else:
            if self.augmentation is not None:
                for (aug, probability) in self.augmentation:
                    if random.random() < probability:
                        video = aug_dict[aug](video)
            UD_transpose = (
                UD_transpose_dict[self.UD_shape[0]], UD_transpose_dict[self.UD_shape[1]],
                UD_transpose_dict[self.UD_shape[2]],
                UD_transpose_dict[self.UD_shape[3]])
            video = torch.from_numpy(np.transpose(video.copy(), UD_transpose)).float()
            return video, graph, label, lesion_name, start_time

    def __len__(self):
        return len(self.filename)


class Dataset3D_MOCO(Dataset):  # numpy type data
    def __init__(self, video_dir, excel_dir, UD_dir, E_dir=None, doppler=False, elastography=False, sample_rate=0.25,
                 time_steps=12, number_classes=1, dataset_split=['train'], augmentation=None, UD_reconstruct=False,
                 E_reconstruct=False, crop_size='r1', E_crop_size='Er1', resize=None, E_resize=None,
                 UD_shape='CHWT', E_shape='CHW', label_smooth=False, E_top=0):
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
        self.E_filedict = {}
        self.filename = []
        self.all_filename = []
        self.resize = resize
        self.E_resize = E_resize
        self.UD_shape = UD_shape
        self.E_shape = E_shape
        self.label_smooth = label_smooth
        self.E_top = E_top
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
            E_build(self.LNs_exc, E_savedir=E_dir, vid_dir=video_dir, resize=resize)

        for folder in os.listdir(E_dir):  # all image name in elastic data set
            for LN in os.listdir(os.path.join(E_dir, folder)):
                E_top_list = []
                self.E_filedict[LN] = []
                for img in os.listdir(os.path.join(E_dir, folder, LN)):
                    if E_top != 0 and len(E_top_list) >= E_top:
                        if float(img.split('_')[1].split('.')[0]) > min(E_top_list):
                            index = E_top_list.index(min(E_top_list))
                            self.E_filedict[LN][index] = os.path.join(self.E_dir, folder, LN, img)
                            E_top_list[index] = float(img.split('_')[1].split('.')[0])
                        else:
                            continue
                    else:
                        # print(LN, img)
                        self.E_filedict[LN].append(os.path.join(self.E_dir, folder, LN, img))
                        E_top_list.append(float(img.split('_')[1].split('.')[0]))
                self.E_filedict[LN].append('')

        if self.UD_reconstruct:  # construct UD clips (npy)
            UD_npy_bulid(self.LNs_exc, UD_savedir=UD_dir, vid_dir=video_dir, cropping_size=cropping_size,
                         time_steps=self.time_steps, sample_rate=self.sample_rate, size=self.crop_size,
                         resize=self.resize)

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
                start_time, graph, percent = str(clip).replace('.npy', '').split('_')
                if not self.isD:
                    if graph != 'grayscale' or percent != '1':
                        continue
                clip_path = os.path.join(ln_npy_dir, clip)
                if E_top == 0:
                    self.filename.append((video_folder, clip_path, self.E_filedict[video_folder], start_time, graph, percent,
                                          cls))  # path, no., Stime, type, percentage, class
                else:
                    for E_path in self.E_filedict[video_folder]:
                        self.filename.append(
                            (video_folder, clip_path, [E_path], start_time, graph, percent,
                             cls))  # path, no., Stime, type, percentage, class

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
        lesion_name, clip_npy_path, E_list, start_time, video_type, percentage, tumor_class = self.filename[index]
        folder, LN, clip_name = clip_npy_path.split('/')[-3:]
        aug_dict = {'GaussianBlur': HorizontalFlip, 'AddGaussianNoise': GaussianNoise, 'RandomHorizontalFlip': HorizontalFlip}

        UD_transpose_dict = {'C': 3, 'H': 1, 'W': 2, 'T': 0}
        E_transpose_dict = {'C': 2, 'H': 0, 'W': 1}

        if tumor_class == "Benign" or tumor_class == "benign":
            label = 0
        else:
            label = 1

        label = torch.tensor(label).long()
        if self.label_smooth:
            label = label_smoothing(label)

        if video_type == 'grayscale':
            graph = torch.from_numpy(np.array([float(percentage), 1 - float(percentage)])).float()
        else:
            graph = torch.from_numpy(np.array([1 - float(percentage), float(percentage)])).float()

        x1, x2, y1, y2 = cropping_size[self.crop_size]
        if not self.resize:
            video = np.zeros((self.time_steps, y2 - y1, x2 - x1, 3))  # t, h, w, c
        else:
            video = np.zeros((self.time_steps, self.resize[0], self.resize[1], 3))  # t, h, w, c

        data_np = np.load(clip_npy_path) / 255.
        if data_np.shape == video.shape:
            video = data_np
        else:
            for i in range(self.time_steps):
                video[i] = cv2.resize(data_np[i], (video.shape[2], video.shape[1]))

        UD_transpose = (
            UD_transpose_dict[self.UD_shape[0]], UD_transpose_dict[self.UD_shape[1]],
            UD_transpose_dict[self.UD_shape[2]],
            UD_transpose_dict[self.UD_shape[3]])
        E_transpose = (
            E_transpose_dict[self.E_shape[0]], E_transpose_dict[self.E_shape[1]],
            E_transpose_dict[self.E_shape[2]])
        if self.isE:
            E_path = random.choice(E_list)
            if E_path:
                e_image = cv2.resize(cv2.imread(E_path) / 255., (video.shape[2], video.shape[1]))
            else:
                e_image = np.zeros(video.shape[1:])
            video_q = video
            video_k = video
            e_q = e_image
            e_k = e_image
            if self.augmentation is not None:  # MOCO train
                #aug_list = self.augmentation[:]
                #aug_q = random.choice(aug_list)
                #aug_list.remove(aug_q)
                #aug_k = random.choice(aug_list)
                for (aug, probability) in self.augmentation:
                    if random.random() < probability:
                        video_q = aug_dict[aug](video)
                        e_q = aug_dict[aug](e_image)
                    if random.random() < probability:
                        video_k = aug_dict[aug](video)
                        e_k = aug_dict[aug](e_image)
                video_q = torch.from_numpy(np.transpose(video_q.copy(), UD_transpose)).float()
                video_k = torch.from_numpy(np.transpose(video_k.copy(), UD_transpose)).float()
                e_q = torch.from_numpy(np.transpose(e_q.copy(), E_transpose)).float()
                e_k = torch.from_numpy(np.transpose(e_k.copy(), E_transpose)).float()
                return (video_q, e_q), (video_k, e_k), label, lesion_name, start_time
            else:  # valid or test
                video = torch.from_numpy(np.transpose(video, UD_transpose)).float()
                e_image = torch.from_numpy(np.transpose(e_image, E_transpose)).float()
                return video, e_image, label, lesion_name, start_time

        else:
            if self.augmentation is not None:
                video_q = video
                video_k = video
                for (aug, probability) in self.augmentation:
                    if random.random() < probability:
                        video_q = aug_dict[aug](video)
                    if random.random() < probability:
                        video_k = aug_dict[aug](video)
                video_q = torch.from_numpy(np.transpose(video_q, UD_transpose)).float()
                video_k = torch.from_numpy(np.transpose(video_k, UD_transpose)).float()
                return video_q, video_k, label, lesion_name, start_time
            else:
                video = torch.from_numpy(np.transpose(video, UD_transpose)).float()
                return video, label, lesion_name, start_time

    def __len__(self):
        return len(self.filename)


class DatasetEBUSNet(Dataset):
    def __init__(self, video_dir, U_img_dir, D_img_dir, E_img_dir, name_dir, sample_rate, max_img_pre_LN,
                 number_classes, split, augmentation=None, reconstruct=False):
        self.csv = pd.read_excel(name_dir)
        self.video_dir = video_dir
        self.U_img_dir = U_img_dir
        self.D_img_dir = D_img_dir
        self.E_img_dir = E_img_dir
        self.sample_rate = sample_rate
        self.max_img_pre_LN = max_img_pre_LN
        self.number_classes = number_classes
        self.augmentation = augmentation
        self.split = split
        self.reconstruct = reconstruct
        self.tumor_times = self.csv[self.csv['dataset'].isin(split)]
        self.filename = []
        self.all_video = []
        self.all_LN = []
        self.U_dict = {}
        self.D_dict = {}
        self.E_dict = {}
        B_lesion_count = 0
        M_lesion_count = 0
        B_img_count = 0
        M_img_count = 0
        for root, dirs, files in os.walk(video_dir):  # all video name in data set
            for file in files:
                if '.mp4' in file:
                    self.all_video.append(file)

        if self.reconstruct:
            UD_2Dimg_bulid(self.video_dir, self.U_img_dir, self.D_img_dir, self.tumor_times, self.sample_rate,
                           self.max_img_pre_LN)

        for (i, LN) in self.tumor_times.iterrows():
            LN_folder = str(LN[7])
            LN_video_name = str(LN[8])
            LN_num = str(LN[2])
            LN_class = str(LN[5])

            self.all_LN.append(LN_video_name)
            self.U_dict[LN_video_name] = []
            self.D_dict[LN_video_name] = [None]
            self.E_dict[LN_video_name] = [None, None, None]

            if LN_class == 'B' or LN_class == 'Benign' or LN_class == 'benign':
                B_lesion_count += 1
                self.U_dict[LN_video_name].append(0)
            else:
                M_lesion_count += 1
                self.U_dict[LN_video_name].append(1)
            self.U_dict[LN_video_name].append(None)

            U_img_folder = os.path.join(self.U_img_dir, LN_folder + '_' + LN_num)
            D_img_folder = os.path.join(self.D_img_dir, LN_folder + '_' + LN_num)
            E_img_folder = os.path.join(self.E_img_dir, LN_folder, LN_video_name)
            color_list = [0, 0, 0]
            for U_img_name in os.listdir(U_img_folder):
                self.U_dict[LN_video_name].append(os.path.join(U_img_folder, U_img_name))
                # U_list.append(os.path.join(U_img_folder, U_img_name))
            for D_img_name in os.listdir(D_img_folder):
                self.D_dict[LN_video_name].append(os.path.join(D_img_folder, D_img_name))
                # D_list.append(os.path.join(D_img_folder, D_img_name))
            for E_img_name in os.listdir(E_img_folder):
                color_area = int(E_img_name.replace('.jpg', '').split('_')[1])
                if color_area > color_list[0]:
                    self.E_dict[LN_video_name][2] = self.E_dict[LN_video_name][1]
                    self.E_dict[LN_video_name][1] = self.E_dict[LN_video_name][0]
                    self.E_dict[LN_video_name][0] = os.path.join(E_img_folder, E_img_name)
                    color_list[2] = color_list[1]
                    color_list[1] = color_list[0]
                    color_list[0] = color_area
                    continue
                if color_area > color_list[1]:
                    self.E_dict[LN_video_name][2] = self.E_dict[LN_video_name][1]
                    self.E_dict[LN_video_name][1] = os.path.join(E_img_folder, E_img_name)
                    color_list[2] = color_list[1]
                    color_list[1] = color_area
                    continue
                if color_area > color_list[2]:
                    self.E_dict[LN_video_name][2] = os.path.join(E_img_folder, E_img_name)
                    color_list[2] = color_area
                    continue

            '''for U_img in U_list:
                for D_img in D_list:
                    for E_img in E_list:
                        if LN_class == 'B' or LN_class == 'Benign' or LN_class == 'benign':
                            self.filename.append((U_img, D_img, E_img, 0, LN_video_name,
                                                  LN_num))  # U_path, D, E, label, lesion_name, no_
                            B_img_count += 1
                        else:
                            self.filename.append((U_img, D_img, E_img, 1, LN_video_name, LN_num))
                            M_img_count += 1
            '''
        print(self.split, 'data amount:', len(self.filename))
        print('-Benign lesion:', B_lesion_count)
        print('-Malignant lesion:', M_lesion_count)
        # print('-Benign pair:', B_img_count)
        # print('-Malignant pair:', M_img_count)

    def __getitem__(self, index):
        lesion_video_name = self.all_LN[floor(index / 10)]
        label = self.U_dict[lesion_video_name][0]
        U_path = random.choice(self.U_dict[lesion_video_name][1:])
        D_path = random.choice(self.D_dict[lesion_video_name])
        E_path = random.choice(self.E_dict[lesion_video_name])
        if U_path is None:
            U_image = np.zeros((224, 224, 3))
        else:
            U_image = cv2.resize(cv2.imread(U_path) / 255., (224, 224))
        if D_path is None:
            D_image = np.zeros((224, 224, 3))
        else:
            D_image = cv2.resize(cv2.imread(D_path) / 255., (224, 224))
        if E_path is None:
            E_image = np.zeros((224, 224, 3))
        else:
            E_image = cv2.resize(cv2.imread(E_path) / 255., (224, 224))
        if self.augmentation is not None:
            U_image, D_image, E_image = self.augmentation(U_image, D_image, E_image)

        label = torch.tensor(label).float()
        U_image = torch.from_numpy(U_image).float()
        D_image = torch.from_numpy(D_image).float()
        E_image = torch.from_numpy(E_image).float()

        return U_image, D_image, E_image, label, lesion_video_name

    def __len__(self):
        return len(self.all_LN) * 10


class Dataset2D(Dataset):
    def __init__(self, video_dir, name_dir, U_img_dir='data/2D_data/U_img', D_img_dir='data/2D_data/D_img',
                 E_img_dir='data/2D_data/E_img', sample_rate=0.25, max_img_pre_LN=20, number_classes=1,
                 split='train', resize=(224, 224), augmentation=None, reconstruct=False):
        self.csv = pd.read_excel(name_dir)
        self.video_dir = video_dir
        self.U_img_dir = U_img_dir
        self.D_img_dir = D_img_dir
        self.E_img_dir = E_img_dir
        self.sample_rate = sample_rate
        self.max_img_pre_LN = max_img_pre_LN
        self.number_classes = number_classes
        self.augmentation = augmentation
        self.split = split
        self.resize = resize
        self.reconstruct = reconstruct
        self.tumor_times = self.csv[self.csv['dataset'].isin(split)]
        self.filename = []
        self.all_video = []
        self.all_img = []
        self.U_dict = {}
        self.D_dict = {}
        self.E_dict = {}
        B_lesion_count = 0
        M_lesion_count = 0
        B_img_count = 0
        M_img_count = 0
        for root, dirs, files in os.walk(video_dir):  # all video name in data set
            for file in files:
                if '.mp4' in file:
                    self.all_video.append(file)

        if self.reconstruct:
            UD_2Dimg_bulid(self.video_dir, self.U_img_dir, self.D_img_dir, self.tumor_times, self.sample_rate,
                           self.max_img_pre_LN)

        for (i, LN) in self.tumor_times.iterrows():
            LN_folder = str(LN[7])
            LN_video_name = str(LN[8])
            LN_num = str(LN[2])
            LN_class = str(LN[5])
            U_LN_path = os.path.join(U_img_dir, LN_folder + '_' + LN_num)
            D_LN_path = os.path.join(D_img_dir, LN_folder + '_' + LN_num)
            E_LN_path = os.path.join(E_img_dir, LN_folder, LN_folder + '_' + LN_num + '.mp4')
            if LN_class == 'B' or LN_class == 'Benign' or LN_class == 'benign':
                cls = 0
                B_lesion_count += 1
            else:
                cls = 1
                M_lesion_count += 1

            if U_img_dir:
                for img in os.listdir(U_LN_path):
                    self.all_img.append((os.path.join(U_LN_path, img), cls, LN_video_name, 'grayscale'))
                    if LN_class == 'B' or LN_class == 'Benign' or LN_class == 'benign':
                        B_img_count += 1
                    else:
                        M_img_count += 1
            if D_img_dir:
                for img in os.listdir(D_LN_path):
                    self.all_img.append((os.path.join(D_LN_path, img), cls, LN_video_name, 'doppler'))
                    if LN_class == 'B' or LN_class == 'Benign' or LN_class == 'benign':
                        B_img_count += 1
                    else:
                        M_img_count += 1
            if E_img_dir:
                for img in os.listdir(E_LN_path):
                    self.all_img.append((os.path.join(E_LN_path, img), cls, LN_video_name, 'elastohraphy'))
                    if LN_class == 'B' or LN_class == 'Benign' or LN_class == 'benign':
                        B_img_count += 1
                    else:
                        M_img_count += 1

        print(self.split, 'data amount:', len(self.all_img))
        print('-Benign lesion:', B_lesion_count)
        print('-Malignant lesion:', M_lesion_count)
        print('-Benign pair:', B_img_count)
        print('-Malignant pair:', M_img_count)

    def __getitem__(self, index):
        img_path, label, vid_name, graph = self.all_img[index]
        img = cv2.resize(cv2.imread(img_path) / 255., self.resize)

        if self.augmentation is not None:
            img = self.augmentation(img)

        label = torch.tensor(label).float()
        img = torch.from_numpy(img).float()

        return img, label, vid_name

    def __len__(self):
        return len(self.all_img)


class Dataset3DSimCLR(Dataset):
    def __init__(self, video_dir, UD_dir, excel_dir, sample_rate, time_steps, doppler=False,
                 elastography=False, E_dir=None, augmentation=None, UD_reconstruct=False, E_reconstruct=False,
                 crop_size='r1', E_crop_size='Er1', resize=None, E_resize=None, UD_shape='CHWT', E_shape='CHW'):

        self.exc = pd.read_excel(excel_dir)
        self.isD = doppler
        self.isE = elastography
        self.video_dir = video_dir
        self.E_dir = E_dir
        self.UD_dir = UD_dir
        self.sample_rate = sample_rate
        self.time_steps = time_steps
        self.augmentation = augmentation
        self.UD_reconstruct = UD_reconstruct
        self.E_reconstruct = E_reconstruct
        self.crop_size = crop_size
        self.E_crop_size = E_crop_size
        self.slice_length = sample_rate * time_steps
        self.E_filename = []
        self.UD_filename = []
        self.all_vidname = []
        self.all_patients = []
        self.all_LNs = []
        self.resize = resize
        self.E_resize = E_resize
        self.UD_shape = UD_shape
        self.E_shape = E_shape
        x1, x2, y1, y2 = cropping_size[self.crop_size]
        Ex1, Ex2, Ey1, Ey2 = E_cropping_size[self.E_crop_size]

        for root, dirs, files in os.walk(video_dir):  # all video name in data set
            for file in files:
                if '.mp4' in file:
                    self.all_vidname.append(file)
        if self.isE:
            if self.E_reconstruct:
                E_build(self.exc, E_savedir=E_dir, vid_dir=video_dir)

            for patient in os.listdir(E_dir):  # all image name in elastic data set
                E_patient_dir = os.path.join(E_dir, patient)
                for LN in os.listdir(E_patient_dir):
                    E_LN_dir = os.path.join(E_patient_dir, LN)
                    self.E_filename.append((patient, LN, None, None))
                    self.E_filename.append((patient, LN, None, None))
                    for img in os.listdir(E_LN_dir):
                        E_img_dir = os.path.join(E_LN_dir, img)
                        self.E_filename.append((patient, LN, img, E_img_dir))

        if self.UD_reconstruct:  # construct UD clips (npy)
            UD_npy_bulid(self.exc, UD_savedir=UD_dir, vid_dir=video_dir, cropping_size=cropping_size,
                         time_steps=self.time_steps, sample_rate=self.sample_rate, size=self.crop_size,
                         resize=self.resize)

        for patient in os.listdir(UD_dir):  # all npy file name in UD dataset
            self.all_patients.append(patient)
            patient_dir = os.path.join(UD_dir, patient)
            for LN in os.listdir(patient_dir):
                LN_dir = os.path.join(patient_dir, LN)
                npy_count = 0
                for npy in os.listdir(LN_dir):
                    npy_dir = os.path.join(LN_dir, npy)
                    self.UD_filename.append((patient, LN, npy, npy_dir))
                    npy_count += 1
                if npy_count >= 2:
                    self.all_LNs.append(LN)

        print('LN count:', len(self.all_LNs))
        print('UD count:', len(self.UD_filename))
        print('E count:', len(self.E_filename))

    def __getitem__(self, index):

        UD_transpose_dict = {'C': 3, 'H': 1, 'W': 2, 'T': 0}
        E_transpose_dict = {'C': 2, 'H': 0, 'W': 1}

        LN_name = self.all_LNs[index]

        UD_pool = [UD for UD in self.UD_filename if UD[1] == LN_name]
        E_pool = [E for E in self.E_filename if E[1] == LN_name]

        UD_1, UD_2 = random.choices(UD_pool, k=2)
        UD_patient_1, UD_LN_1, UD_npy_1, UD_npy_dir_1 = UD_1
        UD_patient_2, UD_LN_2, UD_npy_2, UD_npy_dir_2 = UD_2
        start_time_1, video_type_1, percent_1 = UD_npy_1.replace('.npy', '').split('_')
        start_time_2, video_type_2, percent_2 = UD_npy_2.replace('.npy', '').split('_')

        x1, x2, y1, y2 = cropping_size[self.crop_size]

        data_np_1 = np.load(UD_npy_dir_1) / 255.
        data_np_2 = np.load(UD_npy_dir_2) / 255.
        if not self.resize:
            clip_1 = data_np_1
            clip_2 = data_np_2
        else:
            clip_1 = np.zeros((self.time_steps, self.resize[0], self.resize[1], 3))
            clip_2 = np.zeros((self.time_steps, self.resize[0], self.resize[1], 3))  # t, h, w, c
            for i in range(self.time_steps):
                clip_1[i] = cv2.resize(data_np_1[i], (clip_1.shape[2], clip_1.shape[1]))
            for i in range(self.time_steps):
                clip_2[i] = cv2.resize(data_np_2[i], (clip_2.shape[2], clip_2.shape[1]))
        UD_transpose = (
            UD_transpose_dict[self.UD_shape[0]], UD_transpose_dict[self.UD_shape[1]],
            UD_transpose_dict[self.UD_shape[2]],
            UD_transpose_dict[self.UD_shape[3]])
        clip_1 = torch.from_numpy(np.transpose(clip_1, UD_transpose)).float()
        clip_2 = torch.from_numpy(np.transpose(clip_2, UD_transpose)).float()

        if self.isE:
            E_1, E_2 = random.choices(E_pool, k=2)
            E_patient_1, E_LN_1, E_npy_1, E_img_dir_1 = E_1
            E_patient_2, E_LN_2, E_npy_2, E_img_dir_2 = E_2
            if E_img_dir_1 is not None:
                E_image_1 = cv2.resize(cv2.imread(E_img_dir_1) / 255., (clip_1.shape[2], clip_1.shape[1]))
                if self.augmentation is not None:
                    clip_1, E_image_1 = self.augmentation([clip_1, E_image_1])
            else:
                E_image_1 = np.zeros(clip_1.shape[1:])
                if self.augmentation is not None:
                    clip_1 = self.augmentation([clip_1])[0]

            if E_img_dir_2 is not None:
                E_image_2 = cv2.resize(cv2.imread(E_img_dir_2) / 255., (clip_2.shape[2], clip_2.shape[1]))
                if self.augmentation is not None:
                    clip_2, E_image_2 = self.augmentation([clip_2, E_image_2])
            else:
                E_image_2 = np.zeros(clip_2.shape[1:])
                if self.augmentation is not None:
                    clip_2 = self.augmentation([clip_2])[0]
            E_transpose = (E_transpose_dict[self.E_shape[0]], E_transpose_dict[self.E_shape[1]],
                           E_transpose_dict[self.E_shape[2]])

            E_image_1 = torch.from_numpy(np.transpose(E_image_1, E_transpose)).float()
            E_image_2 = torch.from_numpy(np.transpose(E_image_2, E_transpose)).float()

        UD_transpose = (
            UD_transpose_dict[self.UD_shape[0]], UD_transpose_dict[self.UD_shape[1]],
            UD_transpose_dict[self.UD_shape[2]],
            UD_transpose_dict[self.UD_shape[3]])

        if video_type_1 == 'grayscale':
            graph_1 = torch.from_numpy(np.array([float(percent_1), 1 - float(percent_1)])).float()
        else:
            graph_1 = torch.from_numpy(np.array([1 - float(percent_1), float(percent_1)])).float()

        if video_type_2 == 'grayscale':
            graph_2 = torch.from_numpy(np.array([float(percent_2), 1 - float(percent_2)])).float()
        else:
            graph_2 = torch.from_numpy(np.array([1 - float(percent_2), float(percent_2)])).float()

        if self.isE:
            return (clip_1, E_image_1, graph_1), (clip_2, E_image_2, graph_2)
        else:
            return clip_1, graph_1, clip_2, graph_2

    def __len__(self):
        return len(self.all_LNs)


class DatasetMix(Dataset):
    def __init__(self, video_dir, U_img_dir, D_img_dir, E_img_dir, img_depth, exc_dir, sample_rate=0.25,
                 max_img_pre_LN=20,
                 number_classes=1, split='train', resize=(224, 224), augmentation=None, reconstruct=False,
                 out_shape='CHWT'):
        self.csv = pd.read_excel(exc_dir)
        self.video_dir = video_dir
        self.U_img_dir = U_img_dir
        self.D_img_dir = D_img_dir
        self.E_img_dir = E_img_dir
        self.img_depth = img_depth
        self.sample_rate = sample_rate
        self.max_img_pre_LN = max_img_pre_LN
        self.number_classes = number_classes
        self.augmentation = augmentation
        self.split = split
        self.resize = resize
        self.reconstruct = reconstruct
        self.tumor_times = self.csv[self.csv['dataset'].isin(split)]
        self.filename = []
        self.all_video = []
        self.all_LN = []
        self.U_dict = {}
        self.D_dict = {}
        self.E_dict = {}
        self.out_shape = out_shape
        B_lesion_count = 0
        M_lesion_count = 0
        B_img_count = 0
        M_img_count = 0
        for root, dirs, files in os.walk(video_dir):  # all video name in data set
            for file in files:
                if '.mp4' in file:
                    self.all_video.append(file)

        if self.reconstruct:
            UD_2Dimg_bulid(self.video_dir, self.U_img_dir, self.D_img_dir, self.tumor_times, self.sample_rate,
                           self.max_img_pre_LN)

        for (i, LN) in self.tumor_times.iterrows():
            LN_folder = str(LN[7])
            LN_video_name = str(LN[8])
            LN_num = str(LN[2])
            LN_class = str(LN[5])
            self.all_LN.append(LN_folder + '_' + LN_num)
            self.U_dict[LN_folder + '_' + LN_num] = []
            self.D_dict[LN_folder + '_' + LN_num] = []
            self.E_dict[LN_folder + '_' + LN_num] = []

            U_LN_path = os.path.join(U_img_dir, LN_folder + '_' + LN_num)
            D_LN_path = os.path.join(D_img_dir, LN_folder + '_' + LN_num)
            E_LN_path = os.path.join(E_img_dir, LN_folder, LN_video_name)
            if LN_class == 'B' or LN_class == 'Benign' or LN_class == 'benign':
                cls = 0
                B_lesion_count += 1
            else:
                cls = 1
                M_lesion_count += 1

            for img in os.listdir(U_LN_path):
                self.U_dict[LN_folder + '_' + LN_num].append((os.path.join(U_LN_path, img), cls, 'grayscale'))
                if cls == 0:
                    B_img_count += 1
                else:
                    M_img_count += 1

            for img in os.listdir(D_LN_path):
                self.D_dict[LN_folder + '_' + LN_num].append((os.path.join(D_LN_path, img), cls, 'doppler'))
                if cls == 0:
                    B_img_count += 1
                else:
                    M_img_count += 1

            for img in os.listdir(E_LN_path):
                self.E_dict[LN_folder + '_' + LN_num].append((os.path.join(E_LN_path, img), cls, 'elastography'))
                if cls == 0:
                    B_img_count += 1
                else:
                    M_img_count += 1

        print(self.split, 'data amount:', B_img_count + M_img_count)
        print('-Benign lesion:', B_lesion_count)
        print('-Malignant lesion:', M_lesion_count)
        print('-Benign pair:', B_img_count)
        print('-Malignant pair:', M_img_count)

    def __getitem__(self, index):
        LN_name = self.all_LN[floor(index / 10)]
        all_list = random.choices(self.U_dict[LN_name] + self.D_dict[LN_name] + self.E_dict[LN_name], k=12)
        _, cls, _ = all_list[0]

        img_3D = np.zeros((self.img_depth, self.resize[0], self.resize[1], 3))
        mode_list = np.zeros(self.img_depth)

        transpose_dict = {'C': 3, 'H': 1, 'W': 2, 'D': 0}
        mode_dict = {'grayscale': 0, 'doppler': 1, 'elastography': 2}

        for i in range(self.img_depth):
            if i < len(all_list):
                img_path = all_list[i][0]
                mode_list[i] = mode_dict[all_list[i][2]]
                img_3D[i] = cv2.imread(img_path) / 255.

        if self.augmentation is not None:
            img_3D = self.augmentation(img_3D)

        label = torch.tensor(cls).float()
        UD_transpose = (
            transpose_dict[self.out_shape[0]], transpose_dict[self.out_shape[1]], transpose_dict[self.out_shape[2]],
            transpose_dict[self.out_shape[3]])
        img_3D = torch.from_numpy(np.transpose(img_3D, UD_transpose)).float()

        return img_3D, label, LN_name, mode_list

    def __len__(self):
        return len(self.all_LN) * 10
