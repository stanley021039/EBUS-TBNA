import cv2
import time
import random
import numpy as np
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from math import cos, pi
import os
from skimage.metrics import structural_similarity as ssim
from math import floor
import pandas as pd
import torch


def plot_confusion_matrix_and_scores(confusion_mat, savedir=''):
    plt.figure(figsize=(4, 4))
    sns.heatmap(confusion_mat, annot=True, cbar=False, fmt='g', cmap='viridis')
    plt.title('Confusion matrix', fontsize=15)
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    TN, FP, FN, TP = confusion_mat[0, 0], confusion_mat[0, 1], confusion_mat[1, 0], confusion_mat[1, 1]
    print("Accuracy   : %.2f%%" % ((TN+TP) / (TN+FN+FP+TP) * 100))
    print("Sensitivity: %.2f%%" % (TP / (FN + TP) * 100))
    print("Specificity: %.2f%%" % (TN / (FP + TN) * 100))
    if savedir:
        plt.savefig(savedir)
    else:
        plt.show()
    plt.close()

def Roc_curve(y_true, y_pred, savedir='', show=False, re_auc=False):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    y_pred = y_pred >= optimal_th
    optimAcc = np.sum(y_pred == y_true) / np.size(y_true)
    print('Optimal_Threshold:', optimal_th)
    print('Optimal_Accuracy:', optimAcc)
    optimal_hist = np.bincount(2 * y_true.astype(int) + y_pred, minlength=2 ** 2,).reshape(2, 2)
    roc_auc = auc(fpr, tpr)
    print('AUC: %0.4f' % roc_auc)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    if savedir:
        plt.savefig(savedir)
    if show:
        plt.show()
    plt.close()
    if re_auc:
        return optimal_th, optimal_hist, roc_auc
    else:
        return optimal_th, optimal_hist

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def plot_fig(*args, legend=None, savedir='', title=''):
    plt.figure('Draw')
    for array in args:
        plt.plot(np.arange(1, len(array)+1), array)
    if legend:
        plt.legend(legend)
    if title:
        plt.title(title)
    if savedir:
        plt.savefig(savedir)
    else:
        plt.show()
    plt.close()

def sen_spe(hist):
    sensitivity = hist[1, 1] / (hist[1, 1] + hist[1, 0])
    specificity = hist[0, 0] / (hist[0, 0] + hist[0, 1])
    return sensitivity, specificity

def cosine_decay(ini_lr, global_step, decay_steps, optim, alpha=0.0):
    global_step = min(global_step, decay_steps)
    cos_decay = 0.5 * (1 + cos(pi * global_step / decay_steps))
    decayed_coff = (1 - alpha) * cos_decay + alpha
    decayed_learning_rate = ini_lr * decayed_coff
    for param_group in optim.param_groups:
        param_group['lr'] = decayed_learning_rate

def model_paras_count(model):
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0
    for param in model.parameters():
        mulValue = np.prod(param.size())
        Total_params += mulValue
        if param.requires_grad:
            Trainable_params += mulValue
        else:
            NonTrainable_params += mulValue

    return Total_params, Trainable_params, NonTrainable_params


def PCA_reduce(X, n_components=2):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    X_pca = pca.transform(X)
    return X_pca


def E_build(exc, E_savedir, vid_dir, resize=None):
    LN_exc = exc
    for (i, LN) in LN_exc.iterrows():  # construct input list
        patient_folder = str(LN[7])
        LN_video = str(LN[8])
        E_patient_dir = os.path.join(E_savedir, patient_folder)
        E_LN_dir = os.path.join(E_patient_dir, LN_video)
        print(E_LN_dir)
        if not os.path.exists(E_patient_dir):
            os.mkdir(E_patient_dir)
        if not os.path.exists(E_LN_dir):
            os.mkdir(E_LN_dir)
        else:
            print(E_LN_dir, 'done')
            continue
        count = 0
        compare_frame = np.zeros([10, 10, 3])
        video_path = os.path.join(vid_dir, patient_folder, LN_video)
        if os.path.isfile(video_path):
            for node_col in range(10, len(LN), 2):  # check video graph of nodes
                if LN[node_col] == 'elastography':
                    t = LN[node_col - 1]
                    end_t = LN[node_col + 1]
                    video_cap = cv2.VideoCapture(video_path)
                    video_cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                    count_LN = 0
                    while t < end_t:
                        # video_cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                        ret, frame = video_cap.read()
                        if not ret:
                            break
                        t = round(video_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 2)
                        elastic_frame = frame[160:695, 1010:1610, :]
                        abs_channel = np.max(elastic_frame, axis=2) - np.min(elastic_frame, axis=2)
                        color_pixel = np.sum(abs_channel > 30)

                        if color_pixel >= 100000:
                            compare_frame_ = cv2.resize(elastic_frame, (10, 10))
                            score = ssim(compare_frame, compare_frame_, multichannel=True)
                            if score <= 0.7:
                                print(t, color_pixel)
                                if resize is not None:
                                    cv2.imwrite(os.path.join(E_LN_dir, str(t) + '_' + str(color_pixel) + '.jpg'), cv2.resize(elastic_frame, resize))
                                else:
                                    cv2.imwrite(os.path.join(E_LN_dir, str(t) + '_' + str(color_pixel) + '.jpg'), elastic_frame)
                                count += 1
                                count_LN += 1
                                compare_frame = compare_frame_
                        # t = round(t + 0.25, 2)
                elif pd.isna(LN[node_col]):
                    break
        else:
            print(video_path, "not exist")


def two_stream_dataset_builder(exc, vid_dir, clip_savedir, E_savedir, cropping_size, time_steps, sample_rate, size='r1', resize=None):
    x1, x2, y1, y2 = cropping_size
    for (i, LN) in exc.iterrows():
        patient_folder = str(LN[7])
        LN_video = str(LN[8])
        UD_patient_dir = os.path.join(clip_savedir, patient_folder)
        UD_LN_dir = os.path.join(UD_patient_dir, LN_video)
        E_patient_dir = os.path.join(E_savedir, patient_folder)
        E_LN_dir = os.path.join(E_patient_dir, LN_video)
        print(LN_video)

        if not os.path.exists(UD_patient_dir):
            os.mkdir(UD_patient_dir)
            print('build folder:', UD_patient_dir)
        if not os.path.exists(UD_LN_dir):
            os.mkdir(UD_LN_dir)
            print('build folder:', UD_LN_dir)
        if not os.path.exists(E_patient_dir):
            os.mkdir(E_patient_dir)
            print('build folder:', E_patient_dir)
        if not os.path.exists(E_LN_dir):
            os.mkdir(E_LN_dir)
            print('build folder:', E_LN_dir)
        '''else:
            print(UD_LN_dir, 'done')
            continue'''

        LN_video_path = os.path.join(vid_dir, patient_folder, LN_video)
        if os.path.isfile(LN_video_path):
            video_cap = cv2.VideoCapture(LN_video_path)
            if resize is not None:
                clip = np.zeros((time_steps, resize[0], resize[1], 3), dtype='uint8')  # T, H, W
            else:
                clip = np.zeros((time_steps, y2 - y1, x2 - x1, 3), dtype='uint8')  # T, H, W
            graph = np.zeros(time_steps)
            compare_frame = np.zeros([10, 10, 3])
            clip_count = 0
            Eimg_count = 0
            need_new_frames = time_steps  # first frame don't need overlap
            new_frames_count = 0
            vid_time = 0
            while True:  # construct data in a graph during
                video_cap.set(cv2.CAP_PROP_POS_MSEC, vid_time * 1000)
                ret, frame = video_cap.read()
                if not ret:
                    break
                c_1 = frame[150:290, 455:520]
                c_2 = frame[150:290, 300:365]
                graph_ = graph_classify(c_1, c_2)
                if graph_ != 'elastography':
                    if resize is not None:
                        crop_frame = cv2.resize(frame[y1:y2, x1:x2, :], (clip.shape[2], clip.shape[1]))
                    else:
                        crop_frame = frame[y1:y2, x1:x2, :]
                    clip = np.concatenate((clip[1:], np.expand_dims(crop_frame, axis=0)), axis=0)
                    graph = np.concatenate((graph[1:], np.array([0]) if graph_ == 'grayscale' else np.array([1])))
                    new_frames_count += 1
                    if new_frames_count == need_new_frames:
                        clip.astype('uint8')
                        np.save(os.path.join(UD_LN_dir, 'clip_' + str(clip_count) + '.npy'), clip)
                        np.save(os.path.join(UD_LN_dir, 'graph_' + str(clip_count) + '.npy'), graph)
                        need_new_frames = time_steps // 2  # 50% overlap
                        clip_count += 1
                        new_frames_count = 0
                else:
                    if new_frames_count != 0:  # save last clips
                        clip.astype('uint8')
                        np.save(os.path.join(UD_LN_dir, 'clip_' + str(clip_count) + '.npy'), clip)
                        np.save(os.path.join(UD_LN_dir, 'graph_' + str(clip_count) + '.npy'), graph)
                        need_new_frames = time_steps
                        clip_count += 1
                        new_frames_count = 0

                    crop_frame = frame[160:695, 1010:1610, :]
                    abs_channel = np.max(crop_frame, axis=2) - np.min(crop_frame, axis=2)
                    color_pixel = np.sum(abs_channel > 30)

                    if color_pixel >= 100000:
                        compare_frame_ = cv2.resize(crop_frame, (10, 10))
                        if ssim(compare_frame, compare_frame_, multichannel=True) <= 0.7:
                            if resize is not None:
                                cv2.imwrite(os.path.join(E_LN_dir, 'Eimg_' + str(Eimg_count) + '.jpg'),
                                            cv2.resize(crop_frame, resize))
                            else:
                                cv2.imwrite(os.path.join(E_LN_dir, 'Eimg_' + str(Eimg_count) + '.jpg'), crop_frame)
                            Eimg_count += 1
                            compare_frame = compare_frame_

                vid_time = round(vid_time + sample_rate, 2)
                video_cap.set(cv2.CAP_PROP_POS_MSEC, vid_time * 1000)

            video_cap.release()

        else:  # target video is not exist
            print(LN_video_path, "not exist")



def UD_npy_bulid(exc, UD_savedir, vid_dir, cropping_size, time_steps, sample_rate, size='r1', resize=None):
    slice_length = time_steps * sample_rate

    BL_count = 0
    ML_count = 0
    for (i, LN) in exc.iterrows():
        patient_folder = str(LN[7])
        LN_video = str(LN[8])
        patient_dir = os.path.join(UD_savedir, patient_folder)
        UD_LN_dir = os.path.join(patient_dir, LN_video)

        print(UD_LN_dir)
        if not os.path.exists(patient_dir):
            os.mkdir(patient_dir)
        if not os.path.exists(UD_LN_dir):
            os.mkdir(UD_LN_dir)
        else:
            print(UD_LN_dir, 'done')
            continue

        video_path = os.path.join(vid_dir, patient_folder, LN_video)
        if os.path.isfile(video_path):
            unuse_nodes = [8]
            for node_col in range(10, len(LN), 2):  # check video graph of nodes
                if not (LN[node_col] == 'grayscale' or LN[node_col] == 'doppler'):
                    unuse_nodes.append(node_col)
                    if pd.isna(LN[node_col]):
                        break

            video_cap = cv2.VideoCapture(video_path)

            for j in range(len(unuse_nodes) - 1):  # construct data of a lesion
                if LN[unuse_nodes[j + 1] - 1] - LN[unuse_nodes[j] + 1] >= slice_length:
                    start_time = LN[unuse_nodes[j] + 1]
                    end_time = start_time + slice_length
                    graph_col = unuse_nodes[j] + 2

                    while True:  # construct data in a graph during
                        vid_time = start_time
                        video_cap.set(cv2.CAP_PROP_POS_MSEC, vid_time * 1000)

                        x1, x2, y1, y2 = cropping_size
                        if not resize:
                            video = np.zeros((time_steps, y2 - y1, x2 - x1, 3))  # t, h, w, c
                        else:
                            video = np.zeros((time_steps, resize[0], resize[1], 3))  # t, h, w, c

                        for k in range(time_steps):
                            ret, frame = video_cap.read()
                            if not ret:
                                video[k, :, :, :] = video[k - 1, :, :, :]  # res1
                                # print(video_path, "Error")
                                # print(self.filename[index])
                            else:
                                if not resize:
                                    video[k, :, :, :] = frame[y1:y2, x1:x2, :]  # res1
                                else:
                                    video[k, :, :, :] = cv2.resize(frame[y1:y2, x1:x2, :],
                                                                   (video.shape[2], video.shape[1]))  # res1

                            vid_time = round(vid_time + sample_rate, 2)
                            video_cap.set(cv2.CAP_PROP_POS_MSEC, vid_time * 1000)
                        video = video.astype('uint8')
                        print(round(start_time, 2))
                        np.save(os.path.join(UD_LN_dir,
                                             str(round(start_time, 2)) + '_' + str(LN[graph_col]) + '_' + str(
                                                 round(min(1, (LN[graph_col + 1] - start_time) / slice_length),
                                                       2)) + '.npy'), video)
                        if start_time + slice_length / 2 <= LN[
                            unuse_nodes[j + 1] - 1] - slice_length:  # not the end of a graph
                            start_time += slice_length / 2
                        elif end_time == LN[unuse_nodes[j + 1] - 1]:  # the end of a lesion
                            break
                        else:  # the end of a graph
                            start_time = LN[unuse_nodes[j + 1] - 1] - slice_length
                        end_time = start_time + slice_length

                        if start_time >= LN[graph_col + 1]:  # next graph
                            graph_col += 2
            video_cap.release()

        else:  # target video is not exist
            print(video_path, "not exist")


def dataset_split_exc(exc_path, exc_savedir, v_amount=50):
    exc = pd.read_excel(exc_path)
    candidate_LN = []
    for row, video in exc.iterrows():
        if video['dataset'] != 'x' and video['dataset'] != 'test':
            candidate_LN.append(row)
    valid_LN = np.random.choice(candidate_LN, v_amount, False)

    for i in candidate_LN:
        if i in valid_LN:
            exc['dataset'][i] = 'valid'
        else:
            exc['dataset'][i] = 'train'
    exc.to_excel(exc_savedir, index=False, header=False)


def UD_2Dimg_bulid(vid_dir, U_img_dir, D_img_dir, LN_exc, sample_rate, max_img_pre_LN, resize=(224, 224)):
    for (i, LN) in LN_exc.iterrows():  # construct input list
        LN_folder = str(LN[7])
        LN_video_name = str(LN[8])
        LN_num = str(LN[2])
        LN_class = str(LN[5])
        print(LN_video_name)

        U_img_folder = os.path.join(U_img_dir, LN_folder + '_' + LN_num)
        D_img_folder = os.path.join(D_img_dir, LN_folder + '_' + LN_num)
        if not os.path.isdir(U_img_folder):
            os.mkdir(U_img_folder)
        if not os.path.isdir(D_img_folder):
            os.mkdir(D_img_folder)

        video_path = os.path.join(vid_dir, LN_folder, LN_video_name)
        video_cap = cv2.VideoCapture(video_path)

        U_during = 0
        D_during = 0
        for node_col in range(10, len(LN), 2):  # sum of each mode
            mode = LN[node_col]
            if mode == 'grayscale':
                t = LN[node_col - 1]
                end_t = LN[node_col + 1]
                U_during += end_t - t

            elif mode == 'doppler':
                t = LN[node_col - 1]
                end_t = LN[node_col + 1]
                D_during += end_t - t


        for node_col in range(10, len(LN), 2):  # check video graph of nodes
            mode = LN[node_col]
            print(mode)
            if mode == 'grayscale':
                t = LN[node_col - 1]
                end_t = LN[node_col + 1]
                t_during = U_during
                if t_during / sample_rate > max_img_pre_LN:
                    t_gap = floor(t_during / max_img_pre_LN * 1000) / 1000
                else:
                    t_gap = sample_rate
                while t < end_t:
                    video_cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                    ret, frame = video_cap.read()
                    img = cv2.resize(frame[160:736, 748:1452, :], resize)
                    cv2.imwrite(os.path.join(U_img_folder, str(round(t, 2)) + '.jpg'), img)
                    print(str(round(t, 2)) + '.jpg')
                    t = round(t + t_gap, 2)

            elif mode == 'doppler':
                t = LN[node_col - 1]
                end_t = LN[node_col + 1]
                t_during = D_during
                if t_during / sample_rate > max_img_pre_LN:
                    t_gap = floor(t_during / max_img_pre_LN * 1000) / 1000
                else:
                    t_gap = sample_rate
                while t < end_t:
                    video_cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                    ret, frame = video_cap.read()
                    img = cv2.resize(frame[160:736, 748:1452, :], resize)
                    cv2.imwrite(os.path.join(D_img_folder, str(round(t, 2)) + '.jpg'), img)
                    print(str(round(t, 2)) + '.jpg')
                    t = round(t + t_gap, 2)


def E_rename(E_dir):
    for LN_folder in os.listdir(E_dir):
        print(LN_folder)
        for LN_video in os.listdir(os.path.join(E_dir, LN_folder)):
            for img_name in os.listdir(os.path.join(E_dir, LN_folder, LN_video)):
                old_img_path = os.path.join(E_dir, LN_folder, LN_video, img_name)
                print(old_img_path)
                t = img_name.split('.')[0] + '.' + img_name.split('.')[1]
                img = cv2.imread(old_img_path)
                abs_channel = np.max(img, axis=2) - np.min(img, axis=2)
                color_pixel = np.sum(abs_channel > 30)
                new_img_path = os.path.join(E_dir, LN_folder, LN_video, str(t) + '_' + str(color_pixel) + '.jpg')
                print(new_img_path)
                os.rename(old_img_path, new_img_path)


def label_smoothing(labels, factor=0.1):
    if len(labels.shape) >= 1:
        num_labels = labels.shape[0]
        labels = ((1-factor) * labels) + (factor / num_labels)
        return labels
    else:
        return labels * (1-factor) + (1-labels) * factor


def train_valid_split(LN_exc, exc_save_dir, valid_proportion=0.2):
    patient_folder_tmp = ''
    dataset_tmp = 'train'
    for row, LN in LN_exc.iterrows():
        patient_folder = str(LN[7])
        LN_num = str(LN[2])
        if LN['dataset'] == 'train' or LN['dataset'] == 'valid':
            if patient_folder != patient_folder_tmp:
                patient_folder_tmp = patient_folder
                if random.random() < valid_proportion:
                    dataset_tmp = 'valid'
                else:
                    dataset_tmp = 'train'

            print(patient_folder, LN_num, dataset_tmp)
            LN_exc['dataset'][row] = dataset_tmp

    exe_data = pd.DataFrame(LN_exc)
    exe_data.to_excel(exc_save_dir, index=False)

    df = pd.read_excel(exc_save_dir)
    count_ = np.zeros((2, 2))
    for row, video in df.iterrows():
        if video['dataset'] == 'valid':
            if video[5] == 'B' or video[5] == 'benign':
                count_[0, 0] += 1
            else:
                count_[0, 1] += 1
        if video['dataset'] == 'train':
            if video[5] == 'B' or video[5] == 'benign':
                count_[1, 0] += 1
            else:
                count_[1, 1] += 1

    print(count_)
    print(np.sum(count_, axis=1))


def raw2LN(raw_video_dir, LN_video_dir, excel_name='EBUS_data_0524.xlsx', frame_rate=20):
    LN_exc = pd.read_excel(excel_name, sheet_name='labeled_detail')  # 病灶excel檔
    for (i, LN) in LN_exc.iterrows():  # 一行一個病灶
        patient_folder = LN[7]  # 影片所在資料夾
        LN_num = LN[2]  # 相同病人的第N個病灶

        if not os.path.isdir(LN_video_dir + str(patient_folder)):  # 建立片段資料夾
            os.mkdir(LN_video_dir + str(patient_folder))

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 建立writer
        out = cv2.VideoWriter(os.path.join(LN_video_dir, str(patient_folder), str(patient_folder) + '_' + str(LN_num) + '.mp4'), fourcc, frame_rate, (1920, 1200))  # 定義片段路徑, 格式, writer, 影像大小
        for vid_col in [8, 12, 16]:  # 影片1、影片2、影片3
            vid_name = LN[vid_col]
            if pd.isna(vid_name):  # 沒有影片
                break
            else:
                vid_path = os.path.join(raw_video_dir, patient_folder, vid_name)  # 影片位址
                vid_cap = cv2.VideoCapture(vid_path)  # 建立capture來讀取影片
                # 獲取病灶開始與結束時間
                start_time = LN[vid_col+1]
                end_time = LN[vid_col+3]
                _time = start_time
                t_gap = 0.05  # 取樣步長

                vid_cap.set(cv2.CAP_PROP_POS_MSEC, _time * 1000)  # 設定影片開始時間
                while time <= end_time:  # 截圖並寫道writer中
                    ret, frame = vid_cap.read()
                    out.write(frame)
                    _time = round(_time + t_gap, 2)
                    vid_cap.set(cv2.CAP_PROP_POS_MSEC, _time * 1000)

                vid_cap.release()
        out.release()


def graph_classify(c_img1, c_img2):
    p_1 = np.mean(c_img1[:, :, 2]) - np.mean(c_img1[:, :, 0])
    p_2 = np.mean(np.max(c_img2, axis=2) - np.min(c_img2, axis=2))

    if p_2 > 5:
        return 'elastography'
    elif p_1 > 5:
        return 'doppler'
    else:
        return 'grayscale'


def graph_auto_label(exc_dir, data_dir, save_exc_dir, sheet=''):
    LNs_graph_list = []
    # 讀取excel
    if sheet:
        LN_exc = pd.read_excel(exc_dir, sheet_name=sheet)
    else:
        LN_exc = pd.read_excel(exc_dir)

    for (i, LN) in LN_exc.iterrows():  # 逐行讀取excel
        graph_list = []
        for col in range(7):
            graph_list.append(LN[col])
        folder = str(LN[7])
        LN_num = LN(2)

        # 讀取影片
        vid_path = os.path.join(data_dir, folder, folder + '_' + str(LN_num) + '.mp4')
        graph_list.append(vid_path)
        video_cap = cv2.VideoCapture(vid_path)
        graph = ''
        t = -1

        # 分析每幀的模式
        while True:
            ret, frame = video_cap.read()  # 讀取下一幀
            t_ = round(video_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 2)  # 獲取當前幀的時間點
            if not ret or t_ < t:
                graph_list.append(t)  # 紀錄結束時間
                break
            t = t_

            c_1 = frame[150:290, 455:520]
            c_2 = frame[150:290, 300:365]
            graph_ = graph_classify(c_1, c_2)
            # 比較當前幀與前一幀模式是否一樣
            if graph_ != graph:
                print('    ', t, graph_)
                graph = graph_
                graph_list.append(t)
                graph_list.append(graph)

        LNs_graph_list.append(graph_list)
    data_xlsx = pd.DataFrame(LNs_graph_list)
    data_xlsx.to_excel(save_exc_dir, index=False, header=False)
