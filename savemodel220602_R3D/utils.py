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
import pandas as pd
from skimage.metrics import structural_similarity as ssim


def plot_confusion_matrix_and_scores(confusion_mat, savedir=''):
    plt.figure(figsize=(4, 4))
    sns.heatmap(confusion_mat, annot=True, cbar=False, fmt='g', cmap='viridis')
    plt.title('Confusion matrix', fontsize=15)
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    TN, FN, FP, TP = confusion_mat[0, 0], confusion_mat[0, 1], confusion_mat[1, 0], confusion_mat[1, 1]
    print("Accuracy : %.2f%%" % ((TN+TP) / (TN+FN+FP+TP) * 100))
    print("F1_score : %.2f%%" % (TP / (TP + (FN + FP) / 2) * 100))
    print("Precision: %.2f%%" % (TP / (TP + FP) * 100))
    print("Recall   : %.2f%%" % (TP / (TP + FN) * 100))
    if savedir:
        plt.savefig(savedir)
    else:
        plt.show()
    plt.close()

def Roc_curve(y_true, y_pred, savedir='', re_auc=False):
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

def plot_fig(array, savedir='', title=''):
    plt.figure('Draw')
    plt.plot(np.arange(1, len(array)+1), array)
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


def graph_classify(c_img1, c_img2):
    p_1 = np.mean(c_img1[:, :, 2]) - np.mean(c_img1[:, :, 0])
    p_2 = np.mean(np.max(c_img2, axis=2) - np.min(c_img2, axis=2))

    if p_2 > 5:
        return 'elastography'
    elif p_1 > 5:
        return 'doppler'
    else:
        return 'grayscale'

def graph_auto_label(data_dir, save_exc_dir):
    LNs_graph_list = []
    LN_num = 0
    _folder = ''

    LN_exc = pd.read_excel('EBUS_data_0524.xlsx', sheet_name='labeled_detail')  # 病灶excel檔
    for (i, LN) in LN_exc.iterrows():  # 一行一個病灶
        graph_list = []
        for col in range(8):
            graph_list.append(LN[col])
        folder = LN[7]

        if _folder != folder:
            LN_num = 0
            _folder = folder
        LN_num += 1

        graph_list.append(folder + '_' + str(LN_num))

        vid_path = os.path.join(data_dir, folder, folder + '_' + str(LN_num) + '.mp4')
        print(vid_path)
        video_cap = cv2.VideoCapture(vid_path)
        graph = ''
        t = -1
        while True:
            ret, frame = video_cap.read()
            t_ = round(video_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 2)
            if not ret or t_ < t:  # end
                graph_list.append(t)
                break
            t = t_

            c_1 = frame[150:290, 455:520]
            c_2 = frame[150:290, 300:365]
            graph_ = graph_classify(c_1, c_2)
            if graph_ != graph:
                print('    ', t, graph_)
                graph = graph_
                graph_list.append(t)
                graph_list.append(graph)

        LNs_graph_list.append(graph_list)
        i += 1
    data_xlsx = pd.DataFrame(LNs_graph_list)
    data_xlsx.to_excel(save_exc_dir, index=False, header=False)


def PCA_reduce(X, n_components=2):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    X_pca = pca.transform(X)
    return X_pca


def E_build(exc, E_savedir, vid_dir):
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
                                cv2.imwrite(os.path.join(E_LN_dir, str(t) + '.jpg'), elastic_frame)
                                count += 1
                                count_LN += 1
                                compare_frame = compare_frame_
                        # t = round(t + 0.25, 2)
                elif pd.isna(LN[node_col]):
                    break
        else:
            print(video_path, "not exist")

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

                        x1, x2, y1, y2 = cropping_size[size]
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
