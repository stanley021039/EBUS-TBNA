import cv2
import sys
import time
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torchvision.models as model
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from augmentation import get_composed_augmentations
import json
from models.R3D_Modify import R3D_UDE
from torch.nn.modules.loss import _Loss
from matplotlib import pyplot as plt
from dataset0530 import Dataset3D
from torch.utils.data import DataLoader
import seaborn as sns
from math import cos, pi
from utils import plot_confusion_matrix_and_scores, Roc_curve, plot_fig, cosine_decay

save_dir = 'savemodel220609_R3D_TS'

# random seed setting
torch.manual_seed(42)

# check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# torch.cuda.set_device(1)

def train(epoch, dataloader, Net, optimizer, loss_fn):
    Net.train()
    loss = 0
    count = 0
    correct = 0
    hist = np.zeros((2, 2))
    for i, (data, graph, target, e_img, name, stime) in enumerate(tqdm(dataloader, ncols=80)):
        data, target, graph, e_img = data.to(device), target.to(device), graph.to(device), e_img.to(device)
        data = data.permute(0, 4, 2, 3, 1).contiguous()
        e_img = e_img.permute(0, 3, 1, 2).contiguous()
        optimizer.zero_grad()
        pred = Net(data, e_img, graph)
        losses = loss_fn(torch.squeeze(pred, dim=1), target)
        losses.backward()
        optimizer.step()
        pred_ge = torch.ge(pred, threshold).cpu().numpy()
        target = target.cpu().numpy()
        for batch in range(data.shape[0]):
            count += 1
            correct += pred_ge[batch] == target[batch]
            hist[int(target[batch]), int(pred_ge[batch])] += 1
        loss += np.sum(losses.item()) * data.shape[0]

    print('Epoch %3d training: \n loss: %f, accuracy: %f' % (epoch, loss / count, correct / count))
    plot_confusion_matrix_and_scores(hist, savedir=save_dir + '/confusion_matrix/train_cf' + str(epoch) + '.jpg')
    return loss / count, correct / count


# valid
def valid(epoch, dataloader, Net, loss_fn):
    Net.eval()
    loss = 0
    hist_video = np.zeros((2, 2))
    hist_lesion = np.zeros((2, 2))
    lesion_name = ''
    pred_lesion = 0
    count_video = 0
    y_true_video = []
    y_pred_video = []
    y_true_lesion = []
    y_pred_lesion = []
    with torch.no_grad():
        for i, (data, graph, target, e_img, name, stime) in enumerate(tqdm(dataloader, ncols=80)):
            data, target, graph, e_img = data.to(device), target.to(device), graph.to(device), e_img.to(device)
            data = data.permute(0, 4, 2, 3, 1).contiguous()
            e_img = e_img.permute(0, 3, 1, 2).contiguous()
            pred = Net(data, e_img, graph)
            losses = loss_fn(torch.squeeze(pred, dim=1), target)
            pred_ge = torch.ge(pred, threshold).cpu().numpy()
            target = target.cpu().numpy()
            pred = pred.cpu().numpy()
            for batch in range(data.shape[0]):
                if not lesion_name:  # first lesion
                    tar_lesion = target[batch]
                    lesion_name = name[batch]
                    print(lesion_name)
                if lesion_name != name[batch]:
                    pred_lesion_avg = pred_lesion / count_video
                    print('lesion label pred: %d(%.2f), true: %d' % (
                        int(pred_lesion_avg > threshold), pred_lesion_avg, tar_lesion))
                    hist_lesion[int(tar_lesion), int(pred_lesion_avg > threshold)] += 1
                    y_true_lesion.append(tar_lesion)
                    y_pred_lesion.append(pred_lesion_avg)
                    pred_lesion = 0
                    count_video = 0
                    tar_lesion = target[batch]
                    lesion_name = name[batch]
                    print(lesion_name)

                y_true_video.append(target[batch].item())
                y_pred_video.append(pred[batch].item())
                count_video += 1
                pred_lesion += pred[batch].item()
                hist_video[int(target[batch]), int(pred_ge[batch])] += 1
                print('%s(s) %.4f  %d' % (stime[batch], pred[batch].item(), target[batch]))
            loss += np.sum(losses.item()) * data.shape[0]

    # last lesion
    pred_lesion_avg = pred_lesion / count_video
    print('lesion label pred: %d(%.2f), true: %d' % (int(pred_lesion_avg > threshold), pred_lesion_avg, tar_lesion))
    hist_lesion[int(tar_lesion), int(pred_lesion_avg > threshold)] += 1
    y_true_lesion.append(tar_lesion)
    y_pred_lesion.append(pred_lesion_avg)

    total_v = np.sum(hist_video)
    total_l = np.sum(hist_lesion)
    Accuracy_v = np.sum(np.diag(hist_video)) / total_v
    Accuracy_l = np.sum(np.diag(hist_lesion)) / total_l

    print('Epoch %3d validation : \n loss: %f, Accuracy(video): %f, Accuracy(lesion): %f' % (
        epoch, loss / total_v, Accuracy_v, Accuracy_l))
    print('Per video')
    plot_confusion_matrix_and_scores(hist_video, savedir=save_dir + '/confusion_matrix/valid_v_' + str(epoch) + '.jpg')
    optim_threshold_v, optim_cf_v, auc_v = Roc_curve(np.array(y_true_video), np.array(y_pred_video),
                                                     savedir=save_dir + '/Roc_curve/valid_v_' + str(epoch) + '.jpg',
                                                     re_auc=True)
    print('Per lesion')
    plot_confusion_matrix_and_scores(hist_lesion, savedir=save_dir + '/confusion_matrix/valid_l_' + str(epoch) + '.jpg')
    optim_threshold_l, optim_cf_l, auc_l = Roc_curve(np.array(y_true_lesion), np.array(y_pred_lesion),
                                                     savedir=save_dir + '/Roc_curve/valid_l_' + str(epoch) + '.jpg',
                                                     re_auc=True)
    return loss / total_v, Accuracy_v, Accuracy_l, optim_threshold_v, optim_threshold_l, auc_v, auc_l


# test
def test(epoch, dataloader, Net, loss_fn, threshold_v, threshold_l):
    Net.eval()
    loss = 0
    hist_video = np.zeros((2, 2))
    hist_lesion = np.zeros((2, 2))
    lesion_name = ''
    pred_lesion = 0
    count_video = 0
    y_true_video = []
    y_pred_video = []
    y_true_lesion = []
    y_pred_lesion = []
    with torch.no_grad():
        for i, (data, graph, target, e_img, name, stime) in enumerate(dataloader):
            data, target, graph, e_img = data.to(device), target.to(device), graph.to(device), e_img.to(device)
            data = data.permute(0, 4, 2, 3, 1).contiguous()
            e_img = e_img.permute(0, 3, 1, 2).contiguous()
            pred = Net(data, e_img, graph)
            losses = loss_fn(pred, torch.unsqueeze(target, dim=-1))
            pred_ge = torch.ge(pred, threshold_v).cpu().numpy()
            target = target.cpu().numpy()
            pred = pred.cpu().numpy()
            for batch in range(data.shape[0]):
                if not lesion_name:  # first lesion
                    tar_lesion = target[batch]
                    lesion_name = name[batch]
                    print(lesion_name)
                if lesion_name != name[batch]:
                    pred_lesion_avg = pred_lesion / count_video
                    print('lesion label pred: %d(%.2f), true: %d' % (int(pred_lesion_avg > threshold_l), pred_lesion_avg, tar_lesion))
                    hist_lesion[int(tar_lesion), int(pred_lesion_avg > threshold_l)] += 1
                    y_true_lesion.append(tar_lesion)
                    y_pred_lesion.append(pred_lesion_avg)
                    pred_lesion = 0
                    count_video = 0
                    tar_lesion = target[batch]
                    lesion_name = name[batch]
                    print(lesion_name)

                y_true_video.append(target[batch].item())
                y_pred_video.append(pred[batch].item())
                count_video += 1
                pred_lesion += pred[batch].item()
                hist_video[int(target[batch]), int(pred_ge[batch])] += 1
                #print('%s(s) %.4f  %d' % (stime[batch], pred[batch].item(), target[batch]))
            loss += np.sum(losses.item()) * data.shape[0]

    # last lesion
    pred_lesion_avg = pred_lesion / count_video
    print('lesion label pred: %d(%.2f), true: %d' % (int(pred_lesion_avg > threshold_l), pred_lesion_avg, tar_lesion))
    hist_lesion[int(tar_lesion), int(pred_lesion_avg > threshold_l)] += 1
    y_true_lesion.append(tar_lesion)
    y_pred_lesion.append(pred_lesion_avg)

    total_v = np.sum(hist_video)
    total_l = np.sum(hist_lesion)
    Accuracy_v = np.sum(np.diag(hist_video)) / total_v
    Accuracy_l = np.sum(np.diag(hist_lesion)) / total_l

    print('Epoch %3d testing : \n loss: %f, Accuracy(video): %f, Accuracy(lesion): %f' %
          (epoch, loss / total_v, Accuracy_v, Accuracy_l))
    print('Per video')
    plot_confusion_matrix_and_scores(hist_video, savedir=save_dir + '/confusion_matrix/test_v_' + str(epoch) + '.jpg')
    optim_threshold_v, optim_cf_v, auc_v = Roc_curve(np.array(y_true_video), np.array(y_pred_video),
                                                     savedir=save_dir + '/Roc_curve/test_v_' + str(epoch) + '.jpg',
                                                     re_auc=True)
    print('Per lesion')
    plot_confusion_matrix_and_scores(hist_lesion, savedir=save_dir + '/confusion_matrix/test_l_' + str(epoch) + '.jpg')
    optim_threshold_l, optim_cf_l, auc_l = Roc_curve(np.array(y_true_lesion), np.array(y_pred_lesion),
                                                     savedir=save_dir + '/Roc_curve/test_l_' + str(epoch) + '.jpg',
                                                     re_auc=True)
    return loss / total_v, Accuracy_v, Accuracy_l, optim_threshold_v, optim_threshold_l, auc_v, auc_l


train_batch = 6
valid_batch = 12
test_batch = 12
sample_rate = 0.25
time_steps = 12
number_classes = 1
threshold = 0.295

# model = Resnet3DDoppler(num_classes=1, init_weights=True).to(device)
model = nn.DataParallel(R3D_UDE(num_classes=number_classes, time_steps=time_steps, init_weights=True, mode='after'),
                        device_ids=[0, 1])
model.to(device)

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

print(f'Total params: {Total_params}')
print(f'Trainable params: {Trainable_params}')
print(f'Non-trainable params: {NonTrainable_params}\n')

'''with open('augmentation.json', 'r') as f:
    aug_dict = json.load(f)
training_augmentation = get_composed_augmentations(aug_dict)'''
aug_list = [('GaussianBlur', 0.3), ('AddGaussianNoise', 0.3), ('RandomHorizontalFlip', 0.5)]

# video_dir, UD_dir, E_dir, excel_dir, sample_rate, time_steps, number_classes, dataset_split, augmentation=None, UD_reconstruct=False, E_reconstruct=False, crop_size='r1', E_crop_size='Er1', resize=None, E_resize=None):


Epoch = 100
learning_rate = 4e-5
loss_fn = nn.BCELoss()
# weights_CE = torch.FloatTensor([0.66, 1]).to(device)
# Criterion = nn.CrossEntropyLoss(weight=weights_CE)
# loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

# para visualize
lr = []
train_loss = []
train_acc = []
valid_loss = []
valid_acc_video = []
valid_acc_lesion = []
valid_auc_video = []
valid_auc_lesion = []

max_auc = 0
max_acc = 0
best_epoch = 16

# Main
is_training = 1
is_testing = 1

if is_training:
    TrainingDataset = Dataset3D('data_lesion_0524', 'data_lesion_0524.xlsx', 'data/two_stream_data_old/UD_clip', doppler=True, elastography=True,
                                E_dir='data/two_stream_data_old/E_img', sample_rate=sample_rate,
                                time_steps=time_steps, number_classes=number_classes, dataset_split=['train'],
                                augmentation=aug_list, UD_reconstruct=False, E_reconstruct=False,
                                UD_shape='THWC', E_shape='HWC')
    ValidationDataset = Dataset3D('data_lesion_0524', 'data_lesion_0524.xlsx', 'data/two_stream_data_old/UD_clip', doppler=True, elastography=True,
                                  E_dir='data/two_stream_data_old/E_img', sample_rate=sample_rate,
                                  time_steps=time_steps, number_classes=number_classes, dataset_split=['valid'],
                                  UD_reconstruct=False, E_reconstruct=False, UD_shape='THWC', E_shape='HWC')
    TrainingLoader = DataLoader(TrainingDataset, batch_size=train_batch, shuffle=True, num_workers=4)
    ValidationLoader = DataLoader(ValidationDataset, batch_size=valid_batch, shuffle=False, num_workers=4)
    for epoch in range(1, Epoch):
        print("Epoch: %d learning_rate: %f" % (epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        t_loss, t_acc = train(epoch, TrainingLoader, model, optimizer, loss_fn)
        v_loss, v_Acc_video, v_Acc_lesion, optim_th_video, optim_th_lesion, aucv, aucl = valid(epoch, ValidationLoader,
                                                                                               model, loss_fn)

        lr.append(optimizer.state_dict()['param_groups'][0]['lr'])
        train_loss.append(t_loss)
        train_acc.append(t_acc)
        valid_loss.append(v_loss)
        valid_acc_video.append(v_Acc_video)
        valid_acc_lesion.append(v_Acc_lesion)
        valid_auc_video.append(aucv)
        valid_auc_lesion.append(aucl)

        plot_fig(np.array(lr), savedir=save_dir + "/lr.jpg")
        plot_fig(np.array(train_loss), savedir=save_dir + "/train_loss.jpg")
        plot_fig(np.array(train_acc), savedir=save_dir + "/train_acc.jpg")
        plot_fig(np.array(valid_loss), savedir=save_dir + "/valid_loss.jpg")
        plot_fig(np.array(valid_acc_video), savedir=save_dir + "/valid_acc_video.jpg")
        plot_fig(np.array(valid_acc_lesion), savedir=save_dir + "/valid_acc_lesion.jpg")
        plot_fig(np.array(valid_auc_video), savedir=save_dir + "/valid_auc_video.jpg")
        plot_fig(np.array(valid_auc_lesion), savedir=save_dir + "/valid_auc_lesion.jpg")

        cosine_decay(learning_rate, epoch, Epoch, optimizer, alpha=0.0)

        if aucl > max_auc:
            max_auc = aucl
            torch.save(model.state_dict(), save_dir + "/model_state_dict/ResUDE_" + str(epoch))
        elif v_Acc_lesion > max_acc:
            max_acc = v_Acc_lesion
            torch.save(model.state_dict(), save_dir + "/model_state_dict/ResUDE_" + str(epoch))

# testing
if is_testing:
    TestingDataset = Dataset3D('data/raw_video', 'data_lesion_0524.xlsx', 'data/two_stream_data_old/UD_clip',
                               doppler=True, elastography=True,
                               E_dir='data/two_stream_data_old/E_img', sample_rate=sample_rate,
                               time_steps=time_steps, number_classes=number_classes, dataset_split=['test'],
                               UD_reconstruct=False, E_reconstruct=False, UD_shape='THWC', E_shape='HWC')
    TestingLoader = DataLoader(TestingDataset, batch_size=test_batch, shuffle=False, num_workers=4)
    epoch = best_epoch
    model.load_state_dict(torch.load(save_dir + "/model_state_dict/ResUDE_" + str(epoch)))
    test_loss, test_Acc_video, test_Acc_lesion, th_v, th_l, auc_v, auc_l = test(epoch, TestingLoader, model, loss_fn,
                                                                                threshold, threshold)
