import cv2
import sys
import time
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
# import torchvision.models as model
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from augmentation import get_composed_augmentations
import json
from models.TransEBUS_v2 import TransEBUS
from torch.nn.modules.loss import _Loss
from matplotlib import pyplot as plt
from datasets import Dataset3D_MoCo
from torch.utils.data import DataLoader
import seaborn as sns
from math import cos, pi
from utils import plot_confusion_matrix_and_scores, Roc_curve, plot_fig, cosine_decay
import random
from thop import profile

save_dir = 'savemodel221017_TransEBUS_3S_MoCo'

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# torch.cuda.set_device(1)
def momentum_step(m=1.):
    '''
    Momentum step (Eq (2)).
    Args:
        - m (float): momentum value. 1) m = 0 -> copy parameter of encoder to key encoder
                                     2) m = 0.999 -> momentum update of key encoder
    '''
    params_q = encoder.state_dict()
    params_k = momentum_encoder.state_dict()

    dict_params_k = dict(params_k)

    for name in params_q:
        theta_k = dict_params_k[name]
        theta_q = params_q[name].data
        dict_params_k[name].data.copy_(m * theta_k + (1 - m) * theta_q)

    momentum_encoder.load_state_dict(dict_params_k)


def train(epoch, dataloader, Net, optimizer, loss_fn):
    Net.train()
    loss = 0
    count = 0
    correct = 0
    hist = np.zeros((2, 2))
    for i, (data, graph, target, e_img, name, stime) in enumerate(tqdm(dataloader, ncols=80)):
        data, target, graph, e_img = data.to(device), target.to(device), graph.to(device), e_img.to(device)
        optimizer.zero_grad()
        pred = Net((data, e_img))
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


def train_MOCO(epoch, dataloader, encoder, momentum_encoder, queue_0, queue_1, optimizer, loss_fn, temperature=1):
    total_loss_CE = 0
    total_loss_0 = 0
    total_loss_1 = 0
    count = 0
    count_0 = 0
    count_1 = 0
    hist = np.zeros((2, 2))
    for i, (data_q, data_k, graph, target, _, _) in enumerate(tqdm(dataloader, ncols=80)):
        # Preprocess
        encoder.train()
        momentum_encoder.train()
        encoder.zero_grad()
        x_q, e_q, graph = data_q[0].to(device), data_q[1].to(device), graph.to(device)
        x_k, e_k = data_k[0].to(device), data_k[1].to(device)
        target = target.to(device)
        '''
        # Shffled BN : shuffle x_k before distributing it among GPUs (Section. 3.3)
        if config.shuffle_bn:
            idx = torch.randperm(x_k.size(0))
            x_k = x_k[idx]
        '''
        pred_q, q = encoder((x_q, e_q, graph))  # q : (N, 128)
        pred_k, k = momentum_encoder((x_k, e_k, graph))  # k : (N, 128)
        pred_k = pred_k.detach()
        k = k.detach()

        # divide different classes
        q_0 = q[(target == 0).nonzero(as_tuple=True)[0]]  # (N_0, 128)
        q_1 = q[(target == 1).nonzero(as_tuple=True)[0]]  # (N_1, 128)
        k_0 = k[(target == 0).nonzero(as_tuple=True)[0]]  # (N_0, 128)
        k_1 = k[(target == 1).nonzero(as_tuple=True)[0]]  # (N_1, 128)
        '''
        # Shuffled BN : unshuffle k (Section. 3.3)
        if config.shuffle_bn:
            k_temp = torch.zeros_like(k)
            for i, j in enumerate(idx):
                k_temp[j] = k[i]
            k = k_temp
        '''

        # Positive sampling q & k
        l_pos_0 = torch.sum(q_0 * k_0, dim=1, keepdim=True)  # (N, 1)
        l_pos_1 = torch.sum(q_1 * k_1, dim=1, keepdim=True)  # (N, 1)

        # Negative sampling q & queue
        l_neg_0 = torch.mm(q_0, queue_1.t())  # (N, 512)
        l_neg_1 = torch.mm(q_1, queue_0.t())  # (N, 512)

        # Logit and label
        logits_0 = torch.cat([l_pos_0, l_neg_0], dim=1)  # (N, 513) with label [0, 0, ..., 0]
        logits_1 = torch.cat([l_pos_1, l_neg_1], dim=1)  # (N, 513) with label [0, 0, ..., 0]
        labels_0 = torch.zeros(logits_0.size(0), dtype=torch.long).to(device)
        labels_1 = torch.zeros(logits_1.size(0), dtype=torch.long).to(device)

        # Get loss
        loss = loss_fn(pred_q, target)
        loss_0 = loss_fn(logits_0 / temperature, labels_0)
        loss_1 = loss_fn(logits_1 / temperature, labels_1)
        loss_CL = (loss_0 + loss_1) / 2

        # BP
        loss.backward(retain_graph=True)
        loss_CL.backward()

        # Encoder update
        optimizer.step()

        # Momentum encoder update
        momentum_step(m=0.99)

        # Update dictionary
        queue_0 = torch.cat([k_0, queue_0[:queue_0.size(0) - k_0.size(0)]], dim=0)
        queue_1 = torch.cat([k_1, queue_1[:queue_1.size(0) - k_1.size(0)]], dim=0)

        # performence
        target = target.cpu().numpy()
        pred_ge = torch.argmax(pred_q, dim=1).detach().cpu().numpy()
        for idx in range(k.shape[0]):
            hist[int(target[idx]), int(pred_ge[idx])] += 1

        total_loss_CE += np.sum(loss.item()) * k.shape[0]
        total_loss_0 += np.sum(loss_0.item()) * k_0.shape[0]
        total_loss_1 += np.sum(loss_1.item()) * k_1.shape[0]

        count += k.shape[0]
        count_0 += k_0.shape[0]
        count_1 += k_1.shape[0]

    correct = np.sum(np.diag(hist))
    plot_confusion_matrix_and_scores(hist, savedir=save_dir + '/confusion_matrix/train_cf' + str(epoch) + '.jpg')
    print('Epoch %3d training: loss_BCE: %f,  loss_pos: %f, loss_neg: %f, accuracy: %f' % (
        epoch, total_loss_CE / count, total_loss_1 / count_1, total_loss_0 / count_0, correct / count))
    return total_loss_CE / count, total_loss_1 / count_1, total_loss_0 / count_0, correct / count


# valid
def valid(epoch, dataloader, encoder, loss_fn):
    encoder.eval()
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
        for i, (data, e_img, graph, target, name, num_clip) in enumerate(tqdm(dataloader, ncols=80)):
            data, target, e_img, graph = data.to(device), target.to(device), e_img.to(device), graph.to(device)
            pred, q = encoder((data, e_img, graph))
            losses = loss_fn(pred, target)
            pred_ge = torch.argmax(pred, dim=1).detach().cpu().numpy()
            target = target.cpu().numpy()
            pred = pred.cpu().numpy()
            for batch in range(data.shape[0]):
                if not lesion_name:  # first lesion
                    tar_lesion = target[batch]
                    lesion_name = name[batch]
                    # print(lesion_name)
                if lesion_name != name[batch]:
                    pred_lesion_avg = pred_lesion / count_video
                    # print('lesion label pred: %d(%.2f), true: %d' % (int(pred_lesion_avg > threshold), pred_lesion_avg, tar_lesion))
                    hist_lesion[int(tar_lesion), int(pred_lesion_avg > threshold)] += 1
                    y_true_lesion.append(tar_lesion)
                    y_pred_lesion.append(pred_lesion_avg)
                    pred_lesion = 0
                    count_video = 0
                    tar_lesion = target[batch]
                    lesion_name = name[batch]
                    # print(lesion_name)

                y_true_video.append(target[batch].item())
                y_pred_video.append(pred[batch][1].item())
                count_video += 1
                pred_lesion += pred[batch][1].item()
                hist_video[int(target[batch]), int(pred_ge[batch])] += 1
                # print('%s(s) %.4f  %d' % (stime[batch], pred[batch].item(), target[batch]))
            loss += np.sum(losses.item()) * data.shape[0]

    # last lesion
    pred_lesion_avg = pred_lesion / count_video
    # print('lesion label pred: %d(%.2f), true: %d' % (int(pred_lesion_avg > threshold), pred_lesion_avg, tar_lesion))
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
    return loss / total_v, Accuracy_v, Accuracy_l, auc_v, auc_l


# test
def test(epoch, dataloader, encoder, loss_fn):
    encoder.eval()
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
        for i, (data, e_img, graph, target, name, num_clip) in enumerate(tqdm(dataloader, ncols=80)):
            data, target, e_img, graph = data.to(device), target.to(device), e_img.to(device), graph.to(device)
            pred, q = encoder((data, e_img, graph))
            losses = loss_fn(pred, target)
            pred_ge = torch.argmax(pred, dim=1).detach().cpu().numpy()
            target = target.cpu().numpy()
            pred = pred.cpu().numpy()
            for batch in range(data.shape[0]):
                if not lesion_name:  # first lesion
                    tar_lesion = target[batch]
                    lesion_name = name[batch]
                    # print(lesion_name)
                if lesion_name != name[batch]:
                    pred_lesion_avg = pred_lesion / count_video
                    # print('lesion label pred: %d(%.2f), true: %d' % (
                    #    int(pred_lesion_avg > threshold_l), pred_lesion_avg, tar_lesion))
                    hist_lesion[int(tar_lesion), int(pred_lesion_avg > threshold)] += 1
                    y_true_lesion.append(tar_lesion)
                    y_pred_lesion.append(pred_lesion_avg)
                    pred_lesion = 0
                    count_video = 0
                    tar_lesion = target[batch]
                    lesion_name = name[batch]
                    # print(lesion_name)

                y_true_video.append(target[batch].item())
                y_pred_video.append(pred[batch][1].item())
                count_video += 1
                pred_lesion += pred[batch][1].item()
                hist_video[int(target[batch]), int(pred_ge[batch])] += 1
                # print('%s(s) %.4f  %d' % (stime[batch], pred[batch].item(), target[batch]))
            loss += np.sum(losses.item()) * data.shape[0]

    # last lesion
    pred_lesion_avg = pred_lesion / count_video
    # print('lesion label pred: %d(%.2f), true: %d' % (int(pred_lesion_avg > threshold_l), pred_lesion_avg, tar_lesion))
    hist_lesion[int(tar_lesion), int(pred_lesion_avg > threshold)] += 1
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
    return loss / total_v, Accuracy_v, Accuracy_l, auc_v, auc_l


encoder = TransEBUS(
    two_stream=True,
    GD_split=True,
    MoCo_dim=128,
    img_size=(224, 224, 12),
    embedding_dim=768,
    n_conv_layers=4,
    kernel_size=7,
    stride=2,
    padding=3,
    pooling_kernel_size=3,
    pooling_stride=2,
    pooling_padding=1,
    num_layers=8,
    num_heads=6,
    mlp_radio=2.,
    num_classes=2,
    positional_embedding='none',  # ['sine', 'learnable', 'none']
)

momentum_encoder = TransEBUS(
    two_stream=True,
    GD_split=True,
    MoCo_dim=128,
    img_size=(224, 224, 12),
    embedding_dim=768,
    n_conv_layers=4,
    kernel_size=7,
    stride=2,
    padding=3,
    pooling_kernel_size=3,
    pooling_stride=2,
    pooling_padding=1,
    num_layers=8,
    num_heads=6,
    mlp_radio=2.,
    num_classes=2,
    positional_embedding='none',  # ['sine', 'learnable', 'none']
)

momentum_step(m=0)

inp_x = torch.zeros((1, 3, 224, 224, 12))
inp_e = torch.zeros((1, 3, 224, 224))
graph = torch.zeros((1, 12))
flops, params = profile(momentum_encoder, ((inp_x, inp_e, graph),))

Total_params = 0
Trainable_params = 0
NonTrainable_params = 0

for param in encoder.parameters():
    mulValue = np.prod(param.size())
    Total_params += mulValue
    if param.requires_grad:
        Trainable_params += mulValue
    else:
        NonTrainable_params += mulValue

print('Model name: TransEBUS_v2')
print('Save dir:', save_dir)
print(f'Total params: {Total_params}')
print(f'Trainable params: {Trainable_params}')
print(f'Non-trainable params: {NonTrainable_params}')
print(f'floating point operations: {flops}')

encoder = nn.DataParallel(encoder, device_ids=[0, 1])
encoder.to(device)
momentum_encoder = nn.DataParallel(momentum_encoder, device_ids=[0, 1])
momentum_encoder.to(device)

train_batch = 24
valid_batch = 24
test_batch = 24
sample_rate = 0.25
time_steps = 12
number_classes = 1
threshold = 0.3
Epoch = 200
num_keys = 576
learning_rate = 1e-4
loss_fn = nn.CrossEntropyLoss()

'''
with open('augmentation.json', 'r') as f:
    aug_dict = json.load(f)
training_augmentation = get_composed_augmentations(aug_dict)
'''
aug_list = [('GaussianBlur', 0.3), ('AddGaussianNoise', 0.3), ('RandomHorizontalFlip', 0.5)]

# para visualize
lr_list = []
train_loss_list = []
train_loss_neg_list = []
train_loss_pos_list = []
train_Acc_list = []
valid_loss_list = []
valid_Acc_video_list = []
valid_Acc_lesion_list = []
valid_auc_video_list = []
valid_auc_lesion_list = []

max_auc = 0
min_loss = 100
best_epoch = 39

# Main
is_training = 1
is_testing = 1

if is_training:
    f = open(save_dir + "/epoch_log.txt", 'w')
    f.close()
    TrainingDataset = Dataset3D_MoCo('data/lesion_video', 'data_lesion_0728.xlsx',
                                     UD_dir='data/two_stream_data/UD_clip/',
                                     E_dir='data/two_stream_data/E_img/', doppler=True, elastography=True,
                                     sample_rate=sample_rate,
                                     time_steps=time_steps, number_classes=number_classes, dataset_split=['train'],
                                     augmentation=aug_list, dataset_build=False,
                                     resize=(224, 224))
    ValidationDataset = Dataset3D_MoCo('data/lesion_video', 'data_lesion_0728.xlsx',
                                       UD_dir='data/two_stream_data/UD_clip/',
                                       E_dir='data/two_stream_data/E_img/', doppler=True, elastography=True,
                                       sample_rate=sample_rate,
                                       time_steps=time_steps, number_classes=number_classes, dataset_split=['valid'],
                                       dataset_build=False, resize=(224, 224))
    TrainingLoader = DataLoader(TrainingDataset, batch_size=train_batch, shuffle=True, num_workers=12, pin_memory=True)
    ValidationLoader = DataLoader(ValidationDataset, batch_size=valid_batch, shuffle=False, num_workers=12,
                                  pin_memory=True)

    optimizer = optim.SGD(encoder.parameters(), learning_rate, momentum=0.9, weight_decay=5e-4)

    # Initialize queue.
    print('\nInitializing a queue with %d keys.' % num_keys)
    queue_0 = []
    queue_1 = []
    with torch.no_grad():
        for i, (data_q, data_k, graph, target, _, _) in enumerate(TrainingLoader):
            _, key_feature = momentum_encoder((data_k[0].to(device), data_k[1].to(device), graph.to(device)))
            for idx in range(key_feature.shape[0]):
                if int(target[idx].numpy()) == 0 and len(queue_0) < num_keys:
                    queue_0.append(torch.unsqueeze(key_feature[idx], dim=0))
                elif int(target[idx].numpy()) == 1 and len(queue_1) < num_keys:
                    queue_1.append(torch.unsqueeze(key_feature[idx], dim=0))
            if len(queue_0) >= num_keys and len(queue_1) >= num_keys:
                break

        queue_0 = torch.cat(queue_0, dim=0)
        queue_1 = torch.cat(queue_1, dim=0)

    for epoch in range(1, Epoch + 1):
        f = open(save_dir + "/epoch_log.txt", 'a')
        print("Epoch: %d learning_rate: %f" % (epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        train_loss, train_loss_1, train_loss_0, train_Acc = train_MOCO(epoch, TrainingLoader, encoder, momentum_encoder,
                                                                       queue_0, queue_1,
                                                                       optimizer, loss_fn)
        valid_loss, valid_Acc_video, valid_Acc_lesion, valid_auc_video, valid_auc_lesion = valid(epoch,
                                                                                                 ValidationLoader,
                                                                                                 encoder,
                                                                                                 loss_fn)

        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        train_loss_list.append(train_loss)
        train_loss_neg_list.append(train_loss_0)
        train_loss_pos_list.append(train_loss_1)
        train_Acc_list.append(train_Acc)
        valid_loss_list.append(valid_loss)
        valid_Acc_video_list.append(valid_Acc_video)
        valid_Acc_lesion_list.append(valid_Acc_lesion)
        valid_auc_video_list.append(valid_auc_video)
        valid_auc_lesion_list.append(valid_auc_lesion)

        plot_fig(np.array(lr_list), savedir=save_dir + "/lr.jpg", title='Learning rate')
        plot_fig(np.array(train_loss_list), np.array(valid_loss_list), legend=['train', 'valid'],
                 savedir=save_dir + "/loss.jpg", title='Cross-entropy loss')
        plot_fig(np.array(train_loss_neg_list), savedir=save_dir + "/loss_neg.jpg")
        plot_fig(np.array(train_loss_pos_list), savedir=save_dir + "/loss_pos.jpg")
        plot_fig(np.array(train_Acc_list), savedir=save_dir + "/train_acc.jpg")
        plot_fig(np.array(valid_Acc_video_list), savedir=save_dir + "/valid_acc_video.jpg")
        plot_fig(np.array(valid_Acc_lesion_list), savedir=save_dir + "/valid_acc_lesion.jpg")
        plot_fig(np.array(valid_auc_video_list), savedir=save_dir + "/valid_auc_video.jpg")
        plot_fig(np.array(valid_auc_lesion_list), savedir=save_dir + "/valid_auc_lesion.jpg")

        cosine_decay(learning_rate, epoch, Epoch, optimizer, alpha=0.0)

        if valid_auc_lesion > max_auc:
            max_auc = valid_auc_lesion
            best_epoch = epoch
            torch.save(encoder.state_dict(), save_dir + "/model_state_dict/ResUDE_" + str(epoch))
            print('epoch:', epoch, file=f)
            print('train loss: %.4f' % train_loss, 'loss_neg: %.4f' % train_loss_0, 'loss_pos: %.4f' % train_loss_1,
                  file=f)
            print('valid loss: %.4f' % valid_loss, 'valid Acc: %.2f' % valid_Acc_lesion,
                  'valid AUC: %.4f' % valid_auc_lesion, file=f)
        elif valid_loss < min_loss:
            min_loss = valid_loss
            torch.save(encoder.state_dict(), save_dir + "/model_state_dict/ResUDE_" + str(epoch))
            print('epoch:', epoch, file=f)
            print('train loss: %.4f' % train_loss, 'loss_neg: %.4f' % train_loss_0, 'loss_pos: %.4f' % train_loss_1,
                  file=f)
            print('valid loss: %.4f' % valid_loss, 'valid Acc: %.2f' % valid_Acc_lesion,
                  'valid AUC: %.4f' % valid_auc_lesion, file=f)
        f.close()

# testing
if is_testing:
    f = open(save_dir + "/epoch_log.txt", 'a')
    TestingDataset = Dataset3D_MoCo('data/lesion_video', 'data_lesion_0728.xlsx',
                                    UD_dir='data/two_stream_data/UD_clip/',
                                    E_dir='data/two_stream_data/E_img/', doppler=True, elastography=True,
                                    sample_rate=sample_rate,
                                    time_steps=time_steps, number_classes=number_classes, dataset_split=['test'],
                                    dataset_build=False, resize=(224, 224))
    TestingLoader = DataLoader(TestingDataset, batch_size=test_batch, shuffle=False, num_workers=12)

    model = encoder
    epoch = best_epoch
    model.load_state_dict(torch.load(save_dir + "/model_state_dict/ResUDE_" + str(epoch)))

    test_loss, test_Acc_video, test_Acc_lesion, test_auc_video, test_auc_lesion = test(epoch, TestingLoader,
                                                                                       encoder,
                                                                                       loss_fn)
    print('test loss : %.4f' % test_loss, 'test Acc : %.2f' % test_Acc_lesion, 'test AUC : %.4f' % test_auc_lesion,
          file=f)
