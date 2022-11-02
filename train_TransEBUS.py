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
from models.TransEBUS import TransEBUS
from torch.nn.modules.loss import _Loss
from matplotlib import pyplot as plt
from dataset0530 import Dataset3D
from torch.utils.data import DataLoader
import seaborn as sns
from math import cos, pi
from utils import plot_confusion_matrix_and_scores, Roc_curve, plot_fig, cosine_decay
from thop import profile

parser = argparse.ArgumentParser()
parser.add_argument("--train", default=False, type=int, help="re-train the model")
parser.add_argument("--test", default=True, type=int, help="reproduce the testing result")
parser.add_argument("--save_dir", default='savemodel220921_TransEBUS', type=str, help="assign save folder")
args = parser.parse_args()

save_dir = args.save_dir
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
if not os.path.isdir(save_dir + '/confusion_matrix'):
    os.mkdir(save_dir + '/confusion_matrix')
if not os.path.isdir(save_dir + '/Roc_curve'):
    os.mkdir(save_dir + '/Roc_curve')
if not os.path.isdir(save_dir + '/model_state_dict'):
    os.mkdir(save_dir + '/model_state_dict')

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
    for i, (data, graph, target, name, stime) in enumerate(tqdm(dataloader, ncols=80)):
        # data, target, graph, e_img = data.to(device), target.to(device), graph.to(device), e_img.to(device)
        # optimizer.zero_grad()
        # pred = Net(data, e_img, graph)
        data, target, graph = data.to(device), target.to(device), graph.to(device)
        optimizer.zero_grad()
        pred = Net(data)
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
        for i, (data, graph, target, name, stime) in enumerate(tqdm(dataloader, ncols=80)):
            #data, target, graph, e_img = data.to(device), target.to(device), graph.to(device), e_img.to(device)
            #pred = Net(data, e_img, graph)
            data, target, graph = data.to(device), target.to(device), graph.to(device)
            pred = Net(data)
            losses = loss_fn(torch.squeeze(pred, dim=1), target)
            pred_ge = torch.ge(pred, threshold).cpu().numpy()
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
                y_pred_video.append(pred[batch].item())
                count_video += 1
                pred_lesion += pred[batch].item()
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
        for i, (data, graph, target, name, stime) in enumerate(dataloader):
            # data, target, graph, e_img = data.to(device), target.to(device), graph.to(device), e_img.to(device)
            # pred = Net(data, e_img, graph)
            data, target, graph = data.to(device), target.to(device), graph.to(device)
            pred = Net(data)
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
                    print('lesion label pred: %d(%.2f), true: %d' % (
                        int(pred_lesion_avg > threshold_l), pred_lesion_avg, tar_lesion))
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
                print('%s(s) %.4f  %d' % (stime[batch], pred[batch].item(), target[batch]))
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


train_batch = 24
valid_batch = 24
test_batch = 24
sample_rate = 0.25
time_steps = 12
number_classes = 1
threshold = 0.3
Epoch = 600
start_up_lr = 1e-4
loss_fn = nn.BCELoss()

# model = Resnet3DDoppler(num_classes=1, init_weights=True).to(device)
model = TransEBUS(
    img_size=(224, 224, 12),
    embedding_dim=768,
    n_conv_layers=4,
    kernel_size=(7, 7, 1),
    stride=(2, 2, 1),
    padding=(3, 3, 0),
    pooling_kernel_size=(3, 3, 1),
    pooling_stride=(2, 2, 1),
    pooling_padding=(1, 1, 0),
    num_layers=8,
    num_heads=6,
    mlp_radio=2.,
    num_classes=number_classes,
    positional_embedding='learnable',  # ['sine', 'learnable', 'none']
)
#inp_x = torch.zeros((1, 3, 224, 224, 12))
#flops, params = profile(model, (inp_x,))

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

print('Model name: TransEBUS')
print('Save dir:', save_dir)
print(f'Total params: {Total_params}')
print(f'Trainable params: {Trainable_params}')
print(f'Non-trainable params: {NonTrainable_params}')
# print(f'floating point operations: {flops}')

model = nn.DataParallel(model, device_ids=[0, 1])
model.to(device)

'''
with open('augmentation.json', 'r') as f:
    aug_dict = json.load(f)
training_augmentation = get_composed_augmentations(aug_dict)
'''
aug_list = [('GaussianBlur', 0.3), ('AddGaussianNoise', 0.3), ('RandomHorizontalFlip', 0.5)]

# para visualize
lr_list = []
train_loss_list = []
train_Acc_list = []
valid_loss_list = []
valid_Acc_video_list = []
valid_Acc_lesion_list = []
valid_auc_video_list = []
valid_auc_lesion_list = []

max_auc = 0
min_loss = 100
test_epoch = 92

# Main
is_training = args.train
is_testing = args.test

if is_training:
    #f = open(save_dir + "/epoch_log.txt", 'w')
    #f.close()
    TrainingDataset = Dataset3D('data/lesion_video', 'data_lesion_0728.xlsx',
                                UD_dir='data/two_stream_data_old/UD_clip/',
                                E_dir='data/two_stream_data_old/E_img/', doppler=True, elastography=False,
                                sample_rate=sample_rate,
                                time_steps=time_steps, number_classes=number_classes, dataset_split=['train'],
                                augmentation=aug_list, UD_reconstruct=False, E_reconstruct=False,
                                resize=(224, 224))
    ValidationDataset = Dataset3D('data/lesion_video', 'data_lesion_0728.xlsx',
                                  UD_dir='data/two_stream_data_old/UD_clip/',
                                  E_dir='data/two_stream_data_old/E_img/', doppler=True, elastography=False,
                                  sample_rate=sample_rate,
                                  time_steps=time_steps, number_classes=number_classes, dataset_split=['valid'],
                                  UD_reconstruct=False, E_reconstruct=False, resize=(224, 224))
    TrainingLoader = DataLoader(TrainingDataset, batch_size=train_batch, shuffle=True, num_workers=12, pin_memory=True)
    ValidationLoader = DataLoader(ValidationDataset, batch_size=valid_batch, shuffle=False, num_workers=12,
                                  pin_memory=True)

    optimizer = optim.SGD(model.parameters(), lr=start_up_lr / 10, momentum=0.9, weight_decay=5e-4)
    '''
    # pretrain
    pretrain_path = 'savemodel220707_3UD/model_state_dict/ResUDE_23'
    state_dict = torch.load(pretrain_path)
    keys = []
    for k, v in state_dict.items():
        if 'fc' in k:
            continue
        keys.append(k)
    pretrain_dict = {k.replace('module.', ''): state_dict[k] for k in keys}
    model.load_state_dict(pretrain_dict, strict=False)
    '''
    lr_ = start_up_lr / 10
    for epoch in range(1, Epoch + 1):
        f = open(save_dir + "/epoch_log.txt", 'a')
        print("Epoch: %d learning_rate: %f" % (epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        train_loss, train_Acc = train(epoch, TrainingLoader, model, optimizer, loss_fn)
        valid_loss, valid_Acc_video, valid_Acc_lesion, valid_auc_video, valid_auc_lesion = valid(epoch,
                                                                                                 ValidationLoader,
                                                                                                 model, loss_fn)

        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        train_loss_list.append(train_loss)
        train_Acc_list.append(train_Acc)
        valid_loss_list.append(valid_loss)
        valid_Acc_video_list.append(valid_Acc_video)
        valid_Acc_lesion_list.append(valid_Acc_lesion)
        valid_auc_video_list.append(valid_auc_video)
        valid_auc_lesion_list.append(valid_auc_lesion)

        plot_fig(np.array(lr_list), savedir=save_dir + "/lr.jpg", title='Learning rate')
        plot_fig(np.array(train_loss_list), np.array(valid_loss_list), legend=['train', 'valid'],
                 savedir=save_dir + "/loss.jpg", title='Cross-entropy loss')
        plot_fig(np.array(train_Acc_list), savedir=save_dir + "/train_acc.jpg")
        plot_fig(np.array(valid_Acc_video_list), savedir=save_dir + "/valid_acc_video.jpg")
        plot_fig(np.array(valid_Acc_lesion_list), savedir=save_dir + "/valid_acc_lesion.jpg")
        plot_fig(np.array(valid_auc_video_list), savedir=save_dir + "/valid_auc_video.jpg")
        plot_fig(np.array(valid_auc_lesion_list), savedir=save_dir + "/valid_auc_lesion.jpg")

        # cosine_decay(learning_rate, epoch, Epoch, optimizer, alpha=0.0)

        if epoch < 10:  # warm-up LR
            lr_ += start_up_lr / 10
        else:
            if epoch % 100 == 0:
                if epoch % 200 == 0:
                    lr_ /= 5
                else:
                    lr_ /= 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        if valid_auc_lesion > max_auc:
            max_auc = valid_auc_lesion
            test_epoch = epoch
            torch.save(model.state_dict(), save_dir + "/model_state_dict/ResUDE_" + str(epoch))
            print('epoch:', epoch, file=f)
            print('train loss: %.4f' % train_loss, file=f)
            print('valid loss: %.4f' % valid_loss, 'valid Acc: %.2f' % valid_Acc_lesion,
                  'valid AUC: %.4f' % valid_auc_lesion, file=f)
        elif valid_loss < min_loss:
            min_loss = valid_loss
            torch.save(model.state_dict(), save_dir + "/model_state_dict/ResUDE_" + str(epoch))
            print('epoch:', epoch, file=f)
            print('train loss: %.4f' % train_loss, file=f)
            print('valid loss: %.4f' % valid_loss, 'valid Acc: %.2f' % valid_Acc_lesion,
                  'valid AUC: %.4f' % valid_auc_lesion, file=f)
        #f.close()

# testing
if is_testing:
    f = open(save_dir + "/epoch_log.txt", 'a')
    TestingDataset = Dataset3D('data/lesion_video', 'data_lesion_0728.xlsx',
                               UD_dir='data/two_stream_data_old/UD_clip/',
                               E_dir='data/two_stream_data_old/E_img/', doppler=True, elastography=False,
                               sample_rate=sample_rate,
                               time_steps=time_steps, number_classes=number_classes, dataset_split=['test'],
                               UD_reconstruct=False, E_reconstruct=False, resize=(224, 224))
    TestingLoader = DataLoader(TestingDataset, batch_size=test_batch, shuffle=False, num_workers=12)

    epoch = test_epoch
    model.load_state_dict(torch.load(save_dir + "/model_state_dict/ResUDE_" + str(epoch)))

    test_loss, test_Acc_video, test_Acc_lesion, th_v, th_l, auc_v, auc_l = test(epoch, TestingLoader, model, loss_fn,
                                                                                threshold, threshold)
    print('epoch:', epoch, 'test loss:', test_loss, 'test Acc:', test_Acc_lesion, 'test AUC:', auc_l, file=f)

