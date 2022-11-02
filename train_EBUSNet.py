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
from models.EBUSnet import EBUSNet
from torch.nn.modules.loss import _Loss
from matplotlib import pyplot as plt
from dataset0530 import DatasetEBUSNet
from torch.utils.data import DataLoader
import seaborn as sns
from math import cos, pi
from utils import plot_confusion_matrix_and_scores, Roc_curve, plot_fig

parser = argparse.ArgumentParser()
parser.add_argument("--train", default=False, type=int, help="re-train the model")
parser.add_argument("--test", default=True, type=int, help="reproduce the testing result")
parser.add_argument("--save_dir", default='savemodel220620_EBUSnet', type=str, help="assign save folder")
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
#torch.cuda.set_device(1)

#model = EBUSNet(num_classes=1, init_weights=True).to(device)
model = nn.DataParallel(EBUSNet(init_weights=True), device_ids=[0, 1])
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
print(f'Non-trainable params: {NonTrainable_params}')

def cosine_decay(ini_lr, global_step, decay_steps, optim, alpha=0.0):
    global_step = min(global_step, decay_steps)
    cos_decay = 0.5 * (1 + cos(pi * global_step / decay_steps))
    decayed_coff = (1 - alpha) * cos_decay + alpha
    decayed_learning_rate = ini_lr * decayed_coff
    for param_group in optim.param_groups:
        param_group['lr'] = decayed_learning_rate

def train(epoch, dataloader, Net, optimizer, loss_fn):
    Net.train()
    loss = 0
    count = 0
    correct = 0
    hist = np.zeros((2, 2))
    for i, (U, D, E, target, name) in enumerate(tqdm(dataloader, ncols=80)):
        U, D, E, target = U.to(device), D.to(device), E.to(device), target.to(device)
        U = U.permute(0, 3, 1, 2).contiguous()
        D = D.permute(0, 3, 1, 2).contiguous()
        E = E.permute(0, 3, 1, 2).contiguous()
        optimizer.zero_grad()
        pred   = Net(U, D, E)
        losses = loss_fn(torch.squeeze(pred, dim=1), target)
        losses.backward()
        optimizer.step()
        pred_ge = torch.ge(pred, 0.5).cpu().numpy()
        target  = target.cpu().numpy()
        for batch in range(U.shape[0]):
            count += 1
            correct += pred_ge[batch] == target[batch]
            hist[int(target[batch]), int(pred_ge[batch])] += 1
        loss += np.sum(losses.item()) * U.shape[0]

    print('Epoch %3d training: \n loss: %f, accuracy: %f' % (epoch, loss / count, correct / count))
    plot_confusion_matrix_and_scores(hist, savedir=save_dir + '/confusion_matrix/train_'+str(epoch)+'.jpg')
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
    y_true_img = []
    y_pred_img = []
    y_true_lesion = []
    y_pred_lesion = []
    with torch.no_grad():
        for i, (U, D, E, target, name) in enumerate(dataloader):
            U, D, E, target = U.to(device), D.to(device), E.to(device), target.to(device)
            U = U.permute(0, 3, 1, 2).contiguous()
            D = D.permute(0, 3, 1, 2).contiguous()
            E = E.permute(0, 3, 1, 2).contiguous()
            pred   = Net(U, D, E)
            losses = loss_fn(pred, torch.unsqueeze(target, dim=-1))
            pred_ge = torch.ge(pred, threshold).cpu().numpy()
            target = target.cpu().numpy()
            pred = pred.cpu().numpy()
            for batch in range(U.shape[0]):
                if not lesion_name:  # first lesion
                    tar_lesion = target[batch]
                    lesion_name = name[batch]
                    print(lesion_name)
                if lesion_name != name[batch]:
                    pred_lesion_avg = pred_lesion / count_video
                    print('lesion label pred: %d(%.2f), true: %d' % (int(pred_lesion_avg > threshold), pred_lesion_avg, tar_lesion))
                    hist_lesion[int(tar_lesion), int(pred_lesion_avg > threshold)] += 1
                    y_true_lesion.append(tar_lesion)
                    y_pred_lesion.append(pred_lesion_avg)
                    pred_lesion = 0
                    count_video = 0
                    tar_lesion = target[batch]
                    lesion_name = name[batch]
                    print(lesion_name)

                y_true_img.append(target[batch].item())
                y_pred_img.append(pred[batch].item())
                count_video += 1
                pred_lesion += pred[batch].item()
                hist_video[int(target[batch]), int(pred_ge[batch])] += 1
                #print('%.4f  %d' % (pred[batch].item(), target[batch]))
            loss += np.sum(losses.item()) * U.shape[0]

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

    print('Per lesion')
    plot_confusion_matrix_and_scores(hist_lesion, savedir=save_dir + '/confusion_matrix/valid_l_' + str(epoch) + '.jpg')
    optim_threshold_l, optim_cf_l, auc_l = Roc_curve(np.array(y_true_lesion), np.array(y_pred_lesion),
                                                     savedir=save_dir + '/Roc_curve/valid_l_' + str(epoch) + '.jpg',
                                                     re_auc=True)
    return loss / total_v, Accuracy_v, Accuracy_l, optim_threshold_l, auc_l

# test
def test(epoch, dataloader, Net, loss_fn):
    Net.eval()
    loss = 0
    hist_video = np.zeros((2, 2))
    hist_lesion = np.zeros((2, 2))
    lesion_name = ''
    pred_lesion = 0
    count_video = 0
    y_true_img = []
    y_pred_img = []
    y_true_lesion = []
    y_pred_lesion = []
    with torch.no_grad():
        for i, (U, D, E, target, name) in enumerate(dataloader):
            U, D, E, target = U.to(device), D.to(device), E.to(device), target.to(device)
            U = U.permute(0, 3, 1, 2).contiguous()
            D = D.permute(0, 3, 1, 2).contiguous()
            E = E.permute(0, 3, 1, 2).contiguous()
            pred   = Net(U, D, E)
            losses = loss_fn(pred, torch.unsqueeze(target, dim=-1))
            pred_ge = torch.ge(pred, threshold).cpu().numpy()
            target = target.cpu().numpy()
            pred = pred.cpu().numpy()
            for batch in range(U.shape[0]):
                if not lesion_name:  # first lesion
                    tar_lesion = target[batch]
                    lesion_name = name[batch]
                    print(lesion_name)
                if lesion_name != name[batch]:
                    pred_lesion_avg = pred_lesion / count_video
                    print('lesion label pred: %d(%.2f), true: %d' % (int(pred_lesion_avg > threshold), pred_lesion_avg, tar_lesion))
                    hist_lesion[int(tar_lesion), int(pred_lesion_avg > threshold)] += 1
                    y_true_lesion.append(tar_lesion)
                    y_pred_lesion.append(pred_lesion_avg)
                    pred_lesion = 0
                    count_video = 0
                    tar_lesion = target[batch]
                    lesion_name = name[batch]
                    print(lesion_name)

                y_true_img.append(target[batch].item())
                y_pred_img.append(pred[batch].item())
                count_video += 1
                pred_lesion += pred[batch].item()
                hist_video[int(target[batch]), int(pred_ge[batch])] += 1
                #print('%.4f  %d' % (pred[batch].item(), target[batch]))
            loss += np.sum(losses.item()) * U.shape[0]

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

    print('Epoch %3d testing : \n loss: %f, Accuracy(video): %f, Accuracy(lesion): %f' % (
        epoch, loss / total_v, Accuracy_v, Accuracy_l))
    print('Per video')
    plot_confusion_matrix_and_scores(hist_video, savedir=save_dir + '/confusion_matrix/test_v_' + str(epoch) + '.jpg')

    print('Per lesion')
    plot_confusion_matrix_and_scores(hist_lesion, savedir=save_dir + '/confusion_matrix/test_l_' + str(epoch) + '.jpg')
    optim_threshold_l, optim_cf_l, auc_l = Roc_curve(np.array(y_true_lesion), np.array(y_pred_lesion),
                                                     savedir=save_dir + '/Roc_curve/test_l_' + str(epoch) + '.jpg',
                                                     re_auc=True)
    return loss / total_v, Accuracy_v, Accuracy_l, optim_threshold_l, auc_l

train_batch = 128
valid_batch = 128
sample_sate = 0.25
threshold = 0.3

with open('augmentation.json', 'r') as f:
    aug_dict = json.load(f)
training_augmentation = get_composed_augmentations(aug_dict)

Epoch = 450
learning_rate = 0.001
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

# para visualize
lr = []
train_loss = []
train_acc = []
valid_loss = []
valid_acc_img = []
valid_acc_lesion = []
valid_auc_lesion = []

max_auc = 0
max_acc = 0

# Main
is_training = args.train
is_testing = args.test

if is_training:
    TrainingDataset = DatasetEBUSNet('data/raw_video', 'data/2D_data/U_img', 'data/2D_data/D_img',
                                     'data/2D_data/E_img', 'data_lesion_0524.xlsx', 0.25, 20, 1, split=['train'],
                                     reconstruct=False)
    ValidationDataset = DatasetEBUSNet('data/raw_video', 'data/2D_data/U_img', 'data/2D_data/D_img',
                                       'data/2D_data/E_img', 'data_lesion_0524.xlsx', 0.25, 20, 1, split=['valid'],
                                       reconstruct=False)

    TrainingLoader = DataLoader(TrainingDataset, batch_size=train_batch, shuffle=True, num_workers=4)
    ValidationLoader = DataLoader(ValidationDataset, batch_size=valid_batch, shuffle=False, num_workers=4)
    for epoch in range(1, Epoch):
        print("Epoch: %d learning_rate: %f" % (epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        t_loss, t_acc = train(epoch, TrainingLoader, model, optimizer, loss_fn)
        v_loss, v_Acc_video, v_Acc_lesion, optim_th_lesion, aucl = valid(epoch, ValidationLoader, model, loss_fn)

        lr.append(optimizer.state_dict()['param_groups'][0]['lr'])
        train_loss.append(t_loss)
        train_acc.append(t_acc)
        valid_loss.append(v_loss)
        valid_acc_img.append(v_Acc_video)
        valid_acc_lesion.append(v_Acc_lesion)
        valid_auc_lesion.append(aucl)

        plot_fig(np.array(lr), savedir=save_dir + "/lr.jpg")
        plot_fig(np.array(train_loss), savedir=save_dir + "/train_loss.jpg")
        plot_fig(np.array(train_acc), savedir=save_dir + "/train_acc.jpg")
        plot_fig(np.array(valid_loss), savedir=save_dir + "/valid_loss.jpg")
        plot_fig(np.array(valid_acc_img), savedir=save_dir + "/valid_acc_video.jpg")
        plot_fig(np.array(valid_acc_lesion), savedir=save_dir + "/valid_acc_lesion.jpg")
        plot_fig(np.array(valid_auc_lesion), savedir=save_dir + "/valid_auc_lesion.jpg")

        if epoch > 20 and epoch % 5 == 0 and optimizer.state_dict()['param_groups'][0]['lr'] > 4e-5:
            for param_group in optimizer.param_groups:
                param_group['lr'] = optimizer.state_dict()['param_groups'][0]['lr'] / 1.05776756

        if aucl > max_auc:
            max_auc = aucl
            torch.save(model.state_dict(), save_dir + "/model_state_dict/EBUSNet_" + str(epoch))
        elif v_Acc_lesion > max_acc:
            max_acc = v_Acc_lesion
            torch.save(model.state_dict(), save_dir + "/model_state_dict/EBUSNet_" + str(epoch))

# testing
if is_testing:
    TestingDataset = DatasetEBUSNet('data/raw_video', 'data/2D_data/U_img', 'data/2D_data/D_img',
                                    'data/2D_data/E_img', 'data_lesion_0524.xlsx', 0.25, 20, 1, split=['test'],
                                    reconstruct=False)
    TestingLoader = DataLoader(TestingDataset, batch_size=valid_batch, shuffle=False, num_workers=8)

    epoch = 242
    model.load_state_dict(torch.load(save_dir + "/model_state_dict/EBUSNet_" + str(epoch)))
    test_loss, test_Acc_img, test_Acc_lesion, th_l, auc_l = test(epoch, TestingLoader, model, loss_fn)
