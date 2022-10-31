import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models
from models.TransEBUS import TransEBUS
from dataset0530 import Dataset3D_MOCO
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image

methods = \
    {"gradcam": GradCAM,
     "scorecam": ScoreCAM,
     "gradcam++": GradCAMPlusPlus,
     "ablationcam": AblationCAM,
     "xgradcam": XGradCAM,
     "eigencam": EigenCAM,
     "eigengradcam": EigenGradCAM,
     "layercam": LayerCAM,
     "fullgrad": FullGrad}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MoCo = True

if MoCo:
    save_dir = 'savemodel220726_CCT_SMOCO'
    model = TransEBUS(
        two_stream=True,
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
        num_classes=2,
        MoCo_dim=128,
        positional_embedding='learnable',  # ['sine', 'learnable', 'none']
    )
    pretrain_path = "savemodel220726_CCT_SMOCO/model_state_dict/ResUDE_193"
    state_dict = torch.load(pretrain_path)
    pretrain_dict = {k.replace('module.', ''): state_dict[k] for k in state_dict}
    model.load_state_dict(pretrain_dict)
else:
    save_dir = 'savemodel220921_TransEBUS_TS'
    model = TransEBUS(
        two_stream=True,
        img_size=(224, 224, 12),
        embedding_dim=768,
        n_conv_layers=4,
        kernel_size=(7, 7, 1),
        stride=(2, 2, 1),
        padding=(3, 3, 1),
        pooling_kernel_size=(3, 3, 3),
        pooling_stride=(2, 2, 2),
        pooling_padding=(1, 1, 1),
        num_layers=8,
        num_heads=6,
        mlp_radio=2.,
        num_classes=1,
        positional_embedding='learnable',  # ['sine', 'learnable', 'none']
    )
    pretrain_path = "savemodel220921_TransEBUS_TS/model_state_dict//ResUDE_67"
    state_dict = torch.load(pretrain_path)
    pretrain_dict = {k.replace('module.', ''): state_dict[k] for k in state_dict}
    model.load_state_dict(pretrain_dict)

model.to(device)
model.eval()

print(model.tokenizer.conv_layers[2][0])
target_layers = [model.tokenizer.conv_layers[2][0]]

aug_list = [('GaussianBlur', 0.3), ('AddGaussianNoise', 0.3), ('RandomHorizontalFlip', 0.5)]
'''
Dataset = Dataset3D_MOCO('data/lesion_video', 'data_lesion_0728.xlsx',
                         UD_dir='data/two_stream_data_old/UD_clip/',
                         E_dir='data/two_stream_data_old/E_img/', doppler=True, elastography=True,
                         sample_rate=0.25,
                         time_steps=12, number_classes=1, dataset_split=['train'],
                         augmentation=aug_list, UD_reconstruct=False, E_reconstruct=False,
                         resize=(224, 224))
'''
Dataset = Dataset3D_MOCO('data/lesion_video', 'data_lesion_0728.xlsx',
                         UD_dir='data/two_stream_data_old/UD_clip/',
                         E_dir='data/two_stream_data_old/E_img/', doppler=True, elastography=True,
                         sample_rate=0.25,
                         time_steps=12, number_classes=1, dataset_split=['test'],
                         UD_reconstruct=False, E_reconstruct=False, resize=(224, 224), E_top=3)
dataloader = DataLoader(Dataset, batch_size=1, shuffle=True, num_workers=0)

# for i, (data_q, data_k, target, _, _) in enumerate(dataloader):
# data, e_img = data_q[0], data_q[1]
for i, (data, e_img, target, name, stime) in enumerate(dataloader):
    data_numpy = data.permute(0, 4, 2, 3, 1).contiguous().numpy()[0]
    e_numpy = e_img.numpy()
    '''cv2.imshow('My Image', data_numpy[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    for j in range(data_numpy.shape[0]):
        cv2.imwrite(save_dir + '/gradcam/ori_pic/' + str(i) + '_' + str(j) + '_ori.jpg', (data_numpy[j] * 255.).astype('int32'))
    data_gpu, e_gpu = data.to(device), e_img.to(device)
    inp = (data_gpu, e_gpu)

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods["gradcam++"]
    pred = model(inp)
    if MoCo:
        pred, q = pred
    pred_ge = torch.argmax(pred, dim=1).detach().cpu().numpy()
    # pred_ge = torch.ge(pred, 0.3).cpu().numpy()

    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=True) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 1

        grayscale_cam = cam(input_tensor=(inp),
                            target_category=None,
                            aug_smooth='store_true',
                            eigen_smooth='store_true')

        # Here grayscale_cam has only one image in the batch
        # print('grayscale_cam', grayscale_cam.shape)
        grayscale_cam = grayscale_cam[0]

        target, pred_ge = target.item(), pred_ge.item()
        for j in range(data_numpy.shape[0]):
            cam_image = show_cam_on_image(cv2.cvtColor(data_numpy[j], cv2.COLOR_BGR2RGB), grayscale_cam[:, :, j], use_rgb=True)
            # print('cam_image', cam_image.shape)
            # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            # print(target, pred)
            if target == 1 and pred_ge == 1:
                cv2.imwrite(save_dir + '/gradcam/TP/' + str(i) + '_' + str(j) + '_cam.jpg', cam_image)
            elif target == 1:
                cv2.imwrite(save_dir + '/gradcam/FN/' + str(i) + '_' + str(j) + '_cam.jpg', cam_image)
            elif pred_ge == 1:
                cv2.imwrite(save_dir + '/gradcam/FP/' + str(i) + '_' + str(j) + '_cam.jpg', cam_image)
            else:
                cv2.imwrite(save_dir + '/gradcam/TN/' + str(i) + '_' + str(j) + '_cam.jpg', cam_image)


    # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
    # gb = gb_model((data_gpu, e_gpu), target_category=None)

    # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    # cam_gb = deprocess_image(cam_mask * gb)
    # gb = deprocess_image(gb)
    # cv2.imwrite('savemodel1005_noisy/gradcam/' + str(i) + '_gb.jpg', gb)
    # cv2.imwrite('savemodel1005_noisy/gradcam/' + str(i) + '_cam_gb.jpg', cam_gb)
