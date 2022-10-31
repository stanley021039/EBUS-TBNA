# EBUS-TBNA
## Models
The codes of models are in folder 'models'.
### 2D
 - VGG-19
 - Resnet-50
 - EBUSNet

### 3D
 - CNN-LSTM
 - CNN-LSTM-TS
 - R3D
 - R3D-TS
 - ViT-3D
 - TransEBUS
 - TransEBUS-TS
 - TransEBUS-TS-MoCo
 - TransEBUS-3S-MoCo
 - TransEBUS-TS-MoCo-ClsT

## Training & testing models
1. These codes are the default codes for training. If you want to re-train the model, just enter the appropriate command below.  
If you want to reproduce the testing result, please change the value of the parameter 'is_training' into '0' as follow:  
![image](https://github.com/stanley021039/EBUS-TBNA/blob/main/%E6%93%B7%E5%8F%96.PNG)
2. If the folder does not exist, create the folder tree as follows: 
>savemodelxxxxxx
>>confusion_matrix  
>>Roc_curve  
>>model_state_dict

|  Model   | Command |
|  :----  | :----  |
| VGG-19  | python3 train_VGG19.py |
| Resnet-50  | python3 train_Res50.py |
| EBUSNet  | python3 train_EBUSNet.py |
| CNN-LSTM  | python3 train_CNNLSTM.py |
| CNN-LSTM-TS  | python3 train_CNNLSTM_TS.py |
| R3D  | python3 train_R3D.py |
| R3D-TS  | python3 train_R3D_TS.py |
| ViT-3D  | python3 train_ViT_3D.py |
| TransEBUS  | python3 train_TransEBUS.py |
| TransEBUS-TS  | python3 train_TransEBUS_TS.py |
| TransEBUS-TS-MoCo  | python3 train_TransEBUS_TS_MoCo.py |
| TransEBUS-3S-MoCo  | python3 train_TransEBUS_3S_MoCo.py |
| TransEBUS-TS-MoCo-ClsT  | python3 train_TransEBUS_TS_MoCo_ClsT.py |
