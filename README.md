# EBUS-TBNA
## 事前準備
請先在terminal進入資料夾後，啟動虛擬環境  
啟動虛擬環境指令: source EBUS_env/bin/activate

## Models
模型code在資料夾'models'，包含：
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
訓練用的code在最外層資料夾，但個別模型的資料夾內亦有一個備份，儲存的模型資料夾內容為
>savemodelYYMMDD_xxx  
>>confusion_matrix (每個epoch的混淆矩陣，包含以clip為單位(train_v_xx.jpg)、以病灶為單位(train_l_xx.jpg))  
>>model_state_dict (對應epoch的權重)  
>>Roc_curve (每個epoch的Roc curve，包含以clip為單位(train_v_xx.jpg)、以病灶為單位(train_l_xx.jpg))  
1. 重現測試結果指令範例：  
python3 train_TransEBUS_TS_MoCo.py
2. 重新訓練模型指令範例：  
python3 train_TransEBUS_TS_MoCo.py --train 1

以下為各模型對應的測試指令與默認儲存的資料夾
|  Model   | Command | Savedir |
|  :----  | :----  | :---- |
| VGG-19  | python3 train_VGG19.py | savemodel220620_VGG19 |
| Resnet-50  | python3 train_Res50.py | savemodel220621_Res50 |
| EBUSNet  | python3 train_EBUSNet.py | savemodel220620_EBUSnet |
| CNN-LSTM  | python3 train_CNNLSTM.py | savemodel220627_CNNLSTM |
| CNN-LSTM-TS  | python3 train_CNNLSTM_TS.py | savemodel220628_CNNLSTM_TS |
| R3D  | python3 train_R3D.py | savemodel220602_R3D |
| R3D-TS  | python3 train_R3D_TS.py | savemodel220609_R3D_TS |
| ViT-3D  | python3 train_ViT_3D.py | savemodel220709_ViT_3D |
| TransEBUS  | python3 train_TransEBUS.py | savemodel220921_TransEBUS |
| TransEBUS-TS  | python3 train_TransEBUS_TS.py | savemodel220921_TransEBUS_TS |
| TransEBUS-TS-MoCo  | python3 train_TransEBUS_TS_MoCo.py | savemodel220726_TransEBUS_TS_MoCo |
| TransEBUS-3S-MoCo  | python3 train_TransEBUS_3S_MoCo.py | savemodel221017_TransEBUS_3S_MoCo |
| TransEBUS-TS-MoCo-ClsT  | python3 train_TransEBUS_TS_MoCo_ClsT.py | savemodel221014_TransEBUS_TS_MoCo_clsT |
| Clinical physician  | python3 physician_pred.py | - |

## GradCam 可視化 (TransEBUS-TS-MoCo)
1. 重新生成關注區域熱點圖指令 python3 GradCam.py
2. 儲存資料夾內容如下
>savemodel220726_TransEBUS_TS_MoCo
>>gradcam  
>>>TP (真陽性的熱點圖)  
>>>TN (真陰性的熱點圖)  
>>>FP (偽陽性的熱點圖)  
>>>FN (偽陰性的熱點圖)  
>>>ori_ (對應之灰階影像)

## Results
1.	各模型的比較  
![image](https://github.com/stanley021039/EBUS-TBNA/blob/main/Results/result1.png)
2. 雙流模塊的比較  
![image](https://github.com/stanley021039/EBUS-TBNA/blob/main/Results/result2.png)
3. 對比學習的比較  
![image](https://github.com/stanley021039/EBUS-TBNA/blob/main/Results/result3.png)
4. 額外消融測試  
![image](https://github.com/stanley021039/EBUS-TBNA/blob/main/Results/result4.png)
5. GradCam++ 可視化  
![image](https://github.com/stanley021039/EBUS-TBNA/blob/main/Results/GradCam_T.png)  
![image](https://github.com/stanley021039/EBUS-TBNA/blob/main/Results/GradCam_F.png)  
