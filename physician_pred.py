import pandas as pd
import numpy as np
from utils import Roc_curve
from utils import plot_confusion_matrix_and_scores

exc = pd.read_excel('data_physician.xls', sheet_name='測試資料(50筆)')
hist = np.zeros((2, 2))
threshold = 4
y_true = []
y_pred = []
for (i, LN) in exc.iterrows():
    cls = LN[4]
    shape = LN[7]
    size = LN[9]
    margin = LN[10]
    heterogenous = LN[11]
    CHS = LN[13]
    CNS = LN[14]
    matting = LN[15]
    dupplux = LN[18]
    elastography = LN[20]
    if cls == 'benign':
        target = 0
    else:
        target = 1
    mal_score = 0
    if shape == 1:
        mal_score += 1
    if size == 1:
        mal_score += 1
    if margin == 1:
        mal_score += 1
    if heterogenous == 1:
        mal_score += 1
    if CHS == 1:
        mal_score += 1
    if CNS == 1:
        mal_score += 1
    if matting == 1:
        mal_score += 1
    if dupplux == 1:
        mal_score += 1
    if elastography == 1:
        mal_score += 1
    if mal_score >= threshold:
        pred = 1
    else:
        pred = 0
    y_true.append(target)
    y_pred.append(mal_score / 9)
    hist[target, pred] += 1
optimal_th, optimal_hist, roc_auc = Roc_curve(np.array(y_true), np.array(y_pred), re_auc=True, show=True)
plot_confusion_matrix_and_scores(optimal_hist)
print(roc_auc)
