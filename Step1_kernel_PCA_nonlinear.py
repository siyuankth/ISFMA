# Read the csv file
import csv

import pandas as pd
from tensorflow.keras import losses
import Evaluation
import relevant
import numpy as np
import matplotlib.pyplot as plt
import Visualization
from tensorflow.keras import backend as K
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
df = pd.read_csv('X:/Siyuanch/Project2/DATA.csv')
# Relevance analysis (features engineering process)
data = relevant.analysis(df,'Record_ID', False)  # discard the column of Record_ID
# Sliding data
## Replace the data with slided semilandmark for PCA
slided_data = pd.read_csv('X:/Siyuanch/Project2/COMMENTS_FROM_JOURNAL_ANATOMY/Python_code/new_data_after_sliding.csv')
slided_data = slided_data.to_numpy()
data = slided_data
## Reorder data to new_data
X = np.zeros((69,800))
Y = np.zeros((69,800))
Z = np.zeros((69,800))
row_data = data
for iii in range(800):
    X[:,iii] = row_data[:, iii*3 + 0]
    Y[:, iii] = row_data[:, iii * 3 + 1]
    Z[:,iii] = row_data[:,iii*3 + 2]
X1_new = X[:,0:100]
X2_new = X[:,100:200]
X3_new = X[:,200:400]
X4_new = X[:,400:600]
X5_new = X[:,600:650]
X6_new = X[:,650:700]
X7_new = X[:,700:800]
X_new = np.hstack((X2_new, X5_new, X7_new,np.fliplr(X6_new), np.fliplr(X1_new),X4_new,X3_new))
Y1_new = Y[:,0:100]
Y2_new = Y[:,100:200]
Y3_new = Y[:,200:400]
Y4_new = Y[:,400:600]
Y5_new = Y[:,600:650]
Y6_new = Y[:,650:700]
Y7_new = Y[:,700:800]
Y_new = np.hstack((Y2_new, Y5_new, Y7_new, np.fliplr(Y6_new), np.fliplr(Y1_new),Y4_new,Y3_new))
Z1_new = Z[:,0:100]
Z2_new = Z[:,100:200]
Z3_new = Z[:,200:400]
Z4_new = Z[:,400:600]
Z5_new = Z[:,600:650]
Z6_new = Z[:,650:700]
Z7_new = Z[:,700:800]
Z_new = np.hstack((Z2_new, Z5_new, Z7_new, np.fliplr(Z6_new), np.fliplr(Z1_new), Z4_new,Z3_new))
Re_data = np.zeros((69,2400))
for iii in range(800):
    Re_data[:, iii*3 + 0] = X_new[:,iii]
    Re_data[:, iii * 3 + 1] = Y_new[:, iii]
    Re_data[:, iii * 3 + 2] = Z_new[:, iii]
## Random the data
# per = np.random.permutation(data.shape[0]) #打乱后的行号
per = [42,33,49,9,11,1,41,56,37,36,51,48,35,66,46,65,7,6,2,12,24,68,59,22,
       23,13,45,26,52,63,4,53,21,58,17,57,31,27,25,50,43,14,54,61,60,28,0,
       64,30,18,38,5,40,62,29,32,34,3,67,8,10,19,39,47,15,20,16,44,55]
new_data = relevant.random(Re_data,per)		#获取打乱后的训练数据
## Data standardization
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()# Same with scale
data_s = standardizer.fit_transform(new_data)
##### KernelPCA
from sklearn.decomposition import KernelPCA
Error_LSD_train = []
Error_LMD_train = []
Error_LSD_test = []
Error_LMD_test = []
E_LSD_train_List = []
E_LMD_train_List = []
E_LSD_test_List = []
E_LMD_test_List = []
for i in range(10):
    [train, test] = relevant.ten_fold(data_s, i)
    ## PCA modelling
    # With one certain PCs_nr
    PCs_nr = 30
    # kpca = KernelPCA(n_components=PCs_nr, kernel='linear',fit_inverse_transform=True)
  # kpca = KernelPCA(n_components=PCs_nr, kernel='cosine' , fit_inverse_transform=True)
  #   kpca = KernelPCA(n_components=PCs_nr, kernel='sigmoid' , fit_inverse_transform=True)
    kpca = KernelPCA(n_components=PCs_nr, kernel='rbf' , fit_inverse_transform=True)
    # kpca = KernelPCA(n_components=PCs_nr, kernel='laplacian' , fit_inverse_transform=True)
    X = kpca.fit(train)
    ## evaluate in train data
    # Back the real data from the PC value in the format of standardization
    Y_train = kpca.transform(train)
    # difff = X - Y_train
    train_pred = kpca.inverse_transform(Y_train)
    # train_pred = kpca.inverse_transform(Y_train)
    train_diff = train - train_pred
    # LOSS
    real_train = standardizer.inverse_transform(train)
    rebuild_train = standardizer.inverse_transform(train_pred)
    LSD_train = Evaluation.LSD(real_train, rebuild_train)
    LMD_train = Evaluation.LMD(real_train, rebuild_train)
    ## Evaluate in test data
    Y_test = kpca.transform(test)
    test_pred = kpca.inverse_transform(Y_test)
    test_diff = test - test_pred
    # LOSS
    LOSS_test = losses.mse(K.reshape(test, (-1,)), K.reshape(test_pred, (-1,))).numpy()
    # LSD and LMD
    real_test = standardizer.inverse_transform(test)
    rebuild_test = standardizer.inverse_transform(test_pred)
    LSD_test = Evaluation.LSD(real_test, rebuild_test)
    LMD_test = Evaluation.LMD(real_test, rebuild_test)
    ## Get the mean of error
    Mean_LSD_train = np.mean(LSD_train)
    Mean_LMD_train = np.mean(LMD_train)
    Mean_LSD_test = np.mean(LSD_test)
    Mean_LMD_test = np.mean(LMD_test)
    # Save the mean error
    Error_LSD_train = np.append(Error_LSD_train, Mean_LSD_train)
    Error_LMD_train = np.append(Error_LMD_train, Mean_LMD_train)
    Error_LSD_test = np.append(Error_LSD_test, Mean_LSD_test)
    Error_LMD_test = np.append(Error_LMD_test, Mean_LMD_test)

print('mean LMD train error = %f' % np.mean(Error_LMD_train))
print('mean LMD test error = %f' % np.mean(Error_LMD_test))

print('done')




