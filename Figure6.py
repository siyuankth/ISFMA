# Read the csv file
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
# print(data.values[0,0])
# print(int(data.values.shape[1]/3) )
## ADD
length = data.values[:,2400]
width = data.values[:,2401]
circumference = data.values[:,2402]
data = data.values[:,0:2400]
new_data = data
## Replace the data with slided semilandmark for PCA
slided_data = pd.read_csv('X:/Siyuanch/Project2/COMMENTS_FROM_JOURNAL_ANATOMY/Python_code/new_data_after_sliding.csv')
slided_data = slided_data.to_numpy()
new_data = slided_data
## Data standardization
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()# Same with scale
data_s = standardizer.fit_transform(new_data)
from sklearn.decomposition import PCA
PCs_nr = 30
pca = PCA(n_components = PCs_nr)  #n_components = PCs_nr
X = pca.fit_transform(data_s)
EVR = pca.explained_variance_ratio_
fig_y = np.cumsum(EVR)
fig_x = np.array(list(range(1,PCs_nr+1)))
Y = pca.transform(data_s)
pred = pca.inverse_transform(Y)
# LSD and LMD
real = standardizer.inverse_transform(data_s)
rebuild = standardizer.inverse_transform(pred)
diff = real - rebuild
LMD_ = Evaluation.LMD(real, rebuild)
# Visualization.line_([Real, Rebuild], 0, 800, 'the 1st subject')
## Get the mean of error
# Mean_LSD = np.mean(LSD_)
Mean_LMD = np.mean(LMD_)
for i in range(10):
    Visualization.line_([real, rebuild], i, 800, str(i)+'th suture by PCA')
    plt.savefig("X:\Siyuanch\Project2\COMMENTS_FROM_JOURNAL_ANATOMY\Python_code\Figure6/"+str(i+1)+"PC.eps", format='eps')
print('done!')