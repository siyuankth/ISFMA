## To quantitative analyze the morphology of different sutures
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
import math
import Suture_analysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Initialize the error
def Find_point_in_line(POINT,LINE):
    temp1 = 10000
    for i in range(np.shape(LINE)[0]):
        temp2 = distance(POINT, LINE[i, :])
        if temp2 < temp1:
            temp1 = temp2
            POINTB = LINE[i, :]
            POINTB_ID = i
    return temp1, POINTB_ID
def distance(node1, node2):
    distance = math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2 + (node1[2] - node2[2]) ** 2)
    return distance
# Here first using the mean suture created by PCA analysis as an example to compile the code
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
new_data = data.values[:,0:2400]
## Replace the data with slided semilandmark for PCA
slided_data = pd.read_csv('X:/Siyuanch/Project2/COMMENTS_FROM_JOURNAL_ANATOMY/Python_code/new_data_after_sliding.csv')
slided_data = slided_data.to_numpy()
new_data = slided_data
## Data standardization
standardizer = StandardScaler()# Same with scale
data_s = standardizer.fit_transform(new_data)
PCs_nr = 69
pca = PCA(n_components = PCs_nr)  #n_components = PCs_nr
X = pca.fit_transform(data_s)
EVR = pca.explained_variance_ratio_

## The first 30 PCs can account for over 95% variance
num_PC = 30
Ratio_sum = sum(EVR[:num_PC]) * 100
print('The ratio sum of EVR of %f number of the principle components is %.2f%% '
      %(num_PC,Ratio_sum))

Y = pca.transform(data_s)
#######################################################################################
# X_mean value
X_mean = np.mean(Y,axis=0)

# What is the mean value looks like
PCA_mean = pca.inverse_transform(X_mean)
PCA_mean_rebuild = standardizer.inverse_transform(PCA_mean)
Suture_analysis.ALL_Suture(PCA_mean_rebuild,'mean')
Visualization.single_line(PCA_mean_rebuild,0,800,'suture morphology')

print('done')