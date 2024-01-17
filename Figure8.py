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
area = data.values[:,2403]
SI = data.values[:,2435]
data = data.values[:,0:2400]

new_data = data
## Replace the data with slided semilandmark for PCA
slided_data = pd.read_csv('X:/Siyuanch/Project2/COMMENTS_FROM_JOURNAL_ANATOMY/Python_code/new_data_after_sliding.csv')
slided_data = slided_data.to_numpy()
new_data = slided_data
## Data standardization
# from sklearn.preprocessing import scale
# data_s = scale(data)
# data_s = data
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()# Same with scale
data_s = standardizer.fit_transform(new_data)

from sklearn.decomposition import PCA
# Initialize the error

PCs_nr = 30
pca = PCA(n_components = PCs_nr)  #n_components = PCs_nr
    # pca = PCA(n_components=5)  # 2400 is the number of variables
## floating point precision error
Y = pca.fit_transform(data_s)

import statsmodels.api as sm



X = np.vstack((length,width,circumference,area,SI))
X = X.transpose()
##  multivariable regression analysis

# Define a function

# model predict of 5th
FE_size = np.array([1,113.5,98.4,335.8,1890.7,1.13188])
FE_size = np.transpose(FE_size)
PC_FE = []
for i in range(PCs_nr):
    PC1 = Y[:, i]
    x = np.vstack((length, width, circumference,area,SI))
    x = x.transpose()
    y = PC1
    xx = sm.add_constant(x)
    model = sm.OLS(y, xx).fit()
    PCvalue = model.predict(FE_size)
    PC_FE.append(PCvalue)
PC_FE = np.array(PC_FE)
PC_FE = np.transpose(PC_FE)
PCA_FE = pca.inverse_transform(PC_FE)
PCA_FE_rebuild_5 = standardizer.inverse_transform(PCA_FE)
# Visualization.single_line(PCA_FE_rebuild_5,0,800,'5 th')

# model predict of 25th
FE_size = np.array([1,122.3,102.5,359.4,2158.3,1.15525])
FE_size = np.transpose(FE_size)
PC_FE = []
for i in range(PCs_nr):
    PC1 = Y[:, i]
    x = np.vstack((length, width, circumference,area,SI))
    x = x.transpose()
    y = PC1
    xx = sm.add_constant(x)  # 添加常数项
    model = sm.OLS(y, xx).fit()
    PCvalue = model.predict(FE_size)
    PC_FE.append(PCvalue)
PC_FE = np.array(PC_FE)
PC_FE = np.transpose(PC_FE)
PCA_FE = pca.inverse_transform(PC_FE)
PCA_FE_rebuild_25 = standardizer.inverse_transform(PCA_FE)
# Visualization.single_line(PCA_FE_rebuild_25,0,800,'25 th')

# model predict of 50th
FE_size = np.array([1,125.7,107.8,367.2,2402.7,1.16579])
FE_size = np.transpose(FE_size)
PC_FE = []
for i in range(PCs_nr):
    PC1 = Y[:, i]
    x = np.vstack((length, width, circumference,area,SI))
    x = x.transpose()
    y = PC1
    xx = sm.add_constant(x)  # 添加常数项
    model = sm.OLS(y, xx).fit()
    PCvalue = model.predict(FE_size)
    PC_FE.append(PCvalue)
PC_FE = np.array(PC_FE)
PC_FE = np.transpose(PC_FE)
PCA_FE = pca.inverse_transform(PC_FE)
PCA_FE_rebuild_50 = standardizer.inverse_transform(PCA_FE)
# Visualization.single_line(PCA_FE_rebuild_50,0,800,'50 th')

# model predict of 75th
FE_size = np.array([1,128.6,110.9,376.9,2835.5,1.17664])
FE_size = np.transpose(FE_size)
PC_FE = []
for i in range(PCs_nr):
    PC1 = Y[:, i]
    x = np.vstack((length, width, circumference,area,SI))
    x = x.transpose()
    y = PC1
    xx = sm.add_constant(x)  # 添加常数项
    model = sm.OLS(y, xx).fit()
    PCvalue = model.predict(FE_size)
    PC_FE.append(PCvalue)
PC_FE = np.array(PC_FE)
PC_FE = np.transpose(PC_FE)
PCA_FE = pca.inverse_transform(PC_FE)
PCA_FE_rebuild_75 = standardizer.inverse_transform(PCA_FE)
# Visualization.single_line(PCA_FE_rebuild_75,0,800,'75 th')

# model predict of 95th
FE_size = np.array([1,140.0,116.2,401.7,3982.9,1.21439])
FE_size = np.transpose(FE_size)
PC_FE = []
for i in range(PCs_nr):
    PC1 = Y[:, i]
    x = np.vstack((length, width, circumference,area,SI))
    x = x.transpose()
    y = PC1
    xx = sm.add_constant(x)  # 添加常数项
    model = sm.OLS(y, xx).fit()
    PCvalue = model.predict(FE_size)
    PC_FE.append(PCvalue)
PC_FE = np.array(PC_FE)
PC_FE = np.transpose(PC_FE)
PCA_FE = pca.inverse_transform(PC_FE)
PCA_FE_rebuild_95 = standardizer.inverse_transform(PCA_FE)
# Visualization.single_line(PCA_FE_rebuild_95,0,800,'95 th')

# combination
Visualization.line_([PCA_FE_rebuild_5,PCA_FE_rebuild_25,PCA_FE_rebuild_50,PCA_FE_rebuild_75,PCA_FE_rebuild_95],0,800,'100')

import Suture_analysis
PCA_FE_rebuild_5 = np.transpose(PCA_FE_rebuild_5)
Suture_size = np.zeros((5,19))
Suture_size[0,0],Suture_size[0,1],Suture_size[0,14] = Suture_analysis.Metopic_Suture(PCA_FE_rebuild_5,'5')
Suture_size[0,2],Suture_size[0,3],Suture_size[0,15] = Suture_analysis.Sagittal_Suture(PCA_FE_rebuild_5,'5')
Suture_size[0,4],Suture_size[0,5],Suture_size[0,6],Suture_size[0,7],Suture_size[0,16] = Suture_analysis.Coronal_Suture(PCA_FE_rebuild_5,'5')
Suture_size[0,8],Suture_size[0,9],Suture_size[0,10],Suture_size[0,11],Suture_size[0,17] = Suture_analysis.Squamosal_Suture(PCA_FE_rebuild_5,'5')
Suture_size[0,12],Suture_size[0,13],Suture_size[0,18] = Suture_analysis.Lambdoid_Suture(PCA_FE_rebuild_5,'5')

PCA_FE_rebuild_25 = np.transpose(PCA_FE_rebuild_25)

Suture_size[1,0],Suture_size[1,1],Suture_size[1,14] = Suture_analysis.Metopic_Suture(PCA_FE_rebuild_25,'5')
Suture_size[1,2],Suture_size[1,3] ,Suture_size[1,15]= Suture_analysis.Sagittal_Suture(PCA_FE_rebuild_25,'5')
Suture_size[1,4],Suture_size[1,5],Suture_size[1,6],Suture_size[1,7] ,Suture_size[1,16]= Suture_analysis.Coronal_Suture(PCA_FE_rebuild_25,'5')
Suture_size[1,8],Suture_size[1,9],Suture_size[1,10],Suture_size[1,11],Suture_size[1,17] = Suture_analysis.Squamosal_Suture(PCA_FE_rebuild_25,'5')
Suture_size[1,12],Suture_size[1,13],Suture_size[1,18] = Suture_analysis.Lambdoid_Suture(PCA_FE_rebuild_25,'5')

PCA_FE_rebuild_50 = np.transpose(PCA_FE_rebuild_50)

Suture_size[2,0],Suture_size[2,1],Suture_size[2,14]  = Suture_analysis.Metopic_Suture(PCA_FE_rebuild_50,'5')
Suture_size[2,2],Suture_size[2,3],Suture_size[2,15]  = Suture_analysis.Sagittal_Suture(PCA_FE_rebuild_50,'5')
Suture_size[2,4],Suture_size[2,5],Suture_size[2,6],Suture_size[2,7],Suture_size[2,16]  = Suture_analysis.Coronal_Suture(PCA_FE_rebuild_50,'5')
Suture_size[2,8],Suture_size[2,9],Suture_size[2,10],Suture_size[2,11] ,Suture_size[2,17] = Suture_analysis.Squamosal_Suture(PCA_FE_rebuild_50,'5')
Suture_size[2,12],Suture_size[2,13],Suture_size[2,18]  = Suture_analysis.Lambdoid_Suture(PCA_FE_rebuild_50,'5')

PCA_FE_rebuild_75 = np.transpose(PCA_FE_rebuild_75)

Suture_size[3,0],Suture_size[3,1],Suture_size[3,14] = Suture_analysis.Metopic_Suture(PCA_FE_rebuild_75,'5')
Suture_size[3,2],Suture_size[3,3],Suture_size[3,15] = Suture_analysis.Sagittal_Suture(PCA_FE_rebuild_75,'5')
Suture_size[3,4],Suture_size[3,5],Suture_size[3,6],Suture_size[3,7] ,Suture_size[3,16]= Suture_analysis.Coronal_Suture(PCA_FE_rebuild_75,'5')
Suture_size[3,8],Suture_size[3,9],Suture_size[3,10],Suture_size[3,11] ,Suture_size[3,17]= Suture_analysis.Squamosal_Suture(PCA_FE_rebuild_75,'5')
Suture_size[3,12],Suture_size[3,13],Suture_size[3,18] = Suture_analysis.Lambdoid_Suture(PCA_FE_rebuild_75,'5')

PCA_FE_rebuild_95 = np.transpose(PCA_FE_rebuild_95)

Suture_size[4,0],Suture_size[4,1],Suture_size[4,14] = Suture_analysis.Metopic_Suture(PCA_FE_rebuild_95,'5')
Suture_size[4,2],Suture_size[4,3],Suture_size[4,15] = Suture_analysis.Sagittal_Suture(PCA_FE_rebuild_95,'5')
Suture_size[4,4],Suture_size[4,5],Suture_size[4,6],Suture_size[4,7] ,Suture_size[4,16]= Suture_analysis.Coronal_Suture(PCA_FE_rebuild_95,'5')
Suture_size[4,8],Suture_size[4,9],Suture_size[4,10],Suture_size[4,11],Suture_size[4,17] = Suture_analysis.Squamosal_Suture(PCA_FE_rebuild_95,'5')
Suture_size[4,12],Suture_size[4,13],Suture_size[4,18] = Suture_analysis.Lambdoid_Suture(PCA_FE_rebuild_95,'5')
plt.show()
print('done')