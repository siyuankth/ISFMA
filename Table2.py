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

X = pca.fit_transform(data_s)

import statsmodels.api as sm
# Compare the model prediction and true for 101819
# Second figure
Y = pca.transform(data_s)


# Regression between PC values and shape parameters


# Define a function

# # model predict
# Suture_size_PC = np.transpose(Suture_size_PC)
import statsmodels.api as sm
FE_size = np.array([1,125.8,102,372.1])
FE_size = np.transpose(FE_size)
PC_FE = []
Pvalue_PCs = []
R2_PCs = []
x = np.vstack((length, width, circumference))
x = x.transpose()
xx = sm.add_constant(x)
x_length = sm.add_constant(length)
x_width = sm.add_constant(width)
x_circumference = sm.add_constant(circumference)# 添加常数项
x_area = sm.add_constant(area)
x_SI = sm.add_constant(SI)

# length width
x = np.vstack((length, width))
x = x.transpose()
l_w = sm.add_constant(x)
x = np.vstack((length, circumference))
x = x.transpose()
l_c = sm.add_constant(x)
x = np.vstack((width, circumference))
x = x.transpose()
w_c = sm.add_constant(x)
x = np.vstack((length, width,area))
x = x.transpose()
l_w_s = sm.add_constant(x)
x = np.vstack((length, circumference,area))
x = x.transpose()
l_c_s = sm.add_constant(x)
x = np.vstack((width, circumference,area))
x = x.transpose()
w_c_s = sm.add_constant(x)

x = np.vstack((length, width, circumference))
x = x.transpose()
xx = sm.add_constant(x)

x = np.vstack((length, width, circumference,area))
x = x.transpose()
xx_area = sm.add_constant(x)

x = np.vstack((length, width, circumference,SI))
x = x.transpose()
xx_SI = sm.add_constant(x)

x = np.vstack((length, width, circumference,area,SI))
x = x.transpose()
xx_all = sm.add_constant(x)

for i in range(PCs_nr):
    PC1 = Y[:, i]
    y = PC1

    model_length = sm.OLS(y, x_length).fit()
    Pvalue_length = model_length.f_pvalue
    R2_length = model_length.rsquared

    model_width = sm.OLS(y, x_width).fit()
    Pvalue_width = model_width.f_pvalue
    R2_width = model_width.rsquared

    model_cir = sm.OLS(y, x_circumference).fit()
    Pvalue_cir = model_cir.f_pvalue
    R2_cir = model_cir.rsquared

    model_area = sm.OLS(y, x_area).fit()
    Pvalue_area = model_area.f_pvalue
    R2_area = model_area.rsquared

    model_SI = sm.OLS(y, x_SI).fit()
    Pvalue_SI = model_SI.f_pvalue
    R2_SI = model_SI.rsquared

    model_all = sm.OLS(y, xx_all).fit()
    # PCvalue = model.predict(FE_size)
    Pvalue_all = model_all.f_pvalue
    R2_size_all = model_all.rsquared

    Pvalue_PCs.append([Pvalue_length,Pvalue_width,Pvalue_cir,Pvalue_area,Pvalue_SI,Pvalue_all ])
    R2_PCs.append([R2_length,R2_width,R2_cir,R2_area,R2_SI,R2_size_all])

Pvalue_PCs = np.array(Pvalue_PCs)
R2_PCs = np.array(R2_PCs)

print('Done')