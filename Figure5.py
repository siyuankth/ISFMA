# Read the csv file
import pandas as pd
from tensorflow.keras import losses
import Evaluation
import relevant
import numpy as np
import matplotlib.pyplot as plt
import Visualization
import tensorflow as tf
import Suture_analysis
import morphops as mops
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
df = pd.read_csv('X:/Siyuanch/Project2/DATA.csv')
# Relevance analysis (features engineering process)
data = relevant.analysis(df,'Record_ID', False)  # discard the column of Record_ID
length = data.values[:,2400]
width = data.values[:,2401]
circumference = data.values[:,2402]
data = data.values[:,0:2400]
new_data = data

## Replace the data with slided semilandmark for PCA
slided_data = pd.read_csv('X:/Siyuanch/Project2/COMMENTS_FROM_JOURNAL_ANATOMY/Python_code/new_data_after_sliding.csv')
slided_data = slided_data.to_numpy()
new_data = slided_data


# ## Only GPA
# data_list = []
# for j in range(69):
#     A = new_data[j,:]
#     list_A = [A[i:i + 3] for i in range(0, len(A), 3)]
#     data_list.append(list_A)
# ## GPA
# res = mops.gpa(data_list)
# aligned_data = res['aligned']
# scale_f = res['b']
# GPA_data = np.concatenate(aligned_data[0]/scale_f[0])
# for j in range(1,69):
#     GPA_data = np.vstack((GPA_data,np.concatenate(aligned_data[j]/scale_f[j])))
# # original data before GPA is new_data
# # after GPA is GPA_data
# # let GPA_data becomes new_data
# new_data = GPA_data

from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()# Same with scale
data_s = standardizer.fit_transform(new_data)

from sklearn.decomposition import PCA
PCs_nr = 30
pca = PCA(n_components = PCs_nr)  #n_components = PCs_nr
    # pca = PCA(n_components=5)   #2400 is the number of variables

X = pca.fit_transform(data_s)

    # P = pca.components_
EVR = pca.explained_variance_ratio_
fig_y = np.cumsum(EVR)
fig_x = np.array(list(range(1,PCs_nr+1)))
## evaluate in train data
Y = pca.transform(data_s)
# Back the real data from the PC value in the format of standardization
pred = pca.inverse_transform(Y)

# LSD and LMD
real = standardizer.inverse_transform(data_s)
rebuild = standardizer.inverse_transform(pred)
# LSD_ = Evaluation.LSD(real, rebuild)
diff = real - rebuild
LMD_ = Evaluation.LMD(real, rebuild)
# Visualization.line_([Real, Rebuild], 0, 800, 'the 1st subject')
## Get the mean of error
# Mean_LSD = np.mean(LSD_)
Mean_LMD = np.mean(LMD_)
# X = np.vstack((length,width,circumference))
# X = X.transpose()

# To see the principal components effect
# PCA influence
#######################################################################################
# X_mean value
X_mean = np.mean(Y,axis=0)

# What is the mean value looks like
PCA_mean = pca.inverse_transform(X_mean)
PCA_mean_rebuild = standardizer.inverse_transform(PCA_mean)
# Visualization.single_line(PCA_mean_rebuild,0,800,'mean')
# Visualization.line_([new_data,PCA_mean_rebuild],0,800,'0th suture compared with the mean suture')
## What is the standard value looks like
X_std = np.std(Y,axis = 0)
X_std2 = -1*X_std
PCA_std = pca.inverse_transform(X_std)
PCA_std_rebuild = standardizer.inverse_transform(PCA_std)
# Visualization.single_line(PCA_std_rebuild,0,800,'+std')
# Visualization.line_([new_data,PCA_std_rebuild],0,800,'0th suture compared with the + std suture')
PCA_std2 = pca.inverse_transform(X_std2)
PCA_std2_rebuild = standardizer.inverse_transform(PCA_std2)
# Visualization.single_line(PCA_std2_rebuild,0,800,'-std')
# Visualization.line_([PCA_std_rebuild,PCA_std2_rebuild],0,800,' std suture')
# Visualization.single_line(PCA_std_rebuild,0,800,'+std suture')
# Visualization.single_line(PCA_std2_rebuild,0,800,'-std suture')

################change PC1 value
# X_mean[0] = 26.88
# PCA_mean = pca.inverse_transform(X_mean)
# PC1_upper_rebuild = standardizer.inverse_transform(PCA_mean)
# X_mean[0] = -26.88
# PCA_mean = pca.inverse_transform(X_mean)
# PC1_lower_rebuild = standardizer.inverse_transform(PCA_mean)
# Visualization.line_([PC1_upper_rebuild,PCA_mean_rebuild,PC1_lower_rebuild],0,800,'Influence of the 1st PC values')
# cycle: output the first 7 PCs' influence on morphology
for i in range(0,6):
    # X_mean = np.mean(Y, axis=0)
    PC_larger_std = np.copy(X_mean)
    PC_smaller_std = np.copy(X_mean)
    PC_larger_std[i] = X_std[i]
    PC_smaller_std[i] = X_std2[i]
    PCi_upper = pca.inverse_transform(PC_larger_std)
    PCi_lower = pca.inverse_transform(PC_smaller_std)
    PCi_upper_rebuild = standardizer.inverse_transform(PCi_upper)
    PCi_lower_rebuild = standardizer.inverse_transform(PCi_lower)
    Visualization.line_([PCi_upper_rebuild, PCA_mean_rebuild, PCi_lower_rebuild], 0, 800, 'Influence of the '+str(i+1)+'th PC values')
    plt.savefig("X:\Siyuanch\Project2\COMMENTS_FROM_JOURNAL_ANATOMY\Python_code\Figure5/"+str(i+1)+"PC.eps", format='eps')
plt.show()
print('done')
