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
import statsmodels.api as sm
import Suture_analysis
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
Suture = data.values[:,[2416,2417,2419,2420,2422,2423,2425,2426,2428,2429,2418,2421,2424,2427,2430]]
data = data.values[:,0:2400]

ID_nr = len(data)

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
# Initialize the error
Error_LSD_train = []
Error_LMD_train = []
Error_LSD_test = []
Error_LMD_test = []

E_LSD_train_List = []
E_LMD_train_List = []
E_LSD_test_List = []
E_LMD_test_List = []
Suture_size =  np.zeros((ID_nr,19))
Average_real_suture = np.zeros((ID_nr,15))
Average_predict_suture = np.zeros((ID_nr,15))
LMD_parts = np.zeros((ID_nr,10))
# For debug
# i =
for i in range(0,69):
    [train, test] = relevant.one_fold(data_s, i)
    [train_area, test_area] = relevant.one_fold(area, i)
    [train_SI, test_SI] = relevant.one_fold(SI, i)
    [train_length, test_length] = relevant.one_fold(length, i)
    [train_width, test_width] = relevant.one_fold(width, i)
    [train_circumference,test_circumference] = relevant.one_fold(circumference, i)
    [train_suture, test_suture] = relevant.one_fold(Suture,i)
    ## PCA modelling
    # With one certain PCs_nr
    PCs_nr = 30
    pca = PCA(n_components = PCs_nr)
    X = pca.fit_transform(train)
   # Back the real data from the PC value in the format of standardization
    Y_train = pca.transform(train)
    PC_FE = []
# Predict the suture size of test data
    for ii in range(PCs_nr):
        PC1 = Y_train[:, ii]
        x = np.vstack((train_length, train_width,
                       train_circumference, train_area, train_SI))
        x = x.transpose()
        y = PC1
        xx = sm.add_constant(x)  # 添加常数项
        model = sm.OLS(y, xx).fit()
        # test only have one data
        FE_size = np.array([1,test_length,test_width,test_circumference,test_area, test_SI])
        FE_size = np.transpose(FE_size)
        PCvalue = model.predict(FE_size)
        PC_FE.append(PCvalue)
    PC_FE = np.array(PC_FE)
    PC_FE = np.transpose(PC_FE)
    PCA_FE = pca.inverse_transform(PC_FE)
    PCA_FE_rebuild = standardizer.inverse_transform(PCA_FE)
    PCA_FE_real = standardizer.inverse_transform(test)
    Visualization.line_([PCA_FE_rebuild,PCA_FE_real ],0,800, f"i = {i} as test")
    PCA_FE_rebuild = np.transpose(PCA_FE_rebuild)
    PCA_FE_real = PCA_FE_real.reshape(2400,1)
    LMD_parts[i,9] = Evaluation.LMD(PCA_FE_real, PCA_FE_rebuild)
    #####       1        #metopic suture
    ID_metopic = Suture_analysis.Parts_metopic(PCA_FE_real,'i')
    # metopic suture
    MS_real, MS_pred = Suture_analysis.LMD_metopic(PCA_FE_real,PCA_FE_rebuild, ID_metopic)
    LMD_MS = Evaluation.LMD(MS_real,MS_pred)
    LMD_parts[i,0] = LMD_MS[0][0]
    j = i
    Suture_size[j, 0], Suture_size[j, 1], Suture_size[j, 14] = Suture_analysis.Metopic_Suture_pred(PCA_FE_rebuild, ID_metopic)
    ####### # 2 AF
    AF_real, AF_pred = Suture_analysis.Parts_AF(PCA_FE_real,PCA_FE_rebuild,'i')
    AF_real = AF_real.flatten()
    AF_pred = AF_pred.flatten()
    LMD_AF = Evaluation.LMD(AF_real, AF_pred)
    LMD_parts[i, 1] = LMD_AF
    ####### 3Sagittal suture
    SaS_real, SaS_pred,Suture_size[j, 2], Suture_size[j, 3], Suture_size[j, 15]  = Suture_analysis.Parts_SaS(PCA_FE_real,PCA_FE_rebuild,'i')
    SaS_real= SaS_real.flatten()
    SaS_pred = SaS_pred.flatten()
    LMD_SaS = Evaluation.LMD(SaS_real, SaS_pred)
    LMD_parts[i, 2] = LMD_SaS
    ##########4 PF
    PF_real, PF_pred = Suture_analysis.Parts_PF(PCA_FE_real, PCA_FE_rebuild, 'i')

    PF_real = PF_real.flatten()
    PF_pred = PF_pred.flatten()
    LMD_PF = Evaluation.LMD(PF_real, PF_pred)
    LMD_parts[i, 3] = LMD_PF
    ####  5Coronal suture
    CS_real, CS_pred,Suture_size[j, 4], Suture_size[j, 5], Suture_size[j, 6], Suture_size[j, 7], Suture_size[j, 16] \
        = Suture_analysis.Parts_CS(PCA_FE_real, PCA_FE_rebuild, 'i')
    CS_real = CS_real.flatten()
    CS_pred = CS_pred.flatten()
    LMD_CS = Evaluation.LMD(CS_real, CS_pred)
    LMD_parts[i, 4] = LMD_CS
    # #### 6Squamosal suture
    SqS_real, SqS_pred,Suture_size[j, 8], Suture_size[j, 9], Suture_size[j, 10], Suture_size[j, 11], Suture_size[j, 17]  = \
        Suture_analysis.Parts_SqS(PCA_FE_real, PCA_FE_rebuild, 'i')
    SqS_real = SqS_real.flatten()
    SqS_pred = SqS_pred.flatten()
    LMD_SqS = Evaluation.LMD(SqS_real, SqS_pred)
    LMD_parts[i, 5] = LMD_SqS
    ### 7Lambdoidal suture
    LS_real, LS_pred,Suture_size[j, 12], Suture_size[j, 13], Suture_size[j, 18] = Suture_analysis.Parts_LS(PCA_FE_real, PCA_FE_rebuild, 'i')
    LS_real = LS_real.flatten()
    LS_pred = LS_pred.flatten()
    LMD_LS = Evaluation.LMD(LS_real, LS_pred)
    LMD_parts[i, 6] = LMD_LS
    ### 8Sphenoid fontanel
    SF_real, SF_pred = Suture_analysis.Parts_SF(PCA_FE_real, PCA_FE_rebuild, 'i')
    SF_real = SF_real.flatten()
    SF_pred = SF_pred.flatten()
    LMD_SF = Evaluation.LMD(SF_real, SF_pred)
    LMD_parts[i, 7] = LMD_SF
    ### 9Mastoid fontanel
    MF_real, MF_pred = Suture_analysis.Parts_MF(PCA_FE_real, PCA_FE_rebuild, 'i')
    MF_real = MF_real.flatten()
    MF_pred = MF_pred.flatten()
    LMD_MF = Evaluation.LMD(MF_real, MF_pred)
    LMD_parts[i, 8] = LMD_MF

    Average_real_suture[i,:] = test_suture
    Average_predict_suture[i,0:4] = Suture_size[j, 0:4]
    Average_predict_suture[i,4] = np.average([Suture_size[j,4],Suture_size[j,6]])
    Average_predict_suture[i,5] = np.average([Suture_size[j,5],Suture_size[j,7]])
    Average_predict_suture[i,6] = np.average([Suture_size[j,8],Suture_size[j,10]])
    Average_predict_suture[i,7] = np.average([Suture_size[j,9],Suture_size[j,11]])
    Average_predict_suture[i,8] = Suture_size[j,12]
    Average_predict_suture[i,9] = Suture_size[j,13]
    Average_predict_suture[i,10:15] = Suture_size[j,14:19]


plt.show()
# Suture = np.delete(Suture,[31,44])
Real_std = np.std(Suture,axis=0)
Real_mean = np.mean(Suture, axis=0)
# Average_predict_suture = np.delete(Average_predict_suture,[31,44],axis = 0)  #[34,47,49,64]
Average_predict_suture = np.delete(Average_predict_suture,[34,47,49],axis = 0)  #[34,47,49,64]
Predict_mean = np.mean(Average_predict_suture, axis = 0)


plt.rcParams['font.size'] = 26
plt.rcParams['font.family'] = 'Times New Roman'
# Plot suture length column figure
N = 5
Mean_length = Real_mean[[0,2,4,6,8]]
Std_length = Real_std[[0,2,4,6,8]]
Predict_length = Predict_mean[[0,2,4,6,8]]
ind = np.arange(N)
width = 0.35
fig, ax = plt.subplots(figsize=(10,8))

rects1 = ax.bar(ind - width/2, Mean_length, width, yerr=Std_length,
                label='CT scans measurement', capsize=4)
rects2 = ax.bar(ind + width/2, Predict_length, width,
                label='Statistical model prediction', capsize=4)
ax.legend()
ax.set_xticks(ind)
ax.set_xticklabels(('MS', 'SaS', 'CS', 'SqS', 'LS'))
ax.set_xlabel('Cranial sutures',fontsize=30)
ax.set_ylabel('Length (mm)',fontsize=30)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax.tick_params(labelsize=30)

plt.savefig('length.jpg', dpi=600, bbox_inches='tight')
plt.show()

# Plot suture width column figure
plt.rcParams['font.size'] = 26
plt.rcParams['font.family'] = 'Times New Roman'
N = 5
Mean_width = Real_mean[[1,3,5,7,9]]
Std_width = Real_std[[1,3,5,7,9]]
Predict_width = Predict_mean[[1,3,5,7,9]]
ind = np.arange(N)
width = 0.35
fig, ax = plt.subplots(figsize=(10,8))
rects1 = ax.bar(ind - width/2, Mean_width, width, yerr=Std_width,
                label='CT scans measurement', capsize=4)
rects2 = ax.bar(ind + width / 2, Predict_width, width,
                label='Statistical model prediction', capsize=4)
ax.legend()
ax.set_xticks(ind)
ax.set_xticklabels(('MS', 'SaS', 'CS', 'SqS', 'LS'))
ax.set_xlabel('Cranial sutures',fontsize=30)
ax.set_ylabel('Width (mm)',fontsize=30)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax.tick_params(labelsize=30)
plt.savefig('width.jpg', dpi=600, bbox_inches='tight')
plt.show()
# ax.set_title('Scores by group and gender')
import matplotlib.ticker as ticker
# Plot suture SI column figure
N = 5
Mean_SI = Real_mean[[10,11,12,13,14]]
Std_SI = Real_std[[10,11,12,13,14]]
Predict_SI = Predict_mean[[10,11,12,13,14]]
ind = np.arange(N)
width = 0.35
fig, ax = plt.subplots(figsize=(10,8))
rects1 = ax.bar(ind - width/2, Mean_SI, width, yerr=Std_SI,
                label='Measured from the original CT scans', capsize=4)
rects2 = ax.bar(ind + width/2, Predict_SI, width,
                label='Predicted by the statistical model', capsize=4)
ax.set_ylim(0.8, 1.4)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.set_xticks(ind)
ax.set_xticklabels(('MSI', 'SaSI', 'CSI', 'SqSI', 'LSI'))
ax.set_xlabel('Cranial sutures')
ax.set_ylabel('SI')
plt.show()
plt.show()
#Plot



print(data)



