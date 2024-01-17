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
##
## Data standardization
# from sklearn.preprocessing import scale
# data_s = scale(data)
# data_s = data
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()# Same with scale
data_s = standardizer.fit_transform(new_data)

from sklearn.decomposition import PCA
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
## MEASUREMENT OF AREA
Suture_Area_total = np.zeros((69, 8))
Fontanel_area = np.zeros((69,6))

for i in range(0,69):
    current_data = new_data[i,:]
    # Metopic suture
    Suture_Area_total[i,0] = Suture_analysis.Area_Metopic(current_data,str(i),0.3)
    # Coronal suture
    Suture_Area_total[i,1],Suture_Area_total[i,2] = Suture_analysis.Area_Coronal(current_data,str(i),0.3)
    # Sagittal suture
    Suture_Area_total[i,3] = Suture_analysis.Area_Sagittal(current_data,str(i),0.3)
    # Squamosal suture
    Suture_Area_total[i,4], Suture_Area_total[i,5] = Suture_analysis.Area_Squamosal(current_data,str(i),0.3)
    # Lambdoidal suture
    Suture_Area_total[i,6], Suture_Area_total[i,7] = Suture_analysis.Area_Lambdoidal(current_data,str(i),0.3)
    # anterior fontanel
    Fontanel_area[i,0] = Suture_analysis.Anterior_fontanel_area(current_data,str(i),0.3)
    # posterior fontanel
    Fontanel_area[i,1] = Suture_analysis.Posterior_fontanel_area(current_data,str(i),0.3)
    # Sphenoid fontanel
    Fontanel_area[i,2], Fontanel_area[i,3] = Suture_analysis.Area_Sphenoid(current_data,str(i),0.3)
    # Mastoid fontanel
    Fontanel_area[i,4], Fontanel_area[i,5] = Suture_analysis.Area_Mastoid(current_data,str(i),0.3)

## MEASUREMENT OF SUTURES
# Metopic _suture
Metopic_Suture_size = np.zeros((69,3))
for i in range(0,69):
    current_data = new_data[i,:]
    Metopic_Suture_size[i,0],Metopic_Suture_size[i,1] ,Metopic_Suture_size[i,2]= Suture_analysis.Metopic_Suture(
                                                        new_data[i,:],str(i))

# Sagittal _suture
Sagittal_Suture_size = np.zeros((69,3))
for i in range(0,69):
    current_data = new_data[i,:]
    Sagittal_Suture_size[i,0],Sagittal_Suture_size[i,1],Sagittal_Suture_size[i,2] = Suture_analysis.Sagittal_suture(
                                                          new_data[i,:],str(i))


# Coronal _Suture
Coronal_Suture_size = np.zeros((69,5))
for i in range(0,69):
    current_data = new_data[i,:]
    Coronal_Suture_size[i,0], Coronal_Suture_size[i,1],Coronal_Suture_size[i,2],Coronal_Suture_size[i,3],Coronal_Suture_size[i,4]= Suture_analysis.Coronal_Suture(new_data[i,:],str(i))

# Squamosal_Suture
Squamosal_Suture_size = np.zeros((69,5))
for i in range(0,69):
    current_data = new_data[i,:]
    Squamosal_Suture_size[i,0], Squamosal_Suture_size[i,1],\
    Squamosal_Suture_size[i,2],Squamosal_Suture_size[i,3],Squamosal_Suture_size[i,4]= Suture_analysis.Squamosal_Suture(new_data[i,:],str(i))

# Lambdoid suture
Lambdoid_Suture_size = np.zeros((69,3))
for i in range(0,69):
    current_data = new_data[i,:]
    Lambdoid_Suture_size[i,0],Lambdoid_Suture_size[i,1],Lambdoid_Suture_size[i,2] = Suture_analysis.Lambdoid_Suture(
                                                        new_data[i,:],str(i))


print('over')
