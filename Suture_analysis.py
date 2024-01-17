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

def Metopic_Suture(PCA_mean_rebuild,context):
    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)
    # 2. Get the distance between two nodes
    Temp = []
    for i in range(100):
        Temp.append(distance(Suture1[i, :], Suture2[i, :]))
    Temp = np.array(Temp)
    diff = []
    j = []
    for i in range(99):
        diff.append(Temp[i + 1] - Temp[i])
        parameter = 1.5
        if Temp[1] > 5:
            if diff[i] > 0.15 and diff[i] > parameter* diff[i - 1] and diff[i] < 1 and Temp[i] < 12:
                j.append(i)
        else:
            if diff[i] > 0.15 and diff[i] > parameter* diff[i - 1] and diff[i] < 1and Temp[i] < 3:
                # if i>1 and diff[i-1] < 1.5* diff[i-2]:
                j.append(i)
    # marker_X = np.array([Suture1[j[-1], 0], Suture2[j[-1], 0]])
    # marker_Y = np.array([Suture1[j[-1], 1], Suture2[j[-1], 1]])
    # marker_Z = np.array([Suture1[j[-1], 2], Suture2[j[-1], 2]])
    marker_X = np.array([Suture1[j, 0], Suture2[j, 0]])
    marker_Y = np.array([Suture1[j, 1], Suture2[j, 1]])
    marker_Z = np.array([Suture1[j, 2], Suture2[j, 2]])
    diff = np.array(diff)
    ##################### Visualization#####################
    Visualization.single_line_nodes(PCA_mean_rebuild, 0, 800, context, marker_X, marker_Y, marker_Z)
    # plt.savefig('X:\Siyuanch\Project2\COMMENTS_FROM_JOURNAL_ANATOMY\Rebuttal\P1_'+str(parameter)+'/'+context+'.png')
    # plt.show()
    # Measure the Metopic suture
    # Width
    Metopic_Suture_width = np.mean(Temp[1:j[-1] + 1])
    # Length
    Metopic_Suture_length1 = 0
    Metopic_Suture_length2 = 0
    for i in range(j[-1]):
        Metopic_Suture_length1 = Metopic_Suture_length1 + distance(Suture1[i, :], Suture1[i + 1, :])
        Metopic_Suture_length2 = Metopic_Suture_length2 + distance(Suture2[i, :], Suture2[i + 1, :])
    Metopic_Suture_length = (Metopic_Suture_length2 + Metopic_Suture_length1) / 2
    Begining_point = np.mean([Suture1[0, :], Suture2[0, :]], axis=0)
    Ending_point = np.mean([Suture1[j[-1], :], Suture2[j[-1], :]], axis=0)
    Metopic_chord_length = distance(Begining_point, Ending_point)
    SI = Metopic_Suture_length/Metopic_chord_length
    return Metopic_Suture_length, Metopic_Suture_width,SI

def Sagittal_Suture(PCA_mean_rebuild,context):
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # 2. Get the distance between two nodes
    Temp = []
    DISTANCE_ID = []
    for i in range(200):
        # Temp.append(distance(Suture3[i, :], Suture4[i, :]))
        POINT = Suture3[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        Temp.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
    Temp = np.array(Temp)
    diff = []
    j = []
    for i in range(99):
        diff.append(Temp[i + 1] - Temp[i])
        if Temp[1] > 14.5:
            if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and Temp[i + 2] > Temp[i + 1] and Temp[i] < 24.5: #25:
                j.append(i)
        else:
            if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and Temp[i + 2] > Temp[i + 1] and Temp[i] < 8: #10:
                j.append(i)
    marker_X = np.array([Suture3[0, 0], Suture4[0, 0], Suture3[j[-1], 0], Suture4[DISTANCE_ID[j[-1]], 0]])
    marker_Y = np.array([Suture3[0, 1], Suture4[0, 1], Suture3[j[-1], 1], Suture4[DISTANCE_ID[j[-1]], 1]])
    marker_Z = np.array([Suture3[0, 2], Suture4[0, 2], Suture3[j[-1], 2], Suture4[DISTANCE_ID[j[-1]], 2]])
    # Visualization.single_line_nodes(PCA_mean_rebuild, 0, 800, context, marker_X, marker_Y, marker_Z)
    # plt.show()
    Metopic_Suture_width = np.mean(Temp[0:j[-1] + 1])
    # Length
    Metopic_Suture_length1 = 0
    Metopic_Suture_length2 = 0
    for i in range(j[-1]):
        Metopic_Suture_length1 = Metopic_Suture_length1 + distance(Suture3[i, :], Suture3[i + 1, :])

    for i in range(DISTANCE_ID[j[-1]]):
        Metopic_Suture_length2 = Metopic_Suture_length2 + distance(Suture4[i, :], Suture4[i + 1, :])

    Metopic_Suture_length = (Metopic_Suture_length2 + Metopic_Suture_length1) / 2
    Begining_point = np.mean([Suture3[0, :], Suture4[0, :]], axis=0)
    Ending_point = np.mean([Suture3[j[-1], :], Suture4[j[-1], :]], axis=0)
    Metopic_chord_length = distance(Begining_point, Ending_point)
    SI = Metopic_Suture_length / Metopic_chord_length
    return Metopic_Suture_length, Metopic_Suture_width, SI

def Coronal_Suture(PCA_mean_rebuild,context):
    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # Left Coronal Suture
    DISTANCE = []
    DISTANCE_ID = []
    DIFF = []
    for i in range(np.shape(Suture2)[0]):  # 计算每个2suture的点到suture3的距离
        POINT = Suture2[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID = np.array(DISTANCE_ID)
    DIFF = np.array(DIFF)
    j = []
    paramater = 0.4
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < paramater:
                j.append(i)
    POINT = Suture3[DISTANCE_ID[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture2)
    j.append(POINTB_ID)
    marker_X = np.array([Suture2[j, 0], Suture3[DISTANCE_ID[j], 0]])
    marker_Y = np.array([Suture2[j, 1], Suture3[DISTANCE_ID[j], 1]])
    marker_Z = np.array([Suture2[j, 2], Suture3[DISTANCE_ID[j], 2]])
    Visualization.single_line_nodes(PCA_mean_rebuild, 0, 800, context, marker_X, marker_Y, marker_Z)
    # plt.savefig('X:\Siyuanch\Project2\COMMENTS_FROM_JOURNAL_ANATOMY\Rebuttal\P3_' + str(paramater) + '/' + context + '.png')

    # plt.show()
    # measurement
    Left_Coronal_Suture_Length = 0
    Left_Coronal_Suture_Width = []
    for i in range(j[0],j[-1] +1):
        if i == j[-1]:
            Distance = 0
        else:
            Distance = distance(Suture2[i,:],Suture2[i+1,:])
        Left_Coronal_Suture_Length = Left_Coronal_Suture_Length + Distance
        A,B =  Find_point_in_line(Suture2[i,:],Suture3)
        Left_Coronal_Suture_Width.append(A)
    Left_Coronal_Suture_Width = np.mean(np.array(Left_Coronal_Suture_Width))



    # Right Coronal Suture
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    for i in range(np.shape(Suture1)[0]):  # 计算每个suture1的点到suture4的距离
        POINT = Suture1[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    jj = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                jj.append(i)
    POINT = Suture4[DISTANCE_ID1[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture1)
    jj.append(POINTB_ID)
    # 只保留首尾
    j = [j[0], j[-1]]
    jj = [jj[0], jj[-1]]
    marker_X = np.hstack((Suture2[j, 0], Suture3[DISTANCE_ID[j], 0], Suture1[jj, 0], Suture4[DISTANCE_ID1[jj], 0]))
    marker_Y = np.hstack((Suture2[j, 1], Suture3[DISTANCE_ID[j], 1], Suture1[jj, 1], Suture4[DISTANCE_ID1[jj], 1]))
    marker_Z = np.hstack((Suture2[j, 2], Suture3[DISTANCE_ID[j], 2], Suture1[jj, 2], Suture4[DISTANCE_ID1[jj], 2]))
    # Visualization.single_line_nodes(PCA_mean_rebuild, 0, 800, context, marker_X, marker_Y, marker_Z)
    # plt.show()
    # measurement
    Right_Coronal_Suture_Length = 0
    Right_Coronal_Suture_Width = []
    for i in range(jj[0],jj[-1] +1):
        if i == jj[-1]:
            Distance = 0
        else:
            Distance = distance(Suture1[i,:],Suture1[i+1,:])
        Right_Coronal_Suture_Length = Right_Coronal_Suture_Length + Distance
        A,B =  Find_point_in_line(Suture1[i,:],Suture4)
        Right_Coronal_Suture_Width.append(A)
    Right_Coronal_Suture_Width = np.mean(np.array(Right_Coronal_Suture_Width))
    # Left SI
    # Mean begining and ending point
    # Begining_point = np.mean([Suture2[j[0], :], Suture3[DISTANCE_ID[j[0]], :]], axis=0)
    # Ending_point = np.mean([Suture2[j[-1], :], Suture3[DISTANCE_ID[j[-1]], :]], axis=0)
    # Consider begining and ending points on suture 2
    Begining_point = Suture2[j[0], :]
    Ending_point = Suture2[j[-1], :]

    Left_Coronal_S_chord_length = distance(Begining_point, Ending_point)
    LSI = Left_Coronal_Suture_Length / Left_Coronal_S_chord_length
    # Right SI
    # Begining_point = np.mean([Suture1[jj[0], :], Suture4[DISTANCE_ID1[jj[0]], :]], axis=0)
    # Ending_point = np.mean([Suture1[jj[-1], :], Suture4[DISTANCE_ID1[jj[-1]], :]], axis=0)
    Begining_point = Suture1[jj[0], :]
    Ending_point = Suture1[jj[-1], :]

    Right_Coronal_S_chord_length = distance(Begining_point, Ending_point)
    RSI = Right_Coronal_Suture_Length / Right_Coronal_S_chord_length
    SI = (LSI + RSI) / 2
    return Left_Coronal_Suture_Length, Left_Coronal_Suture_Width, Right_Coronal_Suture_Length,Right_Coronal_Suture_Width,SI

def Squamosal_Suture(PCA_mean_rebuild,context):
    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # Left Coronal Suture
    DISTANCE = []
    DISTANCE_ID = []
    DIFF = []

    for i in range(np.shape(Suture2)[0]):  # 计算每个suture2的点到suture3的距离
        POINT = Suture2[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID = np.array(DISTANCE_ID)
    DIFF = np.array(DIFF)
    j = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                j.append(i)
    POINT = Suture3[DISTANCE_ID[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture2)
    j.append(POINTB_ID)
    # marker_X = np.array([Suture2[j, 0], Suture3[DISTANCE_ID[j], 0]])
    # marker_Y = np.array([Suture2[j, 1], Suture3[DISTANCE_ID[j], 1]])
    # marker_Z = np.array([Suture2[j, 2], Suture3[DISTANCE_ID[j], 2]])
    # Visualization.single_line_nodes(PCA_mean_rebuild, 0, 800, context, marker_X, marker_Y, marker_Z)
    # plt.show()

    # Right Coronal Suture
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    for i in range(np.shape(Suture1)[0]):  # 计算每个suture1的点到suture4的距离
        POINT = Suture1[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    jj = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                jj.append(i)
    POINT = Suture4[DISTANCE_ID1[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture1)
    jj.append(POINTB_ID)
    j = [j[0], j[-1]]
    jj = [jj[0], jj[-1]]
    # marker_X = np.hstack((Suture2[j, 0], Suture3[DISTANCE_ID[j], 0], Suture1[jj, 0], Suture4[DISTANCE_ID1[jj], 0]))
    # marker_Y = np.hstack((Suture2[j, 1], Suture3[DISTANCE_ID[j], 1], Suture1[jj, 1], Suture4[DISTANCE_ID1[jj], 1]))
    # marker_Z = np.hstack((Suture2[j, 2], Suture3[DISTANCE_ID[j], 2], Suture1[jj, 2], Suture4[DISTANCE_ID1[jj], 2]))
    # Visualization.single_line_nodes(PCA_mean_rebuild, 0, 800, context, marker_X, marker_Y, marker_Z)
    # plt.show()

    # Find Left Squamosal Suture
    POINT_3 = Suture3[DISTANCE_ID[j[-1]], :]
    Suture5 = []
    Suture6 = []
    for i in range(50):
        Suture5.append([PCA_mean_rebuild[3 * i + 1800], PCA_mean_rebuild[3 * i + 1801], PCA_mean_rebuild[3 * i + 1802]])
        Suture6.append([PCA_mean_rebuild[3 * i + 1950], PCA_mean_rebuild[3 * i + 1951], PCA_mean_rebuild[3 * i + 1952]])
    Suture5 = np.array(Suture5)
    Suture6 = np.array(Suture6)
    useless, POINTB_ID = Find_point_in_line(POINT_3, Suture5)
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    jjj = []
    for i in range(0, 50):  # 计算每个suture5的点到suture3的距离
        POINT = Suture5[i, :]
        temp1, POINTB_ID1 = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID1)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    for i in range(49):
        if DISTANCE[i] < DISTANCE[0] / 2:
            if abs(DIFF[i]) < 0.3:
                jjj.append(i)
            else:
                if i > 0 and abs(DIFF[i] + DIFF[i - 1]) < 0.3:
                    jjj.append(i)
    # measurement
    # marker_X = np.hstack((marker_X, Suture5[jjj, 0], Suture3[DISTANCE_ID1[(np.array(jjj)).tolist()], 0],
    #                       Suture6[jjjj, 0], Suture4[DISTANCE_ID2[(np.array(jjjj)).tolist()], 0]))
    Left_Squamosal_Suture_Length = 0
    Left_Squamosal_Suture_Width = []
    # Suture 5
    for i in range(jjj[0], jjj[-1] + 1):
        if i == jjj[-1]:
            Distance = 0
        else:
            Distance = distance(Suture5[i, :], Suture5[i + 1, :])
        Left_Squamosal_Suture_Length = Left_Squamosal_Suture_Length + Distance
        A, B = Find_point_in_line(Suture5[i, :], Suture3)
        Left_Squamosal_Suture_Width.append(A)
    # Suture 3
    for i in range(DISTANCE_ID1[jjj[-1]], DISTANCE_ID1[jjj[0]]+ 1):
        if i == DISTANCE_ID1[jjj[0]]:
            Distance = 0
        else:
            Distance = distance(Suture3[i, :], Suture3[i + 1, :])
        Left_Squamosal_Suture_Length = Left_Squamosal_Suture_Length + Distance
        A, B = Find_point_in_line(Suture3[i, :], Suture5)
        Left_Squamosal_Suture_Width.append(A)
    Left_Squamosal_Suture_Width = np.mean(np.array(Left_Squamosal_Suture_Width))
    Left_Squamosal_Suture_Length = Left_Squamosal_Suture_Length / 2



    # Initialization Right Squamosal Suture
    jjjj = []
    DISTANCE_ID2 = []
    DISTANCE = []
    DIFF = []
    for i in range(0, 50):  # 计算每个suture6的点到suture4的距离
        POINT = Suture6[i, :]
        temp1, POINTB_ID1 = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID2.append(POINTB_ID1)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID2 = np.array(DISTANCE_ID2)
    DIFF = np.array(DIFF)
    for i in range(49):
        if DISTANCE[i] < DISTANCE[0] / 2:
            if abs(DIFF[i]) < 0.3:
                jjjj.append(i)
            else:
                if i > 0 and abs(DIFF[i] + DIFF[i - 1]) < 0.3:
                    jjjj.append(i)

    jjj = [jjj[0], jjj[-1]]
    jjjj = [jjjj[0], jjjj[-1]]
    marker_X = np.hstack(( Suture5[jjj, 0], Suture3[DISTANCE_ID1[(np.array(jjj)).tolist()], 0],
                          Suture6[jjjj, 0], Suture4[DISTANCE_ID2[(np.array(jjjj)).tolist()], 0]))
    marker_Y = np.hstack(( Suture5[jjj, 1], Suture3[DISTANCE_ID1[(np.array(jjj)).tolist()], 1],
                          Suture6[jjjj, 1], Suture4[DISTANCE_ID2[(np.array(jjjj)).tolist()], 1]))
    marker_Z = np.hstack(( Suture5[jjj, 2], Suture3[DISTANCE_ID1[(np.array(jjj)).tolist()], 2],
                          Suture6[jjjj, 2], Suture4[DISTANCE_ID2[(np.array(jjjj)).tolist()], 2]))
    # Visualization.single_line_nodes(PCA_mean_rebuild, 0, 800, context, marker_X, marker_Y, marker_Z)
    # plt.show()
    # measurement
    Right_Squamosal_Suture_Length = 0
    Right_Squamosal_Suture_Width = []
    # Suture 6
    for i in range(jjjj[0], jjjj[-1] + 1):
        if i == jjjj[-1]:
            Distance = 0
        else:
            Distance = distance(Suture6[i, :], Suture6[i + 1, :])
        Right_Squamosal_Suture_Length = Right_Squamosal_Suture_Length + Distance
        A, B = Find_point_in_line(Suture6[i, :], Suture4)
        Right_Squamosal_Suture_Width.append(A)
    # Suture 4
    for i in range(DISTANCE_ID2[jjjj[-1]], DISTANCE_ID2[jjjj[0]]+ 1):
        if i == DISTANCE_ID2[jjjj[0]]:
            Distance = 0
        else:
            Distance = distance(Suture4[i, :], Suture4[i + 1, :])
        Right_Squamosal_Suture_Length = Right_Squamosal_Suture_Length + Distance
        A, B = Find_point_in_line(Suture4[i, :], Suture6)
        Right_Squamosal_Suture_Width.append(A)
    Right_Squamosal_Suture_Width = np.mean(np.array(Right_Squamosal_Suture_Width))
    Right_Squamosal_Suture_Length = Right_Squamosal_Suture_Length / 2
    # Left SI
    Begining_point = np.mean([Suture5[jjj[0], :], Suture3[DISTANCE_ID1[(np.array(jjj[0])).tolist()], :]], axis=0)
    Ending_point = np.mean([Suture5[jjj[-1], :], Suture3[DISTANCE_ID1[(np.array(jjj[-1])).tolist()], :]], axis=0)
    Left_Squamosal_S_chord_length = distance(Begining_point, Ending_point)
    LSI = Left_Squamosal_Suture_Length / Left_Squamosal_S_chord_length
    # Right SI
    Begining_point = np.mean([Suture6[jjjj[0], :], Suture4[DISTANCE_ID1[(np.array(jjjj[0])).tolist()], :]], axis=0)
    Ending_point = np.mean([Suture6[jjjj[-1], :], Suture4[DISTANCE_ID1[(np.array(jjjj[-1])).tolist()], :]], axis=0)
    Right_Squamosal_S_chord_length = distance(Begining_point, Ending_point)
    RSI = Right_Squamosal_Suture_Length / Right_Squamosal_S_chord_length
    SI = (LSI + RSI) / 2
    return Left_Squamosal_Suture_Length, Left_Squamosal_Suture_Width, Right_Squamosal_Suture_Length,Right_Squamosal_Suture_Width,SI

def Parts_metopic(PCA_mean_rebuild,context):
    # Metopic suture
    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)
    # 2. Get the distance between two nodes
    Temp = []
    for i in range(100):
        Temp.append(distance(Suture1[i, :], Suture2[i, :]))
    Temp = np.array(Temp)
    diff = []
    j = []
    for i in range(99):
        diff.append(Temp[i + 1] - Temp[i])
        if Temp[1] > 5:
            if diff[i] > 0.15 and diff[i] > 1.5* diff[i - 1] and diff[i] < 1 and Temp[i] < 12:
                j.append(i)
        else:
            # if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and diff[i] < 1 and Temp[i] < 3:   OLD
            if diff[i] > 0.15 and diff[i] > 1.5 * diff[i - 1] and diff[i] < 1and Temp[i] < 3:
                # if i>1 and diff[i-1] < 1.5* diff[i-2]:
                j.append(i)
    # ID of end of metopic suture is j[-1]
    return j[-1]

def LMD_metopic(PCA_FE_real,PCA_FE_rebuild, ID_metopic):
    Suture1_real = []
    Suture2_real = []
    Suture1_pred = []
    Suture2_pred = []
    for i in range(100):
        Suture1_real.append([PCA_FE_real[3 * i], PCA_FE_real[3 * i + 1], PCA_FE_real[3 * i + 2]])
        Suture2_real.append([PCA_FE_real[3 * i + 300], PCA_FE_real[3 * i + 301], PCA_FE_real[3 * i + 302]])
        Suture1_pred.append([PCA_FE_rebuild[3 * i], PCA_FE_rebuild[3 * i + 1], PCA_FE_rebuild[3 * i + 2]])
        Suture2_pred.append([PCA_FE_rebuild[3 * i + 300], PCA_FE_rebuild[3 * i + 301], PCA_FE_rebuild[3 * i + 302]])
    Suture1_real = np.array(Suture1_real)
    Suture2_real = np.array(Suture2_real)
    Suture1_pred = np.array(Suture1_pred)
    Suture2_pred = np.array(Suture2_pred)
    # metopic suture
    MS_real = []
    MS_pred = []
    for i in range(ID_metopic):
        MS_real.append(Suture1_real[i][0])
        MS_real.append(Suture1_real[i][1])
        MS_real.append(Suture1_real[i][2])
        MS_real.append(Suture2_real[i][0])
        MS_real.append(Suture2_real[i][1])
        MS_real.append(Suture2_real[i][2])

        MS_pred.append(Suture1_pred[i][0])
        MS_pred.append(Suture1_pred[i][1])
        MS_pred.append(Suture1_pred[i][2])
        MS_pred.append(Suture2_pred[i][0])
        MS_pred.append(Suture2_pred[i][1])
        MS_pred.append(Suture2_pred[i][2])
    MS_real = np.array(MS_real)
    MS_pred = np.array(MS_pred)
    MS_real = np.transpose(MS_real)
    MS_pred = np.transpose(MS_pred)
    return MS_real, MS_pred

def Parts_AF(PCA_mean_rebuild,pred,context):
    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)
    # 2. Get the distance between two nodes
    Temp = []
    for i in range(100):
        Temp.append(distance(Suture1[i, :], Suture2[i, :]))
    Temp = np.array(Temp)
    diff = []
    j = []
    for i in range(99):
        diff.append(Temp[i + 1] - Temp[i])
        if Temp[1] > 5:
            if diff[i] > 0.15 and diff[i] > 1.5 * diff[i - 1] and diff[i] < 1 and Temp[i] < 12:
                j.append(i)
        else:
            # if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and diff[i] < 1 and Temp[i] < 3:   OLD
            if diff[i] > 0.15 and diff[i] > 1.5 * diff[i - 1] and diff[i] < 1 and Temp[i] < 3:
                # if i>1 and diff[i-1] < 1.5* diff[i-2]:
                j.append(i)
    marker_X = np.array([Suture1[j[-1], 0], Suture2[j[-1], 0]])
    marker_Y = np.array([Suture1[j[-1], 1], Suture2[j[-1], 1]])
    marker_Z = np.array([Suture1[j[-1], 2], Suture2[j[-1], 2]])
    diff = np.array(diff)
    Metopic_point_ID = j[-1]

    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # Left Coronal Suture
    DISTANCE = []
    DISTANCE_ID = []
    DIFF = []
    for i in range(np.shape(Suture2)[0]):  # 计算每个2suture的点到suture3的距离
        POINT = Suture2[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID = np.array(DISTANCE_ID)
    DIFF = np.array(DIFF)
    j = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                j.append(i)
    POINT = Suture3[DISTANCE_ID[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture2)
    j.append(POINTB_ID)

    # Right Coronal Suture
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    for i in range(np.shape(Suture1)[0]):  # 计算每个suture1的点到suture4的距离
        POINT = Suture1[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    jj = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                jj.append(i)
    POINT = Suture4[DISTANCE_ID1[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture1)
    jj.append(POINTB_ID)
    # 只保留首尾
    j = [j[0], j[-1]]
    jj = [jj[0], jj[-1]]

    Left_Coronal_point_ID2 = j[0]
    Left_Coronal_point_ID3 = DISTANCE_ID[j[0]]
    Right_Coronal_point_ID6 = DISTANCE_ID1[jj[0]]
    Right_Coronal_point_ID7 = jj[0]

    # Get anterior fontanel points
    Anterior_fontanel_points = []
    Anterior_fontanel_points_pred = []

    Suture1_pred = []
    Suture2_pred = []
    for i in range(100):
        Suture1_pred.append([pred[3 * i], pred[3 * i + 1], pred[3 * i + 2]])
        Suture2_pred.append([pred[3 * i + 300], pred[3 * i + 301], pred[3 * i + 302]])
    Suture1_pred = np.array(Suture1_pred)
    Suture2_pred = np.array(Suture2_pred)
    Suture3_pred = []
    Suture4_pred = []
    for i in range(200):
        Suture3_pred.append([pred[3 * i + 600], pred[3 * i + 601], pred[3 * i + 602]])
        Suture4_pred.append([pred[3 * i + 1200], pred[3 * i + 1201], pred[3 * i + 1202]])
    Suture3_pred = np.array(Suture3_pred)
    Suture4_pred = np.array(Suture4_pred)


    for i in range(Metopic_point_ID, Left_Coronal_point_ID2 + 1):
        Anterior_fontanel_points.append(Suture2[i, :])
        Anterior_fontanel_points_pred.append(Suture2_pred[i, :])
    for i in range(Left_Coronal_point_ID3, 200):
        Anterior_fontanel_points.append(Suture3[i, :])
        Anterior_fontanel_points_pred.append(Suture3_pred[i, :])
    for i in range(199, Right_Coronal_point_ID6 - 1, -1):
        Anterior_fontanel_points.append(Suture4[i, :])
        Anterior_fontanel_points_pred.append(Suture4_pred[i, :])
    for i in range(Right_Coronal_point_ID7, Metopic_point_ID - 1, -1):
        Anterior_fontanel_points.append(Suture1[i, :])
        Anterior_fontanel_points_pred.append(Suture1_pred[i, :])

    Anterior_fontanel_points = np.array(Anterior_fontanel_points)
    Anterior_fontanel_points_pred = np.array(Anterior_fontanel_points_pred)
    return Anterior_fontanel_points,Anterior_fontanel_points_pred

def Parts_SaS(PCA_mean_rebuild,pred,context):
    # 3. Sagittal suture area XY plane
    context = 'PF'
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # 2. Get the distance between two nodes ##### Here find the smallest distance
    Temp = []
    DISTANCE_ID = []
    for i in range(200):
        POINT = Suture3[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        Temp.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        #####################################
        # Temp.append(distance(Suture3[i, :], Suture4[i, :]))
    Temp = np.array(Temp)
    diff = []
    j = []
    for i in range(99):
        diff.append(Temp[i + 1] - Temp[i])
        if Temp[1] > 14.5:
            if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and Temp[i + 2] > Temp[i + 1] and Temp[i] < 24.5:
                j.append(i)
        else:
            if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and Temp[i + 2] > Temp[i + 1] and Temp[i] < 8:
                j.append(i)
    Id_4 = j[-1]
    Id_5 = DISTANCE_ID[j[-1]]

    XY_Z_points = []
    XY_Z_points_pred = []

    Suture3_pred = []
    Suture4_pred = []
    for i in range(200):
        Suture3_pred.append([pred[3 * i + 600], pred[3 * i + 601], pred[3 * i + 602]])
        Suture4_pred.append([pred[3 * i + 1200], pred[3 * i + 1201], pred[3 * i + 1202]])
    Suture3_pred = np.array(Suture3_pred)
    Suture4_pred = np.array(Suture4_pred)
    for i in range(0, Id_4 + 1):
        XY_Z_points.append(Suture3[i, :])
        XY_Z_points_pred.append(Suture3_pred[i, :])
    for i in range(Id_5, -1, -1):
        XY_Z_points.append(Suture4[i, :])
        XY_Z_points_pred.append(Suture4_pred[i, :])

    XY_Z_points = np.array(XY_Z_points)
    XY_Z_points_pred = np.array(XY_Z_points_pred)
    # Get pred size
    Temp_pred = []
    DISTANCE_ID_pred = []
    for i in range(200):
        # Temp.append(distance(Suture3[i, :], Suture4[i, :]))
        POINT = Suture3_pred[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4_pred)
        Temp_pred.append(temp1)
        DISTANCE_ID_pred.append(POINTB_ID)
    Temp_pred = np.array(Temp_pred)
    # width
    Metopic_Suture_width_pred = np.mean(Temp_pred[0:j[-1] + 1])
    # Length
    Metopic_Suture_length_pred_1 = 0
    Metopic_Suture_length_pred_2 = 0
    for i in range(j[-1]):
        Metopic_Suture_length_pred_1 = Metopic_Suture_length_pred_1 + distance(Suture3_pred[i, :], Suture3_pred[i + 1, :])

    for i in range(DISTANCE_ID[j[-1]]):
        Metopic_Suture_length_pred_2 = Metopic_Suture_length_pred_2 + distance(Suture4_pred[i, :], Suture4_pred[i + 1, :])

    Metopic_Suture_length_pred = (Metopic_Suture_length_pred_2 + Metopic_Suture_length_pred_1) / 2
    Begining_point_pred = np.mean([Suture3_pred[0, :], Suture4_pred[0, :]], axis=0)
    Ending_point_pred = np.mean([Suture3_pred[j[-1], :], Suture4_pred[j[-1], :]], axis=0)
    Metopic_chord_length_pred = distance(Begining_point_pred, Ending_point_pred)
    SI_pred = Metopic_Suture_length_pred / Metopic_chord_length_pred
    return XY_Z_points,XY_Z_points_pred,Metopic_Suture_length_pred, Metopic_Suture_width_pred, SI_pred

def Parts_PF(PCA_mean_rebuild,pred,context):
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)

    # 2. Get the distance between two nodes ##### Here find the smallest distance
    Temp = []
    DISTANCE_ID = []
    for i in range(200):
        POINT = Suture3[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        Temp.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        #####################################
        # Temp.append(distance(Suture3[i, :], Suture4[i, :]))
    Temp = np.array(Temp)
    diff = []
    j = []
    for i in range(99):
        diff.append(Temp[i + 1] - Temp[i])
        if Temp[1] > 14.5:
            if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and Temp[i + 2] > Temp[i + 1] and Temp[i] < 24.5:
                j.append(i)
        else:
            if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and Temp[i + 2] > Temp[i + 1] and Temp[i] < 8:
                j.append(i)
    Id_4 = j[-1]
    Id_5 = DISTANCE_ID[j[-1]]

    # Left Lambdoid Suture
    Suture7 = []
    for i in range(100):
        Suture7.append(
            [PCA_mean_rebuild[3 * i + 2100], PCA_mean_rebuild[3 * i + 2101], PCA_mean_rebuild[3 * i + 2102]])
    Suture7 = np.array(Suture7)
    DISTANCE_L = []
    DISTANCE_IDL = []
    DIFF_L = []
    j_L = []
    for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
        POINT_L = Suture3[i, :]
        temp1_L, POINTB_L = Find_point_in_line(POINT_L, Suture7)
        DISTANCE_L.append(temp1_L)
        DISTANCE_IDL.append(POINTB_L)
        if i > 0:
            diff111 = DISTANCE_L[i] - DISTANCE_L[i - 1]
            DIFF_L.append(diff111)
    DISTANCE_L = np.array(DISTANCE_L)
    DISTANCE_IDL = np.array(DISTANCE_IDL)
    DIFF_L = np.array(DIFF_L)
    for i in range(199):
        if DISTANCE_L[i] < 10:
            if abs(DIFF_L[i]) < 0.4:
                j_L.append(i)
    j_L = [j_L[0], j_L[-1]]
    ID_1 = DISTANCE_IDL[j_L[0]]
    ID_6 = j_L[0]
    # Right Lambdoid Suture
    DISTANCE_R = []
    DISTANCE_IDR = []
    DIFF_R = []
    j_R = []
    for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
        POINT_R = Suture4[i, :]
        temp1_R, POINTB_R = Find_point_in_line(POINT_R, Suture7)
        DISTANCE_R.append(temp1_R)
        DISTANCE_IDR.append(POINTB_R)
        if i > 0:
            diff111 = DISTANCE_R[i] - DISTANCE_R[i - 1]
            DIFF_R.append(diff111)
    DISTANCE_R = np.array(DISTANCE_R)
    DISTANCE_IDR = np.array(DISTANCE_IDR)
    DIFF_R = np.array(DIFF_R)
    for i in range(199):
        if DISTANCE_R[i] < 10:
            if abs(DIFF_R[i]) < 0.4:
                j_R.append(i)
    j_R = [j_R[0], j_R[-1]]
    ID_2 = DISTANCE_IDR[j_R[0]]
    ID_3 = j_R[0]
    # Get posterior fontanel points
    Posterior_fontanel_points = []
    Posterior_fontanel_points_pred = []
    Suture7_pred = []
    for i in range(100):
        Suture7_pred.append(
            [pred[3 * i + 2100], pred[3 * i + 2101], pred[3 * i + 2102]])
    Suture7_pred = np.array(Suture7_pred)

    Suture3_pred = []
    Suture4_pred = []
    for i in range(200):
        Suture3_pred.append([pred[3 * i + 600], pred[3 * i + 601], pred[3 * i + 602]])
        Suture4_pred.append([pred[3 * i + 1200], pred[3 * i + 1201], pred[3 * i + 1202]])
    Suture3_pred = np.array(Suture3_pred)
    Suture4_pred = np.array(Suture4_pred)


    for i in range(ID_1, ID_2 + 1):
        Posterior_fontanel_points.append(Suture7[i, :])
        Posterior_fontanel_points_pred.append(Suture7_pred[i, :])
    for i in range(ID_3, Id_5 - 1, -1):
        Posterior_fontanel_points.append(Suture4[i, :])
        Posterior_fontanel_points_pred.append(Suture4_pred[i, :])
    for i in range(Id_4, ID_6 + 1):
        Posterior_fontanel_points.append(Suture3[i, :])
        Posterior_fontanel_points_pred.append(Suture3_pred[i, :])

    Posterior_fontanel_points = np.array(Posterior_fontanel_points)
    Posterior_fontanel_points_pred = np.array(Posterior_fontanel_points_pred)

    return Posterior_fontanel_points, Posterior_fontanel_points_pred

def Parts_CS(PCA_mean_rebuild,pred,context):
    # Left coronal suture
    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # Left Coronal Suture
    DISTANCE = []
    DISTANCE_ID = []
    DIFF = []
    for i in range(np.shape(Suture2)[0]):  # 计算每个2suture的点到suture3的距离
        POINT = Suture2[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID = np.array(DISTANCE_ID)
    DIFF = np.array(DIFF)
    j = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                j.append(i)
    POINT = Suture3[DISTANCE_ID[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture2)
    j.append(POINTB_ID)

    # Get Left coronal suture points
    Left_Coronal_area_points = []
    Left_Coronal_area_points_pred = []
    Suture1_pred = []
    Suture2_pred = []
    for i in range(100):
        Suture1_pred.append([pred[3 * i], pred[3 * i + 1], pred[3 * i + 2]])
        Suture2_pred.append([pred[3 * i + 300], pred[3 * i + 301], pred[3 * i + 302]])
    Suture1_pred = np.array(Suture1_pred)
    Suture2_pred = np.array(Suture2_pred)
    Suture3_pred = []
    Suture4_pred = []
    for i in range(200):
        Suture3_pred.append([pred[3 * i + 600], pred[3 * i + 601], pred[3 * i + 602]])
        Suture4_pred.append([pred[3 * i + 1200], pred[3 * i + 1201], pred[3 * i + 1202]])
    Suture3_pred = np.array(Suture3_pred)
    Suture4_pred = np.array(Suture4_pred)
    for i in range(j[0], j[-1] + 1):
        Left_Coronal_area_points.append(Suture2[i, :])
        Left_Coronal_area_points_pred.append(Suture2_pred[i, :])
    for i in range(DISTANCE_ID[j[-1]], DISTANCE_ID[j[0]] + 1):
        Left_Coronal_area_points.append(Suture3[i, :])
        Left_Coronal_area_points_pred.append(Suture3_pred[i, :])

    Left_Coronal_area_points = np.array(Left_Coronal_area_points)
    Left_Coronal_area_points_pred = np.array(Left_Coronal_area_points_pred)
    # measurement
    # Preditive size measurement Left
    Left_Coronal_Suture_Length_pred = 0
    Left_Coronal_Suture_Width_pred = []
    for i in range(j[0], j[-1] + 1):
        if i == j[-1]:
            Distance = 0
        else:
            Distance = distance(Suture2_pred[i, :], Suture2_pred[i + 1, :])
        Left_Coronal_Suture_Length_pred= Left_Coronal_Suture_Length_pred + Distance
        A, B = Find_point_in_line(Suture2_pred[i, :], Suture3_pred)
        Left_Coronal_Suture_Width_pred.append(A)
    Left_Coronal_Suture_Width_pred = np.mean(np.array(Left_Coronal_Suture_Width_pred))

    # Right coronal suture
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    for i in range(np.shape(Suture1)[0]):  # 计算每个suture1的点到suture4的距离
        POINT = Suture1[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    jj = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                jj.append(i)
    POINT = Suture4[DISTANCE_ID1[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture1)
    jj.append(POINTB_ID)
    # 只保留首尾
    j = [j[0], j[-1]]
    jj = [jj[0], jj[-1]]

    # Get right coronal suture points
    Right_Coronal_area_points = []
    Right_Coronal_area_points_pred = []


    for i in range(jj[0], jj[-1] + 1):
        Right_Coronal_area_points.append(Suture1[i, :])
        Right_Coronal_area_points_pred.append(Suture1_pred[i, :])
    for i in range(DISTANCE_ID1[jj[-1]], DISTANCE_ID1[jj[0]] + 1):
        Right_Coronal_area_points.append(Suture4[i, :])
        Right_Coronal_area_points_pred.append(Suture4_pred[i, :])

    Right_Coronal_area_points = np.array(Right_Coronal_area_points)
    Right_Coronal_area_points_pred = np.array(Right_Coronal_area_points_pred)

    # measurement
    Right_Coronal_Suture_Length_pred = 0
    Right_Coronal_Suture_Width_pred  = []
    for i in range(jj[0], jj[-1] + 1):
        if i == jj[-1]:
            Distance = 0
        else:
            Distance = distance(Suture1_pred[i, :], Suture1_pred[i + 1, :])
        Right_Coronal_Suture_Length_pred  = Right_Coronal_Suture_Length_pred  + Distance
        A, B = Find_point_in_line(Suture1_pred[i, :], Suture4_pred)
        Right_Coronal_Suture_Width_pred.append(A)
    Right_Coronal_Suture_Width_pred = np.mean(np.array(Right_Coronal_Suture_Width_pred))

    # Left SI
    Begining_point_pred = Suture2_pred[j[0], :]
    Ending_point_pred = Suture2_pred[j[-1], :]

    Left_Coronal_S_chord_length_pred = distance(Begining_point_pred, Ending_point_pred)
    LSI_pred = Left_Coronal_Suture_Length_pred / Left_Coronal_S_chord_length_pred
    # Right SI
    Begining_point_pred = Suture1_pred[jj[0], :]
    Ending_point_pred = Suture1_pred[jj[-1], :]

    Right_Coronal_S_chord_length_pred = distance(Begining_point_pred, Ending_point_pred)
    RSI_pred = Right_Coronal_Suture_Length_pred / Right_Coronal_S_chord_length_pred
    SI_pred = (LSI_pred + RSI_pred) / 2


    Coronal_points = np.vstack((Right_Coronal_area_points,Left_Coronal_area_points))
    Coronal_points_pred = np.vstack((Right_Coronal_area_points_pred, Left_Coronal_area_points_pred))
    return  Coronal_points , Coronal_points_pred,Left_Coronal_Suture_Length_pred, Left_Coronal_Suture_Width_pred\
        , Right_Coronal_Suture_Length_pred,Right_Coronal_Suture_Width_pred,SI_pred

def Parts_SqS(PCA_mean_rebuild,pred,context):
    # 4. Squamosal suture
    # left squamosal suture
    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # Left Coronal Suture
    DISTANCE = []
    DISTANCE_ID = []
    DIFF = []
    for i in range(np.shape(Suture2)[0]):  # 计算每个suture2的点到suture3的距离
        POINT = Suture2[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID = np.array(DISTANCE_ID)
    DIFF = np.array(DIFF)
    j = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                j.append(i)
    POINT = Suture3[DISTANCE_ID[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture2)
    j.append(POINTB_ID)

    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    for i in range(np.shape(Suture1)[0]):  # 计算每个suture1的点到suture4的距离
        POINT = Suture1[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    jj = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                jj.append(i)
    POINT = Suture4[DISTANCE_ID1[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture1)
    jj.append(POINTB_ID)
    j = [j[0], j[-1]]
    jj = [jj[0], jj[-1]]

    POINT_3 = Suture3[DISTANCE_ID[j[-1]], :]
    ## Left squamosal
    Suture5 = []
    Suture6 = []
    for i in range(50):
        Suture5.append([PCA_mean_rebuild[3 * i + 1800], PCA_mean_rebuild[3 * i + 1801], PCA_mean_rebuild[3 * i + 1802]])
        Suture6.append([PCA_mean_rebuild[3 * i + 1950], PCA_mean_rebuild[3 * i + 1951], PCA_mean_rebuild[3 * i + 1952]])
    Suture5 = np.array(Suture5)
    Suture6 = np.array(Suture6)
    useless, POINTB_ID = Find_point_in_line(POINT_3, Suture5)
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    jjj = []
    for i in range(0, 50):  # 计算每个suture5的点到suture3的距离
        POINT = Suture5[i, :]
        temp1, POINTB_ID1 = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID1)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    for i in range(49):
        if DISTANCE[i] < DISTANCE[0] / 2:
            if abs(DIFF[i]) < 0.3:
                jjj.append(i)
            else:
                if i > 0 and abs(DIFF[i] + DIFF[i - 1]) < 0.3:
                    jjj.append(i)
    # Get Left Sphenoid suture points
    Left_squamosal_area_points = []
    Left_squamosal_area_points_pred = []
    Suture1_pred = []
    Suture2_pred = []
    for i in range(100):
        Suture1_pred.append([pred[3 * i], pred[3 * i + 1], pred[3 * i + 2]])
        Suture2_pred.append([pred[3 * i + 300], pred[3 * i + 301], pred[3 * i + 302]])
    Suture1_pred = np.array(Suture1_pred)
    Suture2_pred = np.array(Suture2_pred)
    Suture3_pred = []
    Suture4_pred = []
    for i in range(200):
        Suture3_pred.append([pred[3 * i + 600], pred[3 * i + 601], pred[3 * i + 602]])
        Suture4_pred.append([pred[3 * i + 1200], pred[3 * i + 1201], pred[3 * i + 1202]])
    Suture3_pred = np.array(Suture3_pred)
    Suture4_pred = np.array(Suture4_pred)
    Suture5_pred = []
    Suture6_pred = []
    for i in range(50):
        Suture5_pred.append([pred[3 * i + 1800], pred[3 * i + 1801], pred[3 * i + 1802]])
        Suture6_pred.append([pred[3 * i + 1950], pred[3 * i + 1951], pred[3 * i + 1952]])
    Suture5_pred = np.array(Suture5_pred)
    Suture6_pred = np.array(Suture6_pred)
    for i in range(jjj[0], jjj[-1] + 1):
        Left_squamosal_area_points.append(Suture5[i, :])
        Left_squamosal_area_points_pred.append(Suture5_pred[i, :])
    for i in range(DISTANCE_ID1[jjj[-1]], DISTANCE_ID1[jjj[0]] + 1):
        Left_squamosal_area_points_pred.append(Suture3_pred[i, :])
        Left_squamosal_area_points.append(Suture3[i, :])
    Left_squamosal_area_points = np.array(Left_squamosal_area_points)
    Left_squamosal_area_points_pred = np.array(Left_squamosal_area_points_pred)

    # measurement
    # Left size
    Left_Squamosal_Suture_Length_pred = 0
    Left_Squamosal_Suture_Width_pred = []
    # Suture 5
    for i in range(jjj[0], jjj[-1] + 1):
        if i == jjj[-1]:
            Distance = 0
        else:
            Distance = distance(Suture5_pred[i, :], Suture5_pred[i + 1, :])
        Left_Squamosal_Suture_Length_pred = Left_Squamosal_Suture_Length_pred + Distance
        A, B = Find_point_in_line(Suture5_pred[i, :], Suture3_pred)
        Left_Squamosal_Suture_Width_pred.append(A)
    # Suture 3
    for i in range(DISTANCE_ID1[jjj[-1]], DISTANCE_ID1[jjj[0]]+ 1):
        if i == DISTANCE_ID1[jjj[0]]:
            Distance = 0
        else:
            Distance = distance(Suture3_pred[i, :], Suture3_pred[i + 1, :])
        Left_Squamosal_Suture_Length_pred = Left_Squamosal_Suture_Length_pred + Distance
        A, B = Find_point_in_line(Suture3_pred[i, :], Suture5_pred)
        Left_Squamosal_Suture_Width_pred.append(A)
    Left_Squamosal_Suture_Width_pred = np.mean(np.array(Left_Squamosal_Suture_Width_pred))
    Left_Squamosal_Suture_Length_pred = Left_Squamosal_Suture_Length_pred / 2



    # right squamosal suture
    jjjj = []
    DISTANCE_ID2 = []
    DISTANCE = []
    DIFF = []
    for i in range(0, 50):  # 计算每个suture6的点到suture4的距离
        POINT = Suture6[i, :]
        temp1, POINTB_ID1 = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID2.append(POINTB_ID1)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID2 = np.array(DISTANCE_ID2)
    DIFF = np.array(DIFF)
    for i in range(49):
        if DISTANCE[i] < DISTANCE[0] / 2:
            if abs(DIFF[i]) < 0.3:
                jjjj.append(i)
            else:
                if i > 0 and abs(DIFF[i] + DIFF[i - 1]) < 0.3:
                    jjjj.append(i)

    jjj = [jjj[0], jjj[-1]]
    # print(jjj)
    # print(jjjj)
    jjjj = [jjjj[0], jjjj[-1]]

    Right_squamosal_area_points = []
    Right_squamosal_area_points_pred = []
    for i in range(jjjj[0], jjjj[-1] + 1):
        Right_squamosal_area_points.append(Suture6[i, :])
        Right_squamosal_area_points_pred.append(Suture6_pred[i, :])
    for i in range(DISTANCE_ID2[jjjj[-1]], DISTANCE_ID2[jjjj[0]] + 1):
        Right_squamosal_area_points.append(Suture4[i, :])
        Right_squamosal_area_points_pred.append(Suture4_pred[i, :])

    Right_squamosal_area_points_pred = np.array(Right_squamosal_area_points_pred)
    Right_squamosal_area_points = np.array(Right_squamosal_area_points)
    points = np.vstack((Right_squamosal_area_points,Left_squamosal_area_points))
    points_pred = np.vstack((Right_squamosal_area_points_pred, Left_squamosal_area_points_pred))
    # measurement
    # Right
    Right_Squamosal_Suture_Length_pred = 0
    Right_Squamosal_Suture_Width_pred = []
    # Suture 6
    for i in range(jjjj[0], jjjj[-1] + 1):
        if i == jjjj[-1]:
            Distance = 0
        else:
            Distance = distance(Suture6_pred[i, :], Suture6_pred[i + 1, :])
        Right_Squamosal_Suture_Length_pred = Right_Squamosal_Suture_Length_pred + Distance
        A, B = Find_point_in_line(Suture6_pred[i, :], Suture4_pred)
        Right_Squamosal_Suture_Width_pred.append(A)
    # Suture 4
    for i in range(DISTANCE_ID2[jjjj[-1]], DISTANCE_ID2[jjjj[0]] + 1):
        if i == DISTANCE_ID2[jjjj[0]]:
            Distance = 0
        else:
            Distance = distance(Suture4_pred[i, :], Suture4_pred[i + 1, :])
        Right_Squamosal_Suture_Length_pred = Right_Squamosal_Suture_Length_pred + Distance
        A, B = Find_point_in_line(Suture4_pred[i, :], Suture6_pred)
        Right_Squamosal_Suture_Width_pred.append(A)
    Right_Squamosal_Suture_Width_pred = np.mean(np.array(Right_Squamosal_Suture_Width_pred))
    Right_Squamosal_Suture_Length_pred = Right_Squamosal_Suture_Length_pred / 2

    ## SI
    # Left SI
    Begining_point_pred = np.mean(
        [Suture5_pred[jjj[0], :], Suture3_pred[DISTANCE_ID1[(np.array(jjj[0])).tolist()], :]], axis=0)
    Ending_point_pred = np.mean(
        [Suture5_pred[jjj[-1], :], Suture3_pred[DISTANCE_ID1[(np.array(jjj[-1])).tolist()], :]], axis=0)
    Left_Squamosal_S_chord_length_pred = distance(Begining_point_pred, Ending_point_pred)
    LSI_pred = Left_Squamosal_Suture_Length_pred / Left_Squamosal_S_chord_length_pred
    # Right SI
    Begining_point_pred = np.mean(
        [Suture6_pred[jjjj[0], :], Suture4_pred[DISTANCE_ID1[(np.array(jjjj[0])).tolist()], :]], axis=0)
    Ending_point_pred = np.mean(
        [Suture6_pred[jjjj[-1], :], Suture4_pred[DISTANCE_ID1[(np.array(jjjj[-1])).tolist()], :]], axis=0)
    Right_Squamosal_S_chord_length_pred = distance(Begining_point_pred, Ending_point_pred)
    RSI_pred = Right_Squamosal_Suture_Length_pred / Right_Squamosal_S_chord_length_pred
    SI_pred = (LSI_pred + RSI_pred) / 2

    return  points , points_pred,Left_Squamosal_Suture_Length_pred, Left_Squamosal_Suture_Width_pred\
        , Right_Squamosal_Suture_Length_pred,Right_Squamosal_Suture_Width_pred,SI_pred

def Parts_LS(PCA_mean_rebuild,pred,context):
    # 5. Lambdoidal suture
    # Left
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append(
            [PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append(
            [PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # Left Lambdoid Suture
    Suture7 = []
    for i in range(100):
        Suture7.append(
            [PCA_mean_rebuild[3 * i + 2100], PCA_mean_rebuild[3 * i + 2101], PCA_mean_rebuild[3 * i + 2102]])
    Suture7 = np.array(Suture7)
    DISTANCE_L = []
    DISTANCE_IDL = []
    DIFF_L = []
    j_L = []

    Suture3_pred = []
    Suture4_pred = []
    for i in range(200):
        Suture3_pred.append(
            [pred[3 * i + 600], pred[3 * i + 601], pred[3 * i + 602]])
        Suture4_pred.append(
            [pred[3 * i + 1200], pred[3 * i + 1201], pred[3 * i + 1202]])
    Suture3_pred = np.array(Suture3_pred)
    Suture4_pred = np.array(Suture4_pred)
    # Left Lambdoid Suture
    Suture7_pred = []
    for i in range(100):
        Suture7_pred.append(
            [pred[3 * i + 2100], pred[3 * i + 2101], pred[3 * i + 2102]])
    Suture7_pred = np.array(Suture7_pred)

    for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
        POINT_L = Suture3[i, :]
        temp1_L, POINTB_L = Find_point_in_line(POINT_L, Suture7)
        DISTANCE_L.append(temp1_L)
        DISTANCE_IDL.append(POINTB_L)
        if i > 0:
            diff111 = DISTANCE_L[i] - DISTANCE_L[i - 1]
            DIFF_L.append(diff111)
    DISTANCE_L = np.array(DISTANCE_L)
    DISTANCE_IDL = np.array(DISTANCE_IDL)
    DIFF_L = np.array(DIFF_L)
    for i in range(199):
        if DISTANCE_L[i] < 10:
            if abs(DIFF_L[i]) < 0.4:
                j_L.append(i)
    j_L = [j_L[0], j_L[-1]]

    Left_lambdoidal_area_points = []
    Left_lambdoidal_area_points_pred = []
    for i in range(j_L[0], j_L[-1] + 1):
        Left_lambdoidal_area_points.append(Suture3[i, :])
        Left_lambdoidal_area_points_pred.append(Suture3_pred[i, :])
    for i in range(DISTANCE_IDL[j_L[-1]], DISTANCE_IDL[j_L[0]] + 1):
        Left_lambdoidal_area_points.append(Suture7[i, :])
        Left_lambdoidal_area_points_pred.append(Suture7_pred[i, :])


    Left_lambdoidal_area_points = np.array(Left_lambdoidal_area_points)
    Left_lambdoidal_area_points_pred = np.array(Left_lambdoidal_area_points_pred)


    # Right lambdoidal suture
    DISTANCE_R = []
    DISTANCE_IDR = []
    DIFF_R = []
    j_R = []
    for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
        POINT_R = Suture4[i, :]
        temp1_R, POINTB_R = Find_point_in_line(POINT_R, Suture7)
        DISTANCE_R.append(temp1_R)
        DISTANCE_IDR.append(POINTB_R)
        if i > 0:
            diff111 = DISTANCE_R[i] - DISTANCE_R[i - 1]
            DIFF_R.append(diff111)
    DISTANCE_R = np.array(DISTANCE_R)
    DISTANCE_IDR = np.array(DISTANCE_IDR)
    DIFF_R = np.array(DIFF_R)
    for i in range(199):
        if DISTANCE_R[i] < 10:
            if abs(DIFF_R[i]) < 0.4:
                j_R.append(i)
    j_R = [j_R[0], j_R[-1]]

    Right_lambdoidal_area_points = []
    Right_lambdoidal_area_points_pred = []
    for i in range(j_R[0], j_R[-1] + 1):
        Right_lambdoidal_area_points.append(Suture4[i, :])
        Right_lambdoidal_area_points_pred.append(Suture4_pred[i, :])
    for i in range(DISTANCE_IDR[j_R[-1]], DISTANCE_IDR[j_R[0]] - 1, -1):
        Right_lambdoidal_area_points.append(Suture7[i, :])
        Right_lambdoidal_area_points_pred.append(Suture7_pred[i, :])
    Right_lambdoidal_area_points = np.array(Right_lambdoidal_area_points)
    Right_lambdoidal_area_points_pred = np.array(Right_lambdoidal_area_points_pred)
    points = np.vstack((Right_lambdoidal_area_points,Left_lambdoidal_area_points))
    points_pred = np.vstack((Right_lambdoidal_area_points_pred, Left_lambdoidal_area_points_pred))

    # measurement
    # LS size
    Lambdoid_Suture_Length_pred = 0
    Lambdoid_Suture_Width_pred = []
    # Suture 7
    for i in range(DISTANCE_IDL[j_L[-1]], DISTANCE_IDR[j_R[-1]] + 1):
        if i == DISTANCE_IDR[j_R[-1]]:
            Distance = 0
        else:
            Distance = distance(Suture7_pred[i, :], Suture7_pred[i + 1, :])
        Lambdoid_Suture_Length_pred = Lambdoid_Suture_Length_pred + Distance

        if i >= DISTANCE_IDL[j_L[-1]] and i <= DISTANCE_IDL[j_L[0]]:
            A, B = Find_point_in_line(Suture7_pred[i, :], Suture3_pred)
            Lambdoid_Suture_Width_pred.append(A)
        if i >= DISTANCE_IDR[j_R[0]] and i <= DISTANCE_IDR[j_R[0]]:
            A, B = Find_point_in_line(Suture7_pred[i, :], Suture4_pred)
            Lambdoid_Suture_Width_pred.append(A)
    Lambdoid_Suture_Width_pred = np.mean(np.array(Lambdoid_Suture_Width_pred))

    # Left SI
    Begining_point_pred = np.mean([Suture3_pred[j_L[0], :], Suture7_pred[DISTANCE_IDL[j_L[0]], :]], axis=0)
    Ending_point_pred = np.mean([Suture3_pred[j_L[-1], :], Suture7_pred[DISTANCE_IDL[j_L[-1]], :]], axis=0)
    Left_Lambdoidal_S_chord_length_pred = distance(Begining_point_pred, Ending_point_pred)
    left_length_pred = 0
    for i in range(DISTANCE_IDL[j_L[-1]], DISTANCE_IDL[j_L[0]] + 1):
        if i == DISTANCE_IDL[j_L[0]]:
            Distance = 0
        else:
            Distance = distance(Suture7_pred[i, :], Suture7_pred[i + 1, :])
        left_length_pred = left_length_pred + Distance
    for i in range(j_L[0], j_L[-1] + 1):
        if i == j_L[-1]:
            Distance = 0
        else:
            Distance = distance(Suture3_pred[i, :], Suture3_pred[i + 1, :])
        left_length_pred = left_length_pred + Distance
    left_length_pred = left_length_pred / 2
    LSI_pred = left_length_pred / Left_Lambdoidal_S_chord_length_pred

    # Right SI
    Begining_point_pred = np.mean([Suture4_pred[j_R[0], :], Suture7_pred[DISTANCE_IDR[j_R[0]], :]], axis=0)
    Ending_point_pred = np.mean([Suture4_pred[j_R[-1], :], Suture7_pred[DISTANCE_IDR[j_R[-1]], :]], axis=0)
    Right_Lambdoidal_S_chord_length_pred = distance(Begining_point_pred, Ending_point_pred)
    right_length_pred = 0
    for i in range(DISTANCE_IDR[j_R[0]], DISTANCE_IDR[j_R[-1]] + 1):
        if i == DISTANCE_IDR[j_R[-1]]:
            Distance = 0
        else:
            Distance = distance(Suture7_pred[i, :], Suture7_pred[i + 1, :])
        right_length_pred = right_length_pred + Distance
    for i in range(j_R[0], j_R[-1] + 1):
        if i == j_R[-1]:
            Distance = 0
        else:
            Distance = distance(Suture4_pred[i, :], Suture4_pred[i + 1, :])
        right_length_pred = right_length_pred + Distance
    right_length_pred = right_length_pred / 2
    RSI_pred = right_length_pred / Right_Lambdoidal_S_chord_length_pred
    SI_pred = (LSI_pred + RSI_pred) / 2



    return  points , points_pred,Lambdoid_Suture_Length_pred, Lambdoid_Suture_Width_pred, SI_pred

def Parts_SF(PCA_mean_rebuild,pred,context):
    # Left
    # Left coronal suture
    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # Left Coronal Suture
    DISTANCE = []
    DISTANCE_ID = []
    DIFF = []
    for i in range(np.shape(Suture2)[0]):  # 计算每个2suture的点到suture3的距离
        POINT = Suture2[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID = np.array(DISTANCE_ID)
    DIFF = np.array(DIFF)
    j = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                j.append(i)
    POINT = Suture3[DISTANCE_ID[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture2)
    j.append(POINTB_ID)

    # left squamosal suture
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    for i in range(np.shape(Suture1)[0]):  # 计算每个suture1的点到suture4的距离
        POINT = Suture1[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    jj = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                jj.append(i)
    POINT = Suture4[DISTANCE_ID1[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture1)
    jj.append(POINTB_ID)
    j = [j[0], j[-1]]
    jj = [jj[0], jj[-1]]

    POINT_3 = Suture3[DISTANCE_ID[j[-1]], :]
    Suture5 = []
    Suture6 = []
    for i in range(50):
        Suture5.append([PCA_mean_rebuild[3 * i + 1800], PCA_mean_rebuild[3 * i + 1801], PCA_mean_rebuild[3 * i + 1802]])
        Suture6.append([PCA_mean_rebuild[3 * i + 1950], PCA_mean_rebuild[3 * i + 1951], PCA_mean_rebuild[3 * i + 1952]])
    Suture5 = np.array(Suture5)
    Suture6 = np.array(Suture6)
    useless, POINTB_ID = Find_point_in_line(POINT_3, Suture5)
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    jjj = []
    for i in range(0, 50):  # 计算每个suture5的点到suture3的距离
        POINT = Suture5[i, :]
        temp1, POINTB_ID1 = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID1)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    for i in range(49):
        if DISTANCE[i] < DISTANCE[0] / 2:
            if abs(DIFF[i]) < 0.3:
                jjj.append(i)
            else:
                if i > 0 and abs(DIFF[i] + DIFF[i - 1]) < 0.3:
                    jjj.append(i)

    # Get Left Sphenoid suture points
    Left_Sphenoid_fontanel_points = []
    Left_Sphenoid_fontanel_points_pred = []
    Suture1_pred = []
    Suture2_pred = []
    for i in range(100):
        Suture1_pred.append([pred[3 * i], pred[3 * i + 1], pred[3 * i + 2]])
        Suture2_pred.append([pred[3 * i + 300], pred[3 * i + 301], pred[3 * i + 302]])
    Suture1_pred = np.array(Suture1_pred)
    Suture2_pred = np.array(Suture2_pred)
    Suture3_pred = []
    Suture4_pred = []
    for i in range(200):
        Suture3_pred.append([pred[3 * i + 600], pred[3 * i + 601], pred[3 * i + 602]])
        Suture4_pred.append([pred[3 * i + 1200], pred[3 * i + 1201], pred[3 * i + 1202]])
    Suture3_pred = np.array(Suture3_pred)
    Suture4_pred = np.array(Suture4_pred)
    Suture5_pred = []
    Suture6_pred = []
    for i in range(50):
        Suture5_pred.append([pred[3 * i + 1800], pred[3 * i + 1801], pred[3 * i + 1802]])
        Suture6_pred.append([pred[3 * i + 1950], pred[3 * i + 1951], pred[3 * i + 1952]])
    Suture5_pred = np.array(Suture5_pred)
    Suture6_pred = np.array(Suture6_pred)
    for i in range(j[-1], 100):
        Left_Sphenoid_fontanel_points.append(Suture2[i, :])
        Left_Sphenoid_fontanel_points_pred.append(Suture2_pred[i, :])
    for i in range(0, jjj[0] + 1):
        Left_Sphenoid_fontanel_points.append(Suture5[i, :])
        Left_Sphenoid_fontanel_points_pred.append(Suture5_pred[i, :])
    for i in range(DISTANCE_ID1[jjj[0]], DISTANCE_ID[j[-1]] + 1):
        Left_Sphenoid_fontanel_points.append(Suture3[i, :])
        Left_Sphenoid_fontanel_points_pred.append(Suture3_pred[i, :])

    Left_Sphenoid_fontanel_points = np.array(Left_Sphenoid_fontanel_points)
    Left_Sphenoid_fontanel_points_pred = np.array(Left_Sphenoid_fontanel_points_pred)

    # Right sphenoid fontanel
    # Right coronal suture
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    for i in range(np.shape(Suture1)[0]):  # 计算每个suture1的点到suture4的距离
        POINT = Suture1[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    jj = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                jj.append(i)
    POINT = Suture4[DISTANCE_ID1[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture1)
    jj.append(POINTB_ID)
    # 只保留首尾
    j = [j[0], j[-1]]
    jj = [jj[0], jj[-1]]

    # right squamosal suture
    jjjj = []
    DISTANCE_ID2 = []
    DISTANCE = []
    DIFF = []
    for i in range(0, 50):  # 计算每个suture6的点到suture4的距离
        POINT = Suture6[i, :]
        temp1, POINTB_ID1 = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID2.append(POINTB_ID1)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID2 = np.array(DISTANCE_ID2)
    DIFF = np.array(DIFF)
    for i in range(49):
        if DISTANCE[i] < DISTANCE[0] / 2:
            if abs(DIFF[i]) < 0.3:
                jjjj.append(i)
            else:
                if i > 0 and abs(DIFF[i] + DIFF[i - 1]) < 0.3:
                    jjjj.append(i)

    jjj = [jjj[0], jjj[-1]]
    # print(jjj)
    # print(jjjj)
    jjjj = [jjjj[0], jjjj[-1]]

    Right_sphenoid_fontanel_points = []
    Right_sphenoid_fontanel_points_pred = []
    for i in range(jj[-1], 100):
        Right_sphenoid_fontanel_points.append(Suture1[i, :])
        Right_sphenoid_fontanel_points_pred.append(Suture1_pred[i, :])
    for i in range(0, jjjj[0] + 1):
        Right_sphenoid_fontanel_points.append(Suture6[i, :])
        Right_sphenoid_fontanel_points_pred.append(Suture6_pred[i, :])
    for i in range(DISTANCE_ID2[jjjj[0]], DISTANCE_ID1[jj[-1]] + 1):
        Right_sphenoid_fontanel_points.append(Suture4[i, :])
        Right_sphenoid_fontanel_points_pred.append(Suture4_pred[i, :])

    Right_sphenoid_fontanel_points = np.array(Right_sphenoid_fontanel_points)
    Right_sphenoid_fontanel_points_pred = np.array(Right_sphenoid_fontanel_points_pred)

    points = np.vstack((Right_sphenoid_fontanel_points, Left_Sphenoid_fontanel_points))
    points_pred = np.vstack((Right_sphenoid_fontanel_points_pred, Left_Sphenoid_fontanel_points_pred))

    return points, points_pred

def Parts_MF(PCA_mean_rebuild,pred,context):
    # Left
    # left squamosal suture
    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # Left Coronal Suture
    DISTANCE = []
    DISTANCE_ID = []
    DIFF = []
    for i in range(np.shape(Suture2)[0]):  # 计算每个suture2的点到suture3的距离
        POINT = Suture2[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID = np.array(DISTANCE_ID)
    DIFF = np.array(DIFF)
    j = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                j.append(i)
    POINT = Suture3[DISTANCE_ID[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture2)
    j.append(POINTB_ID)

    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    for i in range(np.shape(Suture1)[0]):  # 计算每个suture1的点到suture4的距离
        POINT = Suture1[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    jj = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                jj.append(i)
    POINT = Suture4[DISTANCE_ID1[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture1)
    jj.append(POINTB_ID)
    j = [j[0], j[-1]]
    jj = [jj[0], jj[-1]]

    POINT_3 = Suture3[DISTANCE_ID[j[-1]], :]
    Suture5 = []
    Suture6 = []
    for i in range(50):
        Suture5.append([PCA_mean_rebuild[3 * i + 1800], PCA_mean_rebuild[3 * i + 1801], PCA_mean_rebuild[3 * i + 1802]])
        Suture6.append([PCA_mean_rebuild[3 * i + 1950], PCA_mean_rebuild[3 * i + 1951], PCA_mean_rebuild[3 * i + 1952]])
    Suture5 = np.array(Suture5)
    Suture6 = np.array(Suture6)
    useless, POINTB_ID = Find_point_in_line(POINT_3, Suture5)
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    jjj = []
    for i in range(0, 50):  # 计算每个suture5的点到suture3的距离
        POINT = Suture5[i, :]
        temp1, POINTB_ID1 = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID1)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    for i in range(49):
        if DISTANCE[i] < DISTANCE[0] / 2:
            if abs(DIFF[i]) < 0.3:
                jjj.append(i)
            else:
                if i > 0 and abs(DIFF[i] + DIFF[i - 1]) < 0.3:
                    jjj.append(i)

    # Left lambdoidal suture
    Suture7 = []
    for i in range(100):
        Suture7.append(
            [PCA_mean_rebuild[3 * i + 2100], PCA_mean_rebuild[3 * i + 2101], PCA_mean_rebuild[3 * i + 2102]])
    Suture7 = np.array(Suture7)
    DISTANCE_L = []
    DISTANCE_IDL = []
    DIFF_L = []
    j_L = []
    for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
        POINT_L = Suture3[i, :]
        temp1_L, POINTB_L = Find_point_in_line(POINT_L, Suture7)
        DISTANCE_L.append(temp1_L)
        DISTANCE_IDL.append(POINTB_L)
        if i > 0:
            diff111 = DISTANCE_L[i] - DISTANCE_L[i - 1]
            DIFF_L.append(diff111)
    DISTANCE_L = np.array(DISTANCE_L)
    DISTANCE_IDL = np.array(DISTANCE_IDL)
    DIFF_L = np.array(DIFF_L)
    for i in range(199):
        if DISTANCE_L[i] < 10:
            if abs(DIFF_L[i]) < 0.4:
                j_L.append(i)
    j_L = [j_L[0], j_L[-1]]

    Left_mastoid_fontanel = []
    Left_mastoid_fontanel_pred = []
    Suture3_pred = []
    Suture4_pred = []
    for i in range(200):
        Suture3_pred.append([pred[3 * i + 600], pred[3 * i + 601], pred[3 * i + 602]])
        Suture4_pred.append([pred[3 * i + 1200], pred[3 * i + 1201], pred[3 * i + 1202]])
    Suture3_pred = np.array(Suture3_pred)
    Suture4_pred = np.array(Suture4_pred)
    Suture5_pred = []
    Suture6_pred = []
    for i in range(50):
        Suture5_pred.append([pred[3 * i + 1800], pred[3 * i + 1801], pred[3 * i + 1802]])
        Suture6_pred.append([pred[3 * i + 1950], pred[3 * i + 1951], pred[3 * i + 1952]])
    Suture5_pred = np.array(Suture5_pred)
    Suture6_pred = np.array(Suture6_pred)

    Suture7_pred = []
    for i in range(100):
        Suture7_pred.append(
            [pred[3 * i + 2100], pred[3 * i + 2101], pred[3 * i + 2102]])
    Suture7_pred = np.array(Suture7_pred)
    for i in range(jjj[-1], 50):
        Left_mastoid_fontanel.append(Suture5[i, :])
        Left_mastoid_fontanel_pred.append(Suture5_pred[i, :])
    for i in range(0, DISTANCE_IDL[j_L[-1]] + 1):
        Left_mastoid_fontanel.append(Suture7[i, :])
        Left_mastoid_fontanel_pred.append(Suture7_pred[i, :])
    for i in range(j_L[-1], DISTANCE_ID1[jjj[-1]] + 1):
        Left_mastoid_fontanel.append(Suture3[i, :])
        Left_mastoid_fontanel_pred.append(Suture3_pred[i, :])
    Left_mastoid_fontanel = np.array(Left_mastoid_fontanel)
    Left_mastoid_fontanel_pred = np.array(Left_mastoid_fontanel_pred)

    # right squamosal suture
    jjjj = []
    DISTANCE_ID2 = []
    DISTANCE = []
    DIFF = []
    for i in range(0, 50):  # 计算每个suture6的点到suture4的距离
        POINT = Suture6[i, :]
        temp1, POINTB_ID1 = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID2.append(POINTB_ID1)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID2 = np.array(DISTANCE_ID2)
    DIFF = np.array(DIFF)
    for i in range(49):
        if DISTANCE[i] < DISTANCE[0] / 2:
            if abs(DIFF[i]) < 0.3:
                jjjj.append(i)
            else:
                if i > 0 and abs(DIFF[i] + DIFF[i - 1]) < 0.3:
                    jjjj.append(i)

    jjj = [jjj[0], jjj[-1]]
    jjjj = [jjjj[0], jjjj[-1]]

    # Right lambdoidal suture
    DISTANCE_R = []
    DISTANCE_IDR = []
    DIFF_R = []
    j_R = []
    for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
        POINT_R = Suture4[i, :]
        temp1_R, POINTB_R = Find_point_in_line(POINT_R, Suture7)
        DISTANCE_R.append(temp1_R)
        DISTANCE_IDR.append(POINTB_R)
        if i > 0:
            diff111 = DISTANCE_R[i] - DISTANCE_R[i - 1]
            DIFF_R.append(diff111)
    DISTANCE_R = np.array(DISTANCE_R)
    DISTANCE_IDR = np.array(DISTANCE_IDR)
    DIFF_R = np.array(DIFF_R)
    for i in range(199):
        if DISTANCE_R[i] < 10:
            if abs(DIFF_R[i]) < 0.4:
                j_R.append(i)
    j_R = [j_R[0], j_R[-1]]

    Right_mastoid_fontanel = []
    Right_mastoid_fontanel_pred = []
    for i in range(jjjj[-1], 50):
        Right_mastoid_fontanel.append(Suture6[i, :])
        Right_mastoid_fontanel_pred.append(Suture6_pred[i, :])
    for i in range(99, DISTANCE_IDR[j_R[-1]] - 1, -1):
        Right_mastoid_fontanel.append(Suture7[i, :])
        Right_mastoid_fontanel_pred.append(Suture7_pred[i, :])
    for i in range(j_R[-1], DISTANCE_ID2[jjjj[-1]] + 1):
        Right_mastoid_fontanel.append(Suture4[i, :])
        Right_mastoid_fontanel_pred.append(Suture4_pred[i, :])

    Right_mastoid_fontanel = np.array(Right_mastoid_fontanel)
    Right_mastoid_fontanel_pred = np.array(Right_mastoid_fontanel_pred)


    points = np.vstack((Right_mastoid_fontanel,Left_mastoid_fontanel))
    points_pred = np.vstack((Right_mastoid_fontanel_pred, Left_mastoid_fontanel_pred))

    return  points , points_pred

def ALL_Suture(PCA_mean_rebuild,context):
    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # Left Coronal Suture
    DISTANCE = []
    DISTANCE_ID = []
    DIFF = []

    for i in range(np.shape(Suture2)[0]):  # 计算每个2suture的点到suture3的距离
        POINT = Suture2[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID = np.array(DISTANCE_ID)
    DIFF = np.array(DIFF)
    j = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                j.append(i)
    POINT = Suture3[DISTANCE_ID[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture2)
    j.append(POINTB_ID)
    # marker_X = np.array([Suture2[j, 0], Suture3[DISTANCE_ID[j], 0]])
    # marker_Y = np.array([Suture2[j, 1], Suture3[DISTANCE_ID[j], 1]])
    # marker_Z = np.array([Suture2[j, 2], Suture3[DISTANCE_ID[j], 2]])
    # Visualization.single_line_nodes(PCA_mean_rebuild, 0, 800, context, marker_X, marker_Y, marker_Z)
    # plt.show()

    # Right Coronal Suture
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    for i in range(np.shape(Suture1)[0]):  # 计算每个suture1的点到suture4的距离
        POINT = Suture1[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    jj = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                jj.append(i)
    POINT = Suture4[DISTANCE_ID1[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture1)
    jj.append(POINTB_ID)
    j = [j[0], j[-1]]
    jj = [jj[0], jj[-1]]
    marker_X = np.hstack((Suture1[0,0],Suture2[0,0],Suture2[j, 0], Suture3[DISTANCE_ID[j], 0], Suture1[jj, 0], Suture4[DISTANCE_ID1[jj], 0]))
    marker_Y = np.hstack((Suture1[0,1],Suture2[0,1],Suture2[j, 1], Suture3[DISTANCE_ID[j], 1], Suture1[jj, 1], Suture4[DISTANCE_ID1[jj], 1]))
    marker_Z = np.hstack((Suture1[0,2],Suture2[0,2],Suture2[j, 2], Suture3[DISTANCE_ID[j], 2], Suture1[jj, 2], Suture4[DISTANCE_ID1[jj], 2]))
    # Visualization.single_line_nodes(PCA_mean_rebuild, 0, 800, 's', marker_X, marker_Y, marker_Z)
    # plt.show()
    # Find Left Squamosal Suture
    POINT_3 = Suture3[DISTANCE_ID[j[-1]], :]
    Suture5 = []
    Suture6 = []
    for i in range(50):
        Suture5.append([PCA_mean_rebuild[3 * i + 1800], PCA_mean_rebuild[3 * i + 1801], PCA_mean_rebuild[3 * i + 1802]])
        Suture6.append([PCA_mean_rebuild[3 * i + 1950], PCA_mean_rebuild[3 * i + 1951], PCA_mean_rebuild[3 * i + 1952]])
    Suture5 = np.array(Suture5)
    Suture6 = np.array(Suture6)
    useless, POINTB_ID = Find_point_in_line(POINT_3, Suture5)
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    jjj = []
    for i in range(0, 50):  # 计算每个suture5的点到suture3的距离
        POINT = Suture5[i, :]
        temp1, POINTB_ID1 = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID1)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    for i in range(49):
        if DISTANCE[i] < DISTANCE[0] / 2:
            if abs(DIFF[i]) < 0.3:
                jjj.append(i)
            else:
                if i > 0 and abs(DIFF[i] + DIFF[i - 1]) < 0.3:
                    jjj.append(i)
    # Initialization
    jjjj = []
    DISTANCE_ID2 = []
    DISTANCE = []
    DIFF = []
    for i in range(0, 50):  # 计算每个suture6的点到suture4的距离
        POINT = Suture6[i, :]
        temp1, POINTB_ID1 = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID2.append(POINTB_ID1)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID2 = np.array(DISTANCE_ID2)
    DIFF = np.array(DIFF)
    for i in range(49):
        if DISTANCE[i] < DISTANCE[0] / 2:
            if abs(DIFF[i]) < 0.3:
                jjjj.append(i)
            else:
                if i > 0 and abs(DIFF[i] + DIFF[i - 1]) < 0.3:
                    jjjj.append(i)

    jjj = [jjj[0], jjj[-1]]
    jjjj = [jjjj[0], jjjj[-1]]
    marker_X = np.hstack((marker_X, Suture5[jjj, 0], Suture3[DISTANCE_ID1[(np.array(jjj)).tolist()], 0],
                          Suture6[jjjj, 0], Suture4[DISTANCE_ID2[(np.array(jjjj)).tolist()], 0]))
    marker_Y = np.hstack((marker_Y, Suture5[jjj, 1], Suture3[DISTANCE_ID1[(np.array(jjj)).tolist()], 1],
                          Suture6[jjjj, 1], Suture4[DISTANCE_ID2[(np.array(jjjj)).tolist()], 1]))
    marker_Z = np.hstack((marker_Z, Suture5[jjj, 2], Suture3[DISTANCE_ID1[(np.array(jjj)).tolist()], 2],
                          Suture6[jjjj, 2], Suture4[DISTANCE_ID2[(np.array(jjjj)).tolist()], 2]))

    # Saggital Suture
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # 2. Get the distance between two nodes
    Temp = []
    for i in range(200):
        Temp.append(distance(Suture3[i, :], Suture4[i, :]))
    Temp = np.array(Temp)
    diff = []
    j = []
    for i in range(99):
        diff.append(Temp[i + 1] - Temp[i])
        if Temp[1] > 15:
            if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and Temp[i + 2] > Temp[i + 1] and Temp[i] < 25:
                j.append(i)
        else:
            if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and Temp[i + 2] > Temp[i + 1] and Temp[i] < 10:
                j.append(i)
    marker_X = np.hstack((marker_X,Suture3[0, 0], Suture4[0, 0], Suture3[j[-1], 0], Suture4[j[-1], 0]))
    marker_Y = np.hstack((marker_Y,Suture3[0, 1], Suture4[0, 1], Suture3[j[-1], 1], Suture4[j[-1], 1]))
    marker_Z = np.hstack((marker_Z,Suture3[0, 2], Suture4[0, 2], Suture3[j[-1], 2], Suture4[j[-1], 2]))

    # Visualization.single_line_nodes(PCA_mean_rebuild, 0, 800, 's', marker_X, marker_Y, marker_Z)
    # plt.show()

    # Left Lambdoid Suture
    Suture7 = []
    for i in range(100):
        Suture7.append([PCA_mean_rebuild[3 * i + 2100], PCA_mean_rebuild[3 * i + 2101], PCA_mean_rebuild[3 * i + 2102]])
    Suture7 = np.array(Suture7)
    DISTANCE_L = []
    DISTANCE_IDL = []
    DIFF_L = []
    j_L = []
    for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
        POINT_L = Suture3[i, :]
        temp1_L, POINTB_L = Find_point_in_line(POINT_L, Suture7)
        DISTANCE_L.append(temp1_L)
        DISTANCE_IDL.append(POINTB_L)
        if i > 0:
            diff111 = DISTANCE_L[i] - DISTANCE_L[i-1]
            DIFF_L.append(diff111)
    DISTANCE_L = np.array(DISTANCE_L)
    DISTANCE_IDL = np.array(DISTANCE_IDL)
    DIFF_L = np.array(DIFF_L)
    for i in range(199):
        if DISTANCE_L[i] < 10:
            if abs(DIFF_L[i]) < 0.4:
                j_L.append(i)
    j_L = [j_L[0],j_L[-1]]
    marker_X = np.hstack((marker_X, Suture3[j_L,0],Suture7[DISTANCE_IDL[j_L],0]))
    marker_Y = np.hstack((marker_Y, Suture3[j_L,1],Suture7[DISTANCE_IDL[j_L],1]))
    marker_Z = np.hstack((marker_Z, Suture3[j_L,2],Suture7[DISTANCE_IDL[j_L],2]))
    # Visualization.single_line_nodes(PCA_mean_rebuild, 0, 800, context, marker_X, marker_Y, marker_Z)
    # plt.show()

    # Right Lambdoid Suture
    DISTANCE_R = []
    DISTANCE_IDR = []
    DIFF_R = []
    j_R = []
    for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
        POINT_R = Suture4[i, :]
        temp1_R, POINTB_R = Find_point_in_line(POINT_R, Suture7)
        DISTANCE_R.append(temp1_R)
        DISTANCE_IDR.append(POINTB_R)
        if i > 0:
            diff111 = DISTANCE_R[i] - DISTANCE_R[i-1]
            DIFF_R.append(diff111)
    DISTANCE_R = np.array(DISTANCE_R)
    DISTANCE_IDR = np.array(DISTANCE_IDR)
    DIFF_R = np.array(DIFF_R)
    for i in range(199):
        if DISTANCE_R[i] < 10:
            if abs(DIFF_R[i]) < 0.4:
                j_R.append(i)
    j_R = [j_R[0],j_R[-1]]
    marker_X = np.hstack((marker_X, Suture4[j_R,0],Suture7[DISTANCE_IDR[j_R],0]))
    marker_Y = np.hstack((marker_Y, Suture4[j_R,1],Suture7[DISTANCE_IDR[j_R],1]))
    marker_Z = np.hstack((marker_Z, Suture4[j_R,2],Suture7[DISTANCE_IDR[j_R],2]))
    # Visualization.single_line_nodes(PCA_mean_rebuild, 0, 800, context, marker_X, marker_Y, marker_Z)
    # plt.show()

    # Metopic suture
    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)
    # 2. Get the distance between two nodes
    Temp = []
    for i in range(100):
        Temp.append(distance(Suture1[i, :], Suture2[i, :]))
    Temp = np.array(Temp)
    diff = []
    j = []
    for i in range(99):
        diff.append(Temp[i + 1] - Temp[i])
        if Temp[1] > 5:
            if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and diff[i] < 1 and Temp[i] < 12:
                j.append(i)
        else:
            if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and diff[i] < 1 and Temp[i] < 3:
                j.append(i)
    # ALL SUTURES
    marker_X = np.hstack((marker_X,Suture1[j[-1], 0], Suture2[j[-1], 0]))
    marker_Y = np.hstack((marker_Y,Suture1[j[-1], 1], Suture2[j[-1], 1]))
    marker_Z = np.hstack((marker_Z,Suture1[j[-1], 2], Suture2[j[-1], 2]))
    #
    Visualization.single_line_nodes(PCA_mean_rebuild, 0, 800, context, marker_X, marker_Y, marker_Z)
    plt.show()
    # return marker_X, marker_Y,marker_Z

def Lambdoid_Suture(PCA_mean_rebuild,context):
        Suture3 = []
        Suture4 = []
        for i in range(200):
            Suture3.append(
                [PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
            Suture4.append(
                [PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
        Suture3 = np.array(Suture3)
        Suture4 = np.array(Suture4)
        # Left Lambdoid Suture
        Suture7 = []
        for i in range(100):
            Suture7.append(
                [PCA_mean_rebuild[3 * i + 2100], PCA_mean_rebuild[3 * i + 2101], PCA_mean_rebuild[3 * i + 2102]])
        Suture7 = np.array(Suture7)
        DISTANCE_L = []
        DISTANCE_IDL = []
        DIFF_L = []
        j_L = []
        for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
            POINT_L = Suture3[i, :]
            temp1_L, POINTB_L = Find_point_in_line(POINT_L, Suture7)
            DISTANCE_L.append(temp1_L)
            DISTANCE_IDL.append(POINTB_L)
            if i > 0:
                diff111 = DISTANCE_L[i] - DISTANCE_L[i - 1]
                DIFF_L.append(diff111)
        DISTANCE_L = np.array(DISTANCE_L)
        DISTANCE_IDL = np.array(DISTANCE_IDL)
        DIFF_L = np.array(DIFF_L)
        for i in range(199):
            if DISTANCE_L[i] < 10:
                if abs(DIFF_L[i]) < 0.4:
                    j_L.append(i)
        j_L = [j_L[0], j_L[-1]]
        marker_X = np.hstack(( Suture3[j_L, 0], Suture7[DISTANCE_IDL[j_L], 0]))
        marker_Y = np.hstack(( Suture3[j_L, 1], Suture7[DISTANCE_IDL[j_L], 1]))
        marker_Z = np.hstack(( Suture3[j_L, 2], Suture7[DISTANCE_IDL[j_L], 2]))
        # Visualization.single_line_nodes(PCA_mean_rebuild, 0, 800, context, marker_X, marker_Y, marker_Z)
        # plt.show()

        # Right Lambdoid Suture
        DISTANCE_R = []
        DISTANCE_IDR = []
        DIFF_R = []
        j_R = []
        for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
            POINT_R = Suture4[i, :]
            temp1_R, POINTB_R = Find_point_in_line(POINT_R, Suture7)
            DISTANCE_R.append(temp1_R)
            DISTANCE_IDR.append(POINTB_R)
            if i > 0:
                diff111 = DISTANCE_R[i] - DISTANCE_R[i - 1]
                DIFF_R.append(diff111)
        DISTANCE_R = np.array(DISTANCE_R)
        DISTANCE_IDR = np.array(DISTANCE_IDR)
        DIFF_R = np.array(DIFF_R)
        for i in range(199):
            if DISTANCE_R[i] < 10:
                if abs(DIFF_R[i]) < 0.4:
                    j_R.append(i)
        j_R = [j_R[0], j_R[-1]]
        marker_X = np.hstack((marker_X, Suture4[j_R, 0], Suture7[DISTANCE_IDR[j_R], 0]))
        marker_Y = np.hstack((marker_Y, Suture4[j_R, 1], Suture7[DISTANCE_IDR[j_R], 1]))
        marker_Z = np.hstack((marker_Z, Suture4[j_R, 2], Suture7[DISTANCE_IDR[j_R], 2]))
        # Visualization.single_line_nodes(PCA_mean_rebuild, 0, 800, context, marker_X, marker_Y, marker_Z)
        # plt.show()

        # measurement
        Lambdoid_Suture_Length = 0
        Lambdoid_Suture_Width = []
        # Suture 7
        for i in range(DISTANCE_IDL[j_L[-1]], DISTANCE_IDR[j_R[-1]] + 1):
            if i == DISTANCE_IDR[j_R[-1]]:
                Distance = 0
            else:
                Distance = distance(Suture7[i, :], Suture7[i + 1, :])
            Lambdoid_Suture_Length = Lambdoid_Suture_Length + Distance

            if i >=DISTANCE_IDL[j_L[-1]] and i <= DISTANCE_IDL[j_L[0]]:
                A, B = Find_point_in_line(Suture7[i, :], Suture3)
                Lambdoid_Suture_Width.append(A)
            if i >= DISTANCE_IDR[j_R[0]] and i<=DISTANCE_IDR[j_R[0]]:
                A, B = Find_point_in_line(Suture7[i, :], Suture4)
                Lambdoid_Suture_Width.append(A)
        Lambdoid_Suture_Width = np.mean(np.array(Lambdoid_Suture_Width))

        # Left SI
        Begining_point = np.mean([Suture3[j_L[0], :], Suture7[DISTANCE_IDL[j_L[0]], :]], axis=0)
        Ending_point = np.mean([Suture3[j_L[-1], :], Suture7[DISTANCE_IDL[j_L[-1]], :]], axis=0)
        Left_Lambdoidal_S_chord_length = distance(Begining_point, Ending_point)
        left_length = 0
        for i in range(DISTANCE_IDL[j_L[-1]], DISTANCE_IDL[j_L[0]] + 1):
            if i == DISTANCE_IDL[j_L[0]]:
                Distance = 0
            else:
                Distance = distance(Suture7[i, :], Suture7[i + 1, :])
            left_length = left_length + Distance
        for i in range(j_L[0], j_L[-1] + 1):
            if i == j_L[-1]:
                Distance = 0
            else:
                Distance = distance(Suture3[i, :], Suture3[i + 1, :])
            left_length = left_length + Distance
        left_length = left_length / 2
        LSI = left_length / Left_Lambdoidal_S_chord_length

        # Right SI
        Begining_point = np.mean([Suture4[j_R[0], :], Suture7[DISTANCE_IDR[j_R[0]], :]], axis=0)
        Ending_point = np.mean([Suture4[j_R[-1], :], Suture7[DISTANCE_IDR[j_R[-1]], :]], axis=0)
        Right_Lambdoidal_S_chord_length = distance(Begining_point, Ending_point)
        right_length = 0
        for i in range(DISTANCE_IDR[j_R[0]], DISTANCE_IDR[j_R[-1]] + 1):
            if i == DISTANCE_IDR[j_R[-1]]:
                Distance = 0
            else:
                Distance = distance(Suture7[i, :], Suture7[i + 1, :])
            right_length = right_length + Distance
        for i in range(j_R[0], j_R[-1] + 1):
            if i == j_R[-1]:
                Distance = 0
            else:
                Distance = distance(Suture4[i, :], Suture4[i + 1, :])
            right_length = right_length + Distance
        right_length = right_length / 2
        RSI = right_length / Right_Lambdoidal_S_chord_length
        SI = (LSI + RSI) / 2
        return Lambdoid_Suture_Length, Lambdoid_Suture_Width, SI

def Anterior_fontanel_area(PCA_mean_rebuild,ID,parameter):
    context = 'anterior fontanelle'
    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)
    # 2. Get the distance between two nodes
    Temp = []
    for i in range(100):
        Temp.append(distance(Suture1[i, :], Suture2[i, :]))
    Temp = np.array(Temp)
    diff = []
    j = []
    for i in range(99):
        diff.append(Temp[i + 1] - Temp[i])
        if Temp[1] > 5:
            if diff[i] > 0.15 and diff[i] > 1.5 * diff[i - 1] and diff[i] < 1 and Temp[i] < 12:
                j.append(i)
        else:
            # if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and diff[i] < 1 and Temp[i] < 3:   OLD
            if diff[i] > 0.15 and diff[i] > 1.5 * diff[i - 1] and diff[i] < 1 and Temp[i] < 3:
                # if i>1 and diff[i-1] < 1.5* diff[i-2]:
                j.append(i)
    marker_X = np.array([Suture1[j[-1], 0], Suture2[j[-1], 0]])
    marker_Y = np.array([Suture1[j[-1], 1], Suture2[j[-1], 1]])
    marker_Z = np.array([Suture1[j[-1], 2], Suture2[j[-1], 2]])
    diff = np.array(diff)
    Metopic_point_ID = j[-1]

    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # Left Coronal Suture
    DISTANCE = []
    DISTANCE_ID = []
    DIFF = []
    for i in range(np.shape(Suture2)[0]):  # 计算每个2suture的点到suture3的距离
        POINT = Suture2[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID = np.array(DISTANCE_ID)
    DIFF = np.array(DIFF)
    j = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                j.append(i)
    POINT = Suture3[DISTANCE_ID[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture2)
    j.append(POINTB_ID)

    # Right Coronal Suture
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    for i in range(np.shape(Suture1)[0]):  # 计算每个suture1的点到suture4的距离
        POINT = Suture1[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    jj = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                jj.append(i)
    POINT = Suture4[DISTANCE_ID1[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture1)
    jj.append(POINTB_ID)
    # 只保留首尾
    j = [j[0], j[-1]]
    jj = [jj[0], jj[-1]]

    Left_Coronal_point_ID2 = j[0]
    Left_Coronal_point_ID3 = DISTANCE_ID[j[0]]
    Right_Coronal_point_ID6 = DISTANCE_ID1[jj[0]]
    Right_Coronal_point_ID7 = jj[0]

    # Get anterior fontanel points
    Anterior_fontanel_points = []
    for i in range(Metopic_point_ID, Left_Coronal_point_ID2 + 1):
        Anterior_fontanel_points.append(Suture2[i, :])
    for i in range(Left_Coronal_point_ID3, 200):
        Anterior_fontanel_points.append(Suture3[i, :])
    for i in range(199, Right_Coronal_point_ID6 - 1, -1):
        Anterior_fontanel_points.append(Suture4[i, :])
    for i in range(Right_Coronal_point_ID7, Metopic_point_ID - 1, -1):
        Anterior_fontanel_points.append(Suture1[i, :])
    Anterior_fontanel_points.append(Suture2[Metopic_point_ID, :])
    Anterior_fontanel_points = np.array(Anterior_fontanel_points)

    points = Anterior_fontanel_points
    import matlab.engine

    engine = matlab.engine.start_matlab()  # start matlab process
    # engine = matlab.engine.start_matlab("-desktop")  # start matlab process with graphic UI
    # needs to transfer the data format from python to matlab
    m_points = matlab.double(points.tolist())
    Anterior_fontanel_area = engine.Area_ZX_plane_anterior(m_points, 0,parameter)

    return Anterior_fontanel_area

def Posterior_fontanel_area(PCA_mean_rebuild,ID,parameter):
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)

    # 2. Get the distance between two nodes ##### Here find the smallest distance
    Temp = []
    DISTANCE_ID = []
    for i in range(200):
        POINT = Suture3[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        Temp.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        #####################################
        # Temp.append(distance(Suture3[i, :], Suture4[i, :]))
    Temp = np.array(Temp)
    diff = []
    j = []
    for i in range(99):
        diff.append(Temp[i + 1] - Temp[i])
        if Temp[1] > 14.5:
            if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and Temp[i + 2] > Temp[i + 1] and Temp[i] < 24.5:
                j.append(i)
        else:
            if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and Temp[i + 2] > Temp[i + 1] and Temp[i] < 8:
                j.append(i)
    Id_4 = j[-1]
    Id_5 = DISTANCE_ID[j[-1]]

    # Left Lambdoid Suture
    Suture7 = []
    for i in range(100):
        Suture7.append(
            [PCA_mean_rebuild[3 * i + 2100], PCA_mean_rebuild[3 * i + 2101], PCA_mean_rebuild[3 * i + 2102]])
    Suture7 = np.array(Suture7)
    DISTANCE_L = []
    DISTANCE_IDL = []
    DIFF_L = []
    j_L = []
    for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
        POINT_L = Suture3[i, :]
        temp1_L, POINTB_L = Find_point_in_line(POINT_L, Suture7)
        DISTANCE_L.append(temp1_L)
        DISTANCE_IDL.append(POINTB_L)
        if i > 0:
            diff111 = DISTANCE_L[i] - DISTANCE_L[i - 1]
            DIFF_L.append(diff111)
    DISTANCE_L = np.array(DISTANCE_L)
    DISTANCE_IDL = np.array(DISTANCE_IDL)
    DIFF_L = np.array(DIFF_L)
    for i in range(199):
        if DISTANCE_L[i] < 10:
            if abs(DIFF_L[i]) < 0.4:
                j_L.append(i)
    j_L = [j_L[0], j_L[-1]]
    ID_1 = DISTANCE_IDL[j_L[0]]
    ID_6 = j_L[0]
    # Right Lambdoid Suture
    DISTANCE_R = []
    DISTANCE_IDR = []
    DIFF_R = []
    j_R = []
    for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
        POINT_R = Suture4[i, :]
        temp1_R, POINTB_R = Find_point_in_line(POINT_R, Suture7)
        DISTANCE_R.append(temp1_R)
        DISTANCE_IDR.append(POINTB_R)
        if i > 0:
            diff111 = DISTANCE_R[i] - DISTANCE_R[i - 1]
            DIFF_R.append(diff111)
    DISTANCE_R = np.array(DISTANCE_R)
    DISTANCE_IDR = np.array(DISTANCE_IDR)
    DIFF_R = np.array(DIFF_R)
    for i in range(199):
        if DISTANCE_R[i] < 10:
            if abs(DIFF_R[i]) < 0.4:
                j_R.append(i)
    j_R = [j_R[0], j_R[-1]]
    ID_2 = DISTANCE_IDR[j_R[0]]
    ID_3 = j_R[0]
    # Get posterior fontanel points
    Posterior_fontanel_points = []
    for i in range(ID_1, ID_2 + 1):
        Posterior_fontanel_points.append(Suture7[i, :])
    for i in range(ID_3, Id_5 - 1, -1):
        Posterior_fontanel_points.append(Suture4[i, :])
    for i in range(Id_4, ID_6 + 1):
        Posterior_fontanel_points.append(Suture3[i, :])
    Posterior_fontanel_points.append(Suture7[ID_1, :])
    Posterior_fontanel_points = np.array(Posterior_fontanel_points)
    points = Posterior_fontanel_points
    import matlab.engine

    engine = matlab.engine.start_matlab()  # start matlab process
    # engine = matlab.engine.start_matlab("-desktop")  # start matlab process with graphic UI
    # needs to transfer the data format from python to matlab
    m_points = matlab.double(points.tolist())
    Posterior_fontanel_area = engine.Area_ZX_plane_posterior(m_points, 0,parameter)

    return Posterior_fontanel_area

def Area_XY_plane(PCA_mean_rebuild,ID):
    context = 'PF'
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # 2. Get the distance between two nodes ##### Here find the smallest distance
    Temp = []
    DISTANCE_ID = []
    for i in range(200):
        POINT = Suture3[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        Temp.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        #####################################
        # Temp.append(distance(Suture3[i, :], Suture4[i, :]))
    Temp = np.array(Temp)
    diff = []
    j = []
    for i in range(99):
        diff.append(Temp[i + 1] - Temp[i])
        if Temp[1] > 14.5:
            if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and Temp[i + 2] > Temp[i + 1] and Temp[i] < 24.5:
                j.append(i)
        else:
            if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and Temp[i + 2] > Temp[i + 1] and Temp[i] < 8:
                j.append(i)
    Id_4 = j[-1]
    Id_5 = DISTANCE_ID[j[-1]]
    ########################################
    # Id_5 = Id_4
    ##############################
    # marker_X = np.hstack((Suture3[Id_4, 0], Suture4[Id_5, 0]))
    # marker_Y = np.hstack((Suture3[Id_4, 1], Suture4[Id_5, 1]))
    # marker_Z = np.hstack((Suture3[Id_4, 2], Suture4[Id_5, 2]))
    # Visualization.single_line_nodes(PCA_mean_rebuild, 0, 800, context, marker_X, marker_Y, marker_Z)
    # plt.show()

    # Posterior_fontanel_area = np.zeros((69,1))
    # for i in range(0,10):
    #     Posterior_fontanel_area [i,0] = Suture_analysis.Posterior_fontanel_area (new_data[i,:],i)
    XY_Z_points = []
    for i in range(0, Id_4 + 1):
        XY_Z_points.append(Suture3[i, :])
    for i in range(Id_5, -1, -1):
        XY_Z_points.append(Suture4[i, :])
    XY_Z_points.append(Suture3[0, :])
    XY_Z_points = np.array(XY_Z_points)
    points = XY_Z_points
    import matlab.engine

    engine = matlab.engine.start_matlab()  # start matlab process
    # engine = matlab.engine.start_matlab("-desktop")  # start matlab process with graphic UI
    # needs to transfer the data format from python to matlab
    m_points = matlab.double(points.tolist())
    area = engine.Area_XY_plane(m_points,ID)

    return area

def Area_ZX_plane_anterior(PCA_mean_rebuild,ID):
    context = 'SI'
    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # Left Coronal Suture
    DISTANCE = []
    DISTANCE_ID = []
    DIFF = []
    for i in range(np.shape(Suture2)[0]):  # 计算每个2suture的点到suture3的距离
        POINT = Suture2[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID = np.array(DISTANCE_ID)
    DIFF = np.array(DIFF)
    j = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                j.append(i)
    POINT = Suture3[DISTANCE_ID[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture2)
    j.append(POINTB_ID)
    # marker_X = np.array([Suture2[j, 0], Suture3[DISTANCE_ID[j], 0]])
    # marker_Y = np.array([Suture2[j, 1], Suture3[DISTANCE_ID[j], 1]])
    # marker_Z = np.array([Suture2[j, 2], Suture3[DISTANCE_ID[j], 2]])
    # Visualization.single_line_nodes(PCA_mean_rebuild, 0, 800, context, marker_X, marker_Y, marker_Z)
    # plt.show()

    # Right Coronal Suture
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    for i in range(np.shape(Suture1)[0]):  # 计算每个suture1的点到suture4的距离
        POINT = Suture1[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    jj = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                jj.append(i)
    POINT = Suture4[DISTANCE_ID1[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture1)
    jj.append(POINTB_ID)
    # 只保留首尾
    j = [j[0], j[-1]]
    jj = [jj[0], jj[-1]]
    # marker_X = np.hstack((Suture2[j, 0], Suture3[DISTANCE_ID[j], 0], Suture1[jj, 0], Suture4[DISTANCE_ID1[jj], 0]))
    # marker_Y = np.hstack((Suture2[j, 1], Suture3[DISTANCE_ID[j], 1], Suture1[jj, 1], Suture4[DISTANCE_ID1[jj], 1]))
    # marker_Z = np.hstack((Suture2[j, 2], Suture3[DISTANCE_ID[j], 2], Suture1[jj, 2], Suture4[DISTANCE_ID1[jj], 2]))
    # Visualization.single_line_nodes(PCA_mean_rebuild, 0, 800, context, marker_X, marker_Y, marker_Z)
    # plt.show()

    Left_Coronal_point_ID2 = j[0]
    Left_Coronal_point_ID3 = DISTANCE_ID[j[0]]
    Right_Coronal_point_ID6 = DISTANCE_ID1[jj[0]]
    Right_Coronal_point_ID7 = jj[0]

    # Get anterior fontanel points
    Anterior_ZX_plane_points = []
    for i in range(0, Left_Coronal_point_ID2 + 1):
        Anterior_ZX_plane_points.append(Suture2[i, :])
    for i in range(Left_Coronal_point_ID3, 200):
        Anterior_ZX_plane_points.append(Suture3[i, :])
    for i in range(199, Right_Coronal_point_ID6 - 1, -1):
        Anterior_ZX_plane_points.append(Suture4[i, :])
    for i in range(Right_Coronal_point_ID7, -1, -1):
        Anterior_ZX_plane_points.append(Suture1[i, :])
    Anterior_ZX_plane_points.append(Suture2[0, :])
    Anterior_ZX_plane_points = np.array(Anterior_ZX_plane_points)

    points = Anterior_ZX_plane_points
    import matlab.engine

    engine = matlab.engine.start_matlab()  # start matlab process
    # engine = matlab.engine.start_matlab("-desktop")  # start matlab process with graphic UI
    # needs to transfer the data format from python to matlab
    m_points = matlab.double(points.tolist())
    area = engine.Area_ZX_plane_anterior(m_points,ID)

    return area

def Area_ZX_plane_posterior(PCA_mean_rebuild,ID):
    context = 'SI'
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append(
            [PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append(
            [PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # Left Lambdoid Suture
    Suture7 = []
    for i in range(100):
        Suture7.append(
            [PCA_mean_rebuild[3 * i + 2100], PCA_mean_rebuild[3 * i + 2101], PCA_mean_rebuild[3 * i + 2102]])
    Suture7 = np.array(Suture7)
    DISTANCE_L = []
    DISTANCE_IDL = []
    DIFF_L = []
    j_L = []
    for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
        POINT_L = Suture3[i, :]
        temp1_L, POINTB_L = Find_point_in_line(POINT_L, Suture7)
        DISTANCE_L.append(temp1_L)
        DISTANCE_IDL.append(POINTB_L)
        if i > 0:
            diff111 = DISTANCE_L[i] - DISTANCE_L[i - 1]
            DIFF_L.append(diff111)
    DISTANCE_L = np.array(DISTANCE_L)
    DISTANCE_IDL = np.array(DISTANCE_IDL)
    DIFF_L = np.array(DIFF_L)
    for i in range(199):
        if DISTANCE_L[i] < 10:
            if abs(DIFF_L[i]) < 0.4:
                j_L.append(i)
    j_L = [j_L[0], j_L[-1]]
    ID_1 = j_L[-1]
    ID_2 = DISTANCE_IDL[j_L[-1]]
    # Visualization.single_line_nodes(PCA_mean_rebuild, 0, 800, context, marker_X, marker_Y, marker_Z)
    # plt.show()

    # Right Lambdoid Suture
    DISTANCE_R = []
    DISTANCE_IDR = []
    DIFF_R = []
    j_R = []
    for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
        POINT_R = Suture4[i, :]
        temp1_R, POINTB_R = Find_point_in_line(POINT_R, Suture7)
        DISTANCE_R.append(temp1_R)
        DISTANCE_IDR.append(POINTB_R)
        if i > 0:
            diff111 = DISTANCE_R[i] - DISTANCE_R[i - 1]
            DIFF_R.append(diff111)
    DISTANCE_R = np.array(DISTANCE_R)
    DISTANCE_IDR = np.array(DISTANCE_IDR)
    DIFF_R = np.array(DIFF_R)
    for i in range(199):
        if DISTANCE_R[i] < 10:
            if abs(DIFF_R[i]) < 0.4:
                j_R.append(i)
    j_R = [j_R[0], j_R[-1]]

    ID_3 = DISTANCE_IDR[j_R[-1]]
    ID_4 = j_R[-1]

    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # 2. Get the distance between two nodes ##### Here find the smallest distance
    Temp = []
    DISTANCE_ID = []
    for i in range(200):
        POINT = Suture3[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        Temp.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        #####################################
        # Temp.append(distance(Suture3[i, :], Suture4[i, :]))
    Temp = np.array(Temp)
    diff = []
    j = []
    for i in range(99):
        diff.append(Temp[i + 1] - Temp[i])
        if Temp[1] > 14.5:
            if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and Temp[i + 2] > Temp[i + 1] and Temp[i] < 24.5:
                j.append(i)
        else:
            if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and Temp[i + 2] > Temp[i + 1] and Temp[i] < 8:
                j.append(i)
    ID_6 = j[-1]
    ID_5 = DISTANCE_ID[j[-1]]
    ########################################

    XY_Z_points = []
    for i in range(ID_2, ID_3 + 1):
        XY_Z_points.append(Suture7[i, :])
    for i in range(ID_4, ID_5 - 1, -1):
        XY_Z_points.append(Suture4[i, :])
    for i in range(ID_6, ID_1 + 1):
        XY_Z_points.append(Suture3[i, :])
    XY_Z_points.append(Suture7[ID_2, :])
    XY_Z_points = np.array(XY_Z_points)
    points = XY_Z_points
    import matlab.engine

    engine = matlab.engine.start_matlab()  # start matlab process
    # engine = matlab.engine.start_matlab("-desktop")  # start matlab process with graphic UI
    # needs to transfer the data format from python to matlab
    m_points = matlab.double(points.tolist())
    area = engine.Area_ZX_plane_posterior(m_points,ID)

    return area

def Area_YZ_plane_left(PCA_mean_rebuild,ID):

    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # Left Coronal Suture
    DISTANCE = []
    DISTANCE_ID = []
    DIFF = []
    for i in range(np.shape(Suture2)[0]):  # 计算每个2suture的点到suture3的距离
        POINT = Suture2[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID = np.array(DISTANCE_ID)
    DIFF = np.array(DIFF)
    j = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                j.append(i)
    POINT = Suture3[DISTANCE_ID[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture2)
    j.append(POINTB_ID)

    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    for i in range(np.shape(Suture1)[0]):  # 计算每个suture1的点到suture4的距离
        POINT = Suture1[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)

    j = [j[0], j[-1]]

    Left_Coronal_point_ID2 = j[0]
    Left_Coronal_point_ID3 = DISTANCE_ID[j[0]]
    # Right_Coronal_point_ID6 = DISTANCE_ID1[jj[0]]
    # Right_Coronal_point_ID7 = jj[0]

    # Left Lambdoid Suture
    Suture7 = []
    for i in range(100):
        Suture7.append(
            [PCA_mean_rebuild[3 * i + 2100], PCA_mean_rebuild[3 * i + 2101], PCA_mean_rebuild[3 * i + 2102]])
    Suture7 = np.array(Suture7)
    DISTANCE_L = []
    DISTANCE_IDL = []
    DIFF_L = []
    j_L = []
    for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
        POINT_L = Suture3[i, :]
        temp1_L, POINTB_L = Find_point_in_line(POINT_L, Suture7)
        DISTANCE_L.append(temp1_L)
        DISTANCE_IDL.append(POINTB_L)
        if i > 0:
            diff111 = DISTANCE_L[i] - DISTANCE_L[i - 1]
            DIFF_L.append(diff111)
    DISTANCE_L = np.array(DISTANCE_L)
    DISTANCE_IDL = np.array(DISTANCE_IDL)
    DIFF_L = np.array(DIFF_L)
    for i in range(199):
        if DISTANCE_L[i] < 10:
            if abs(DIFF_L[i]) < 0.4:
                j_L.append(i)
    j_L = [j_L[0], j_L[-1]]
    ID_1 = j_L[-1]
    ID_2 = DISTANCE_IDL[j_L[-1]]

    ########################################
    Suture5 = []
    Suture6 = []
    for i in range(50):
        Suture5.append([PCA_mean_rebuild[3 * i + 1800], PCA_mean_rebuild[3 * i + 1801], PCA_mean_rebuild[3 * i + 1802]])
        Suture6.append([PCA_mean_rebuild[3 * i + 1950], PCA_mean_rebuild[3 * i + 1951], PCA_mean_rebuild[3 * i + 1952]])
    Suture5 = np.array(Suture5)
    Suture6 = np.array(Suture6)
    # Get anterior fontanel points
    Left_YZ_plane_points = []
    for i in range(Left_Coronal_point_ID2, 100):
        Left_YZ_plane_points.append(Suture2[i, :])
    for i in range(0, 50):
        Left_YZ_plane_points.append(Suture5[i, :])
    for i in range(0, ID_2 + 1):
        Left_YZ_plane_points.append(Suture7[i, :])
    for i in range(ID_1, Left_Coronal_point_ID3 + 1):
        Left_YZ_plane_points.append(Suture3[i, :])
    Left_YZ_plane_points.append(Suture2[Left_Coronal_point_ID2, :])
    Left_YZ_plane_points = np.array(Left_YZ_plane_points)

    points = Left_YZ_plane_points
    import matlab.engine

    engine = matlab.engine.start_matlab()  # start matlab process
    # engine = matlab.engine.start_matlab("-desktop")  # start matlab process with graphic UI
    # needs to transfer the data format from python to matlab
    m_points = matlab.double(points.tolist())
    area = engine.Area_YZ_plane_left(m_points,ID)

    return area

def Area_YZ_plane_right(PCA_mean_rebuild,ID):
    context = 'SI'
    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # Right Coronal Suture
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    for i in range(np.shape(Suture1)[0]):  # 计算每个suture1的点到suture4的距离
        POINT = Suture1[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    jj = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                jj.append(i)
    POINT = Suture4[DISTANCE_ID1[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture1)
    jj.append(POINTB_ID)
    # 只保留首尾
    jj = [jj[0], jj[-1]]
    Right_Coronal_point_ID2 = jj[0]
    Right_Coronal_point_ID3 = DISTANCE_ID1[jj[0]]

    # Right Lambdoid Suture
    Suture7 = []
    for i in range(100):
        Suture7.append(
            [PCA_mean_rebuild[3 * i + 2100], PCA_mean_rebuild[3 * i + 2101], PCA_mean_rebuild[3 * i + 2102]])
    Suture7 = np.array(Suture7)
    DISTANCE_R = []
    DISTANCE_IDR = []
    DIFF_R = []
    j_R = []
    for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
        POINT_R = Suture4[i, :]
        temp1_R, POINTB_R = Find_point_in_line(POINT_R, Suture7)
        DISTANCE_R.append(temp1_R)
        DISTANCE_IDR.append(POINTB_R)
        if i > 0:
            diff111 = DISTANCE_R[i] - DISTANCE_R[i - 1]
            DIFF_R.append(diff111)
    DISTANCE_R = np.array(DISTANCE_R)
    DISTANCE_IDR = np.array(DISTANCE_IDR)
    DIFF_R = np.array(DIFF_R)
    for i in range(199):
        if DISTANCE_R[i] < 10:
            if abs(DIFF_R[i]) < 0.4:
                j_R.append(i)
    j_R = [j_R[0], j_R[-1]]

    ID_3 = DISTANCE_IDR[j_R[-1]]
    ID_4 = j_R[-1]

    ########################################
    Suture5 = []
    Suture6 = []
    for i in range(50):
        Suture5.append([PCA_mean_rebuild[3 * i + 1800], PCA_mean_rebuild[3 * i + 1801], PCA_mean_rebuild[3 * i + 1802]])
        Suture6.append([PCA_mean_rebuild[3 * i + 1950], PCA_mean_rebuild[3 * i + 1951], PCA_mean_rebuild[3 * i + 1952]])
    Suture5 = np.array(Suture5)
    Suture6 = np.array(Suture6)

    # Get anterior fontanel points
    Right_YZ_plane_points = []
    for i in range(Right_Coronal_point_ID2, 100):
        Right_YZ_plane_points.append(Suture1[i, :])
    for i in range(0, 50):
        Right_YZ_plane_points.append(Suture6[i, :])
    for i in range(99, ID_3 - 1, -1):
        Right_YZ_plane_points.append(Suture7[i, :])
    for i in range(ID_4, Right_Coronal_point_ID3 + 1):
        Right_YZ_plane_points.append(Suture4[i, :])
    Right_YZ_plane_points.append(Suture1[Right_Coronal_point_ID2, :])
    Right_YZ_plane_points = np.array(Right_YZ_plane_points)

    points = Right_YZ_plane_points

    import matlab.engine

    engine = matlab.engine.start_matlab()  # start matlab process
    # engine = matlab.engine.start_matlab("-desktop")  # start matlab process with graphic UI
    # needs to transfer the data format from python to matlab
    m_points = matlab.double(points.tolist())
    area = engine.Area_YZ_plane_right(m_points,ID)

    return area

def Area_total(PCA_mean_rebuild,ID):
    context = 'PF'
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    Temp = []
    DISTANCE_ID = []
    for i in range(200):
        POINT = Suture3[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        Temp.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
    Temp = np.array(Temp)
    diff = []
    j = []
    for i in range(99):
        diff.append(Temp[i + 1] - Temp[i])
        if Temp[1] > 14.5:
            if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and Temp[i + 2] > Temp[i + 1] and Temp[i] < 24.5:
                j.append(i)
        else:
            if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and Temp[i + 2] > Temp[i + 1] and Temp[i] < 8:
                j.append(i)
    Id_4 = j[-1]
    Id_5 = DISTANCE_ID[j[-1]]
    XY_Z_points = []
    for i in range(0, Id_4 + 1):
        XY_Z_points.append(Suture3[i, :])
    for i in range(Id_5, -1, -1):
        XY_Z_points.append(Suture4[i, :])
    XY_Z_points.append(Suture3[0, :])
    XY_Z_points = np.array(XY_Z_points)
    points_XY = XY_Z_points


    context = 'SI'
    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)

    # Left Coronal Suture
    DISTANCE = []
    DISTANCE_ID = []
    DIFF = []
    for i in range(np.shape(Suture2)[0]):  # 计算每个2suture的点到suture3的距离
        POINT = Suture2[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID = np.array(DISTANCE_ID)
    DIFF = np.array(DIFF)
    j = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                j.append(i)
    POINT = Suture3[DISTANCE_ID[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture2)
    j.append(POINTB_ID)

    # Right Coronal Suture
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    for i in range(np.shape(Suture1)[0]):  # 计算每个suture1的点到suture4的距离
        POINT = Suture1[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    jj = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                jj.append(i)
    POINT = Suture4[DISTANCE_ID1[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture1)
    jj.append(POINTB_ID)
    # 只保留首尾
    j = [j[0], j[-1]]
    jj = [jj[0], jj[-1]]

    Left_Coronal_point_ID2 = j[0]
    Left_Coronal_point_ID3 = DISTANCE_ID[j[0]]
    Right_Coronal_point_ID6 = DISTANCE_ID1[jj[0]]
    Right_Coronal_point_ID7 = jj[0]

    # Get anterior fontanel points
    Anterior_ZX_plane_points = []
    for i in range(0, Left_Coronal_point_ID2 + 1):
        Anterior_ZX_plane_points.append(Suture2[i, :])
    for i in range(Left_Coronal_point_ID3, 200):
        Anterior_ZX_plane_points.append(Suture3[i, :])
    for i in range(199, Right_Coronal_point_ID6 - 1, -1):
        Anterior_ZX_plane_points.append(Suture4[i, :])
    for i in range(Right_Coronal_point_ID7, -1, -1):
        Anterior_ZX_plane_points.append(Suture1[i, :])
    Anterior_ZX_plane_points.append(Suture2[0, :])
    Anterior_ZX_plane_points = np.array(Anterior_ZX_plane_points)

    points_ZX_A = Anterior_ZX_plane_points


    Suture7 = []
    for i in range(100):
        Suture7.append(
            [PCA_mean_rebuild[3 * i + 2100], PCA_mean_rebuild[3 * i + 2101], PCA_mean_rebuild[3 * i + 2102]])
    Suture7 = np.array(Suture7)
    DISTANCE_L = []
    DISTANCE_IDL = []
    DIFF_L = []
    j_L = []
    for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
        POINT_L = Suture3[i, :]
        temp1_L, POINTB_L = Find_point_in_line(POINT_L, Suture7)
        DISTANCE_L.append(temp1_L)
        DISTANCE_IDL.append(POINTB_L)
        if i > 0:
            diff111 = DISTANCE_L[i] - DISTANCE_L[i - 1]
            DIFF_L.append(diff111)
    DISTANCE_L = np.array(DISTANCE_L)
    DISTANCE_IDL = np.array(DISTANCE_IDL)
    DIFF_L = np.array(DIFF_L)
    for i in range(199):
        if DISTANCE_L[i] < 10:
            if abs(DIFF_L[i]) < 0.4:
                j_L.append(i)
    j_L = [j_L[0], j_L[-1]]
    ID_1 = j_L[-1]
    ID_2 = DISTANCE_IDL[j_L[-1]]
    # Visualization.single_line_nodes(PCA_mean_rebuild, 0, 800, context, marker_X, marker_Y, marker_Z)
    # plt.show()

    # Right Lambdoid Suture
    DISTANCE_R = []
    DISTANCE_IDR = []
    DIFF_R = []
    j_R = []
    for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
        POINT_R = Suture4[i, :]
        temp1_R, POINTB_R = Find_point_in_line(POINT_R, Suture7)
        DISTANCE_R.append(temp1_R)
        DISTANCE_IDR.append(POINTB_R)
        if i > 0:
            diff111 = DISTANCE_R[i] - DISTANCE_R[i - 1]
            DIFF_R.append(diff111)
    DISTANCE_R = np.array(DISTANCE_R)
    DISTANCE_IDR = np.array(DISTANCE_IDR)
    DIFF_R = np.array(DIFF_R)
    for i in range(199):
        if DISTANCE_R[i] < 10:
            if abs(DIFF_R[i]) < 0.4:
                j_R.append(i)
    j_R = [j_R[0], j_R[-1]]

    ID_3 = DISTANCE_IDR[j_R[-1]]
    ID_4 = j_R[-1]

    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # 2. Get the distance between two nodes ##### Here find the smallest distance
    Temp = []
    DISTANCE_ID = []
    for i in range(200):
        POINT = Suture3[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        Temp.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        #####################################
        # Temp.append(distance(Suture3[i, :], Suture4[i, :]))
    Temp = np.array(Temp)
    diff = []
    j = []
    for i in range(99):
        diff.append(Temp[i + 1] - Temp[i])
        if Temp[1] > 14.5:
            if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and Temp[i + 2] > Temp[i + 1] and Temp[i] < 24.5:
                j.append(i)
        else:
            if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and Temp[i + 2] > Temp[i + 1] and Temp[i] < 8:
                j.append(i)
    ID_6 = j[-1]
    ID_5 = DISTANCE_ID[j[-1]]
    ########################################


    XY_Z_points = []
    for i in range(ID_2, ID_3 + 1):
        XY_Z_points.append(Suture7[i, :])
    for i in range(ID_4, ID_5 - 1, -1):
        XY_Z_points.append(Suture4[i, :])
    for i in range(ID_6, ID_1 + 1):
        XY_Z_points.append(Suture3[i, :])
    XY_Z_points.append(Suture7[ID_2, :])
    XY_Z_points = np.array(XY_Z_points)
    points_ZX_P = XY_Z_points


    marker_X = np.hstack((Suture3[Id_4, 0], Suture4[Id_5,0],Suture3[0, 0], Suture4[0,0],Suture2[Left_Coronal_point_ID2,0],Suture3[Left_Coronal_point_ID3,0],Suture4[Right_Coronal_point_ID6,0],Suture1[Right_Coronal_point_ID7,0],Suture7[ID_2,0],Suture7[ID_3,0],Suture4[ID_4,0],Suture3[ID_1,0]  ))
    marker_Y = np.hstack((Suture3[Id_4, 1], Suture4[Id_5,1],Suture3[0, 1], Suture4[0,1],Suture2[Left_Coronal_point_ID2,1],Suture3[Left_Coronal_point_ID3,1],Suture4[Right_Coronal_point_ID6,1],Suture1[Right_Coronal_point_ID7,1],Suture7[ID_2,1],Suture7[ID_3,1],Suture4[ID_4,1],Suture3[ID_1,1]  ))
    marker_Z = np.hstack((Suture3[Id_4, 2], Suture4[Id_5,2],Suture3[0, 2], Suture4[0,2],Suture2[Left_Coronal_point_ID2,2],Suture3[Left_Coronal_point_ID3,2],Suture4[Right_Coronal_point_ID6,2],Suture1[Right_Coronal_point_ID7,2],Suture7[ID_2,2],Suture7[ID_3,2],Suture4[ID_4,2],Suture3[ID_1,2]  ))
    # Visualization.single_line_nodes(PCA_mean_rebuild, 0, 800, context, marker_X, marker_Y, marker_Z)
    # plt.show()

    # Left Coronal Suture
    DISTANCE = []
    DISTANCE_ID = []
    DIFF = []
    for i in range(np.shape(Suture2)[0]):  # 计算每个2suture的点到suture3的距离
        POINT = Suture2[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID = np.array(DISTANCE_ID)
    DIFF = np.array(DIFF)
    j = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                j.append(i)
    POINT = Suture3[DISTANCE_ID[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture2)
    j.append(POINTB_ID)

    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    for i in range(np.shape(Suture1)[0]):  # 计算每个suture1的点到suture4的距离
        POINT = Suture1[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)

    j = [j[0], j[-1]]

    Left_Coronal_point_ID2 = j[0]
    Left_Coronal_point_ID3 = DISTANCE_ID[j[0]]
    # Right_Coronal_point_ID6 = DISTANCE_ID1[jj[0]]
    # Right_Coronal_point_ID7 = jj[0]

    # Left Lambdoid Suture

    DISTANCE_L = []
    DISTANCE_IDL = []
    DIFF_L = []
    j_L = []
    for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
        POINT_L = Suture3[i, :]
        temp1_L, POINTB_L = Find_point_in_line(POINT_L, Suture7)
        DISTANCE_L.append(temp1_L)
        DISTANCE_IDL.append(POINTB_L)
        if i > 0:
            diff111 = DISTANCE_L[i] - DISTANCE_L[i - 1]
            DIFF_L.append(diff111)
    DISTANCE_L = np.array(DISTANCE_L)
    DISTANCE_IDL = np.array(DISTANCE_IDL)
    DIFF_L = np.array(DIFF_L)
    for i in range(199):
        if DISTANCE_L[i] < 10:
            if abs(DIFF_L[i]) < 0.4:
                j_L.append(i)
    j_L = [j_L[0], j_L[-1]]
    ID_1 = j_L[-1]
    ID_2 = DISTANCE_IDL[j_L[-1]]

    ########################################
    Suture5 = []
    Suture6 = []
    for i in range(50):
        Suture5.append([PCA_mean_rebuild[3 * i + 1800], PCA_mean_rebuild[3 * i + 1801], PCA_mean_rebuild[3 * i + 1802]])
        Suture6.append([PCA_mean_rebuild[3 * i + 1950], PCA_mean_rebuild[3 * i + 1951], PCA_mean_rebuild[3 * i + 1952]])
    Suture5 = np.array(Suture5)
    Suture6 = np.array(Suture6)
    # Get anterior fontanel points
    Left_YZ_plane_points = []
    for i in range(Left_Coronal_point_ID2, 100):
        Left_YZ_plane_points.append(Suture2[i, :])
    for i in range(0, 50):
        Left_YZ_plane_points.append(Suture5[i, :])
    for i in range(0, ID_2 + 1):
        Left_YZ_plane_points.append(Suture7[i, :])
    for i in range(ID_1, Left_Coronal_point_ID3 + 1):
        Left_YZ_plane_points.append(Suture3[i, :])
    Left_YZ_plane_points.append(Suture2[Left_Coronal_point_ID2, :])
    Left_YZ_plane_points = np.array(Left_YZ_plane_points)

    points_YZ_L = Left_YZ_plane_points


    # Right Coronal Suture
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    for i in range(np.shape(Suture1)[0]):  # 计算每个suture1的点到suture4的距离
        POINT = Suture1[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    jj = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                jj.append(i)
    POINT = Suture4[DISTANCE_ID1[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture1)
    jj.append(POINTB_ID)
    # 只保留首尾
    jj = [jj[0], jj[-1]]
    Right_Coronal_point_ID2 = jj[0]
    Right_Coronal_point_ID3 = DISTANCE_ID1[jj[0]]

    # Right Lambdoid Suture
    DISTANCE_R = []
    DISTANCE_IDR = []
    DIFF_R = []
    j_R = []
    for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
        POINT_R = Suture4[i, :]
        temp1_R, POINTB_R = Find_point_in_line(POINT_R, Suture7)
        DISTANCE_R.append(temp1_R)
        DISTANCE_IDR.append(POINTB_R)
        if i > 0:
            diff111 = DISTANCE_R[i] - DISTANCE_R[i - 1]
            DIFF_R.append(diff111)
    DISTANCE_R = np.array(DISTANCE_R)
    DISTANCE_IDR = np.array(DISTANCE_IDR)
    DIFF_R = np.array(DIFF_R)
    for i in range(199):
        if DISTANCE_R[i] < 10:
            if abs(DIFF_R[i]) < 0.4:
                j_R.append(i)
    j_R = [j_R[0], j_R[-1]]

    ID_3 = DISTANCE_IDR[j_R[-1]]
    ID_4 = j_R[-1]

    ########################################


    # Get anterior fontanel points
    Right_YZ_plane_points = []
    for i in range(Right_Coronal_point_ID2, 100):
        Right_YZ_plane_points.append(Suture1[i, :])
    for i in range(0, 50):
        Right_YZ_plane_points.append(Suture6[i, :])
    for i in range(99, ID_3 - 1, -1):
        Right_YZ_plane_points.append(Suture7[i, :])
    for i in range(ID_4, Right_Coronal_point_ID3 + 1):
        Right_YZ_plane_points.append(Suture4[i, :])
    Right_YZ_plane_points.append(Suture1[Right_Coronal_point_ID2, :])
    Right_YZ_plane_points = np.array(Right_YZ_plane_points)

    points_YZ_R = Right_YZ_plane_points

    import matlab.engine

    engine = matlab.engine.start_matlab()  # start matlab process
    # engine = matlab.engine.start_matlab("-desktop")  # start matlab process with graphic UI
    # needs to transfer the data format from python to matlab
    PXY = matlab.double(points_XY.tolist())
    PZX_A = matlab.double(points_ZX_A.tolist())
    PZX_P = matlab.double(points_ZX_P.tolist())
    PYZ_L = matlab.double(points_YZ_L.tolist())
    PYZ_R = matlab.double(points_YZ_R.tolist())
    [area,area1,area2,area3,area4,area5] = engine.Area_total(PXY,PZX_A,PZX_P,PYZ_L,PYZ_R,ID,nargout=6)
# [area,area1,area2,area3,area4,area5]
    return area, area1,area2,area3,area4,area5

def Area_Metopic(PCA_mean_rebuild,ID,parameter):
    # 1, Metopic suture ZX plane
    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)
    # 2. Get the distance between two nodes
    Temp = []
    for i in range(100):
        Temp.append(distance(Suture1[i, :], Suture2[i, :]))
    Temp = np.array(Temp)
    diff = []
    j = []
    for i in range(99):
        diff.append(Temp[i + 1] - Temp[i])
        if Temp[1] > 5:
            if diff[i] > 0.15 and diff[i] > 1.5 * diff[i - 1] and diff[i] < 1 and Temp[i] < 12:
                j.append(i)
        else:
            # if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and diff[i] < 1 and Temp[i] < 3:   OLD
            if diff[i] > 0.15 and diff[i] > 1.5 * diff[i - 1] and diff[i] < 1 and Temp[i] < 3:
                # if i>1 and diff[i-1] < 1.5* diff[i-2]:
                j.append(i)
    # Get Metopic suture points
    Metopic_area_points = []
    for i in range(0, j[-1] + 1):
        Metopic_area_points.append(Suture1[i, :])
    for i in range(j[-1], -1, -1):
        Metopic_area_points.append(Suture2[i, :])
    Metopic_area_points.append(Suture1[0, :])
    Metopic_area_points = np.array(Metopic_area_points)

    points = Metopic_area_points
    import matlab.engine

    engine = matlab.engine.start_matlab()
    m_points = matlab.double(points.tolist())
    metopic_area = engine.Area_ZX_plane_anterior(m_points, 0,parameter)
    return metopic_area

def Area_Coronal(PCA_mean_rebuild,ID,parameter):
    # Left coronal suture
    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # Left Coronal Suture
    DISTANCE = []
    DISTANCE_ID = []
    DIFF = []
    for i in range(np.shape(Suture2)[0]):  # 计算每个2suture的点到suture3的距离
        POINT = Suture2[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID = np.array(DISTANCE_ID)
    DIFF = np.array(DIFF)
    j = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                j.append(i)
    POINT = Suture3[DISTANCE_ID[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture2)
    j.append(POINTB_ID)

    # Get Left coronal suture points
    Left_Coronal_area_points = []
    for i in range(j[0], j[-1] + 1):
        Left_Coronal_area_points.append(Suture2[i, :])
    for i in range(DISTANCE_ID[j[-1]], DISTANCE_ID[j[0]] + 1):
        Left_Coronal_area_points.append(Suture3[i, :])
    Left_Coronal_area_points.append(Suture2[j[0], :])
    Left_Coronal_area_points = np.array(Left_Coronal_area_points)

    points = Left_Coronal_area_points
    import matlab.engine

    engine = matlab.engine.start_matlab()
    m_points = matlab.double(points.tolist())
    left_coronal_area = engine.Area_YZ_plane_left(m_points, 0,parameter)

    # Right coronal suture
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    for i in range(np.shape(Suture1)[0]):  # 计算每个suture1的点到suture4的距离
        POINT = Suture1[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    jj = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                jj.append(i)
    POINT = Suture4[DISTANCE_ID1[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture1)
    jj.append(POINTB_ID)
    # 只保留首尾
    j = [j[0], j[-1]]
    jj = [jj[0], jj[-1]]

    # Get right coronal suture points
    Right_Coronal_area_points = []
    for i in range(jj[0], jj[-1] + 1):
        Right_Coronal_area_points.append(Suture1[i, :])
    for i in range(DISTANCE_ID1[jj[-1]], DISTANCE_ID1[jj[0]] + 1):
        Right_Coronal_area_points.append(Suture4[i, :])
    Right_Coronal_area_points.append(Suture1[jj[0], :])
    Right_Coronal_area_points = np.array(Right_Coronal_area_points)

    points = Right_Coronal_area_points
    import matlab.engine

    engine = matlab.engine.start_matlab()
    m_points = matlab.double(points.tolist())
    Right_coronal_area = engine.Area_YZ_plane_right(m_points, 0,parameter)
    return left_coronal_area, Right_coronal_area

def Area_Sagittal(PCA_mean_rebuild,ID,parameter):
    # 3. Sagittal suture area XY plane
    context = 'PF'
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # 2. Get the distance between two nodes ##### Here find the smallest distance
    Temp = []
    DISTANCE_ID = []
    for i in range(200):
        POINT = Suture3[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        Temp.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        #####################################
        # Temp.append(distance(Suture3[i, :], Suture4[i, :]))
    Temp = np.array(Temp)
    diff = []
    j = []
    for i in range(99):
        diff.append(Temp[i + 1] - Temp[i])
        if Temp[1] > 14.5:
            if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and Temp[i + 2] > Temp[i + 1] and Temp[i] < 24.5:
                j.append(i)
        else:
            if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and Temp[i + 2] > Temp[i + 1] and Temp[i] < 8:
                j.append(i)
    Id_4 = j[-1]
    Id_5 = DISTANCE_ID[j[-1]]

    XY_Z_points = []
    for i in range(0, Id_4 + 1):
        XY_Z_points.append(Suture3[i, :])
    for i in range(Id_5, -1, -1):
        XY_Z_points.append(Suture4[i, :])
    XY_Z_points.append(Suture3[0, :])
    XY_Z_points = np.array(XY_Z_points)
    points = XY_Z_points
    import matlab.engine

    engine = matlab.engine.start_matlab()  # start matlab process
    m_points = matlab.double(points.tolist())
    Sagittal_area = engine.Area_XY_plane(m_points, 0,parameter)
    return Sagittal_area

def Area_Squamosal(PCA_mean_rebuild,ID,parameter):
    # 4. Squamosal suture
    # left squamosal suture
    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # Left Coronal Suture
    DISTANCE = []
    DISTANCE_ID = []
    DIFF = []
    for i in range(np.shape(Suture2)[0]):  # 计算每个suture2的点到suture3的距离
        POINT = Suture2[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID = np.array(DISTANCE_ID)
    DIFF = np.array(DIFF)
    j = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                j.append(i)
    POINT = Suture3[DISTANCE_ID[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture2)
    j.append(POINTB_ID)

    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    for i in range(np.shape(Suture1)[0]):  # 计算每个suture1的点到suture4的距离
        POINT = Suture1[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    jj = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                jj.append(i)
    POINT = Suture4[DISTANCE_ID1[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture1)
    jj.append(POINTB_ID)
    j = [j[0], j[-1]]
    jj = [jj[0], jj[-1]]

    POINT_3 = Suture3[DISTANCE_ID[j[-1]], :]
    Suture5 = []
    Suture6 = []
    for i in range(50):
        Suture5.append([PCA_mean_rebuild[3 * i + 1800], PCA_mean_rebuild[3 * i + 1801], PCA_mean_rebuild[3 * i + 1802]])
        Suture6.append([PCA_mean_rebuild[3 * i + 1950], PCA_mean_rebuild[3 * i + 1951], PCA_mean_rebuild[3 * i + 1952]])
    Suture5 = np.array(Suture5)
    Suture6 = np.array(Suture6)
    useless, POINTB_ID = Find_point_in_line(POINT_3, Suture5)
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    jjj = []
    for i in range(0, 50):  # 计算每个suture5的点到suture3的距离
        POINT = Suture5[i, :]
        temp1, POINTB_ID1 = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID1)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    for i in range(49):
        if DISTANCE[i] < DISTANCE[0] / 2:
            if abs(DIFF[i]) < 0.3:
                jjj.append(i)
            else:
                if i > 0 and abs(DIFF[i] + DIFF[i - 1]) < 0.3:
                    jjj.append(i)

    Left_squamosal_area_points = []
    for i in range(jjj[0], jjj[-1] + 1):
        Left_squamosal_area_points.append(Suture5[i, :])
    for i in range(DISTANCE_ID1[jjj[-1]], DISTANCE_ID1[jjj[0]] + 1):
        Left_squamosal_area_points.append(Suture3[i, :])
    Left_squamosal_area_points.append(Suture5[jjj[0], :])
    Left_squamosal_area_points = np.array(Left_squamosal_area_points)

    points = Left_squamosal_area_points
    import matlab.engine

    engine = matlab.engine.start_matlab()
    m_points = matlab.double(points.tolist())
    left_squamosal_area = engine.Area_YZ_plane_left(m_points, 0,parameter)

    # right squamosal suture
    jjjj = []
    DISTANCE_ID2 = []
    DISTANCE = []
    DIFF = []
    for i in range(0, 50):  # 计算每个suture6的点到suture4的距离
        POINT = Suture6[i, :]
        temp1, POINTB_ID1 = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID2.append(POINTB_ID1)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID2 = np.array(DISTANCE_ID2)
    DIFF = np.array(DIFF)
    for i in range(49):
        if DISTANCE[i] < DISTANCE[0] / 2:
            if abs(DIFF[i]) < 0.3:
                jjjj.append(i)
            else:
                if i > 0 and abs(DIFF[i] + DIFF[i - 1]) < 0.3:
                    jjjj.append(i)

    jjj = [jjj[0], jjj[-1]]
    jjjj = [jjjj[0], jjjj[-1]]

    Right_squamosal_area_points = []
    for i in range(jjjj[0], jjjj[-1] + 1):
        Right_squamosal_area_points.append(Suture6[i, :])
    for i in range(DISTANCE_ID2[jjjj[-1]], DISTANCE_ID2[jjjj[0]] + 1):
        Right_squamosal_area_points.append(Suture4[i, :])
    Right_squamosal_area_points.append(Suture6[jjjj[0], :])
    Right_squamosal_area_points = np.array(Right_squamosal_area_points)

    points = Right_squamosal_area_points
    import matlab.engine

    engine = matlab.engine.start_matlab()
    m_points = matlab.double(points.tolist())
    right_squamosal_area = engine.Area_YZ_plane_right(m_points, 0,parameter)
    return left_squamosal_area,right_squamosal_area

def Area_Lambdoidal(PCA_mean_rebuild,ID,parameter):
    # 5. Lambdoidal suture
    # Left
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append(
            [PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append(
            [PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # Left Lambdoid Suture
    Suture7 = []
    for i in range(100):
        Suture7.append(
            [PCA_mean_rebuild[3 * i + 2100], PCA_mean_rebuild[3 * i + 2101], PCA_mean_rebuild[3 * i + 2102]])
    Suture7 = np.array(Suture7)
    DISTANCE_L = []
    DISTANCE_IDL = []
    DIFF_L = []
    j_L = []
    for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
        POINT_L = Suture3[i, :]
        temp1_L, POINTB_L = Find_point_in_line(POINT_L, Suture7)
        DISTANCE_L.append(temp1_L)
        DISTANCE_IDL.append(POINTB_L)
        if i > 0:
            diff111 = DISTANCE_L[i] - DISTANCE_L[i - 1]
            DIFF_L.append(diff111)
    DISTANCE_L = np.array(DISTANCE_L)
    DISTANCE_IDL = np.array(DISTANCE_IDL)
    DIFF_L = np.array(DIFF_L)
    for i in range(199):
        if DISTANCE_L[i] < 10:
            if abs(DIFF_L[i]) < 0.4:
                j_L.append(i)
    j_L = [j_L[0], j_L[-1]]

    Left_lambdoidal_area_points = []
    for i in range(j_L[0], j_L[-1] + 1):
        Left_lambdoidal_area_points.append(Suture3[i, :])
    for i in range(DISTANCE_IDL[j_L[-1]], DISTANCE_IDL[j_L[0]] + 1):
        Left_lambdoidal_area_points.append(Suture7[i, :])
    Left_lambdoidal_area_points.append(Suture3[j_L[0], :])
    Left_lambdoidal_area_points = np.array(Left_lambdoidal_area_points)

    points = Left_lambdoidal_area_points
    import matlab.engine

    engine = matlab.engine.start_matlab()
    m_points = matlab.double(points.tolist())
    Left_lambdoidal_area = engine.Area_ZX_plane_posterior(m_points, 0,parameter)

    # Right lambdoidal suture
    DISTANCE_R = []
    DISTANCE_IDR = []
    DIFF_R = []
    j_R = []
    for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
        POINT_R = Suture4[i, :]
        temp1_R, POINTB_R = Find_point_in_line(POINT_R, Suture7)
        DISTANCE_R.append(temp1_R)
        DISTANCE_IDR.append(POINTB_R)
        if i > 0:
            diff111 = DISTANCE_R[i] - DISTANCE_R[i - 1]
            DIFF_R.append(diff111)
    DISTANCE_R = np.array(DISTANCE_R)
    DISTANCE_IDR = np.array(DISTANCE_IDR)
    DIFF_R = np.array(DIFF_R)
    for i in range(199):
        if DISTANCE_R[i] < 10:
            if abs(DIFF_R[i]) < 0.4:
                j_R.append(i)
    j_R = [j_R[0], j_R[-1]]

    Right_lambdoidal_area_points = []
    for i in range(j_R[0], j_R[-1] + 1):
        Right_lambdoidal_area_points.append(Suture4[i, :])
    for i in range(DISTANCE_IDR[j_R[-1]], DISTANCE_IDR[j_R[0]] - 1, -1):
        Right_lambdoidal_area_points.append(Suture7[i, :])
    Right_lambdoidal_area_points.append(Suture4[j_R[0], :])
    Right_lambdoidal_area_points = np.array(Right_lambdoidal_area_points)

    points = Right_lambdoidal_area_points
    import matlab.engine

    engine = matlab.engine.start_matlab()
    m_points = matlab.double(points.tolist())
    Right_lambdoidal_area = engine.Area_ZX_plane_posterior(m_points, 0,parameter)
    ########################################################################
    return Left_lambdoidal_area,Right_lambdoidal_area

def Area_Sphenoid(PCA_mean_rebuild,ID,parameter):
    # Left
    # Left coronal suture
    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # Left Coronal Suture
    DISTANCE = []
    DISTANCE_ID = []
    DIFF = []
    for i in range(np.shape(Suture2)[0]):  # 计算每个2suture的点到suture3的距离
        POINT = Suture2[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID = np.array(DISTANCE_ID)
    DIFF = np.array(DIFF)
    j = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                j.append(i)
    POINT = Suture3[DISTANCE_ID[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture2)
    j.append(POINTB_ID)

    # left squamosal suture
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    for i in range(np.shape(Suture1)[0]):  # 计算每个suture1的点到suture4的距离
        POINT = Suture1[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    jj = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                jj.append(i)
    POINT = Suture4[DISTANCE_ID1[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture1)
    jj.append(POINTB_ID)
    j = [j[0], j[-1]]
    jj = [jj[0], jj[-1]]

    POINT_3 = Suture3[DISTANCE_ID[j[-1]], :]
    Suture5 = []
    Suture6 = []
    for i in range(50):
        Suture5.append([PCA_mean_rebuild[3 * i + 1800], PCA_mean_rebuild[3 * i + 1801], PCA_mean_rebuild[3 * i + 1802]])
        Suture6.append([PCA_mean_rebuild[3 * i + 1950], PCA_mean_rebuild[3 * i + 1951], PCA_mean_rebuild[3 * i + 1952]])
    Suture5 = np.array(Suture5)
    Suture6 = np.array(Suture6)
    useless, POINTB_ID = Find_point_in_line(POINT_3, Suture5)
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    jjj = []
    for i in range(0, 50):  # 计算每个suture5的点到suture3的距离
        POINT = Suture5[i, :]
        temp1, POINTB_ID1 = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID1)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    for i in range(49):
        if DISTANCE[i] < DISTANCE[0] / 2:
            if abs(DIFF[i]) < 0.3:
                jjj.append(i)
            else:
                if i > 0 and abs(DIFF[i] + DIFF[i - 1]) < 0.3:
                    jjj.append(i)

    # Get Left Sphenoid suture points
    Left_Sphenoid_fontanel_points = []
    for i in range(j[-1], 100):
        Left_Sphenoid_fontanel_points.append(Suture2[i, :])
    for i in range(0, jjj[0] + 1):
        Left_Sphenoid_fontanel_points.append(Suture5[i, :])
    for i in range(DISTANCE_ID1[jjj[0]], DISTANCE_ID[j[-1]] + 1):
        Left_Sphenoid_fontanel_points.append(Suture3[i, :])
    Left_Sphenoid_fontanel_points.append(Suture2[j[-1], :])
    Left_Sphenoid_fontanel_points = np.array(Left_Sphenoid_fontanel_points)
    points = Left_Sphenoid_fontanel_points
    import matlab.engine
    engine = matlab.engine.start_matlab()  # start matlab process
    # engine = matlab.engine.start_matlab("-desktop")  # start matlab process with graphic UI
    # needs to transfer the data format from python to matlab
    m_points = matlab.double(points.tolist())
    Left_Sphenoid_fontanel_area = engine.Area_YZ_plane_left(m_points, 0,parameter)

    # Right sphenoid fontanel
    # Right coronal suture
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    for i in range(np.shape(Suture1)[0]):  # 计算每个suture1的点到suture4的距离
        POINT = Suture1[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    jj = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                jj.append(i)
    POINT = Suture4[DISTANCE_ID1[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture1)
    jj.append(POINTB_ID)
    # 只保留首尾
    j = [j[0], j[-1]]
    jj = [jj[0], jj[-1]]

    # right squamosal suture
    jjjj = []
    DISTANCE_ID2 = []
    DISTANCE = []
    DIFF = []
    for i in range(0, 50):  # 计算每个suture6的点到suture4的距离
        POINT = Suture6[i, :]
        temp1, POINTB_ID1 = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID2.append(POINTB_ID1)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID2 = np.array(DISTANCE_ID2)
    DIFF = np.array(DIFF)
    for i in range(49):
        if DISTANCE[i] < DISTANCE[0] / 2:
            if abs(DIFF[i]) < 0.3:
                jjjj.append(i)
            else:
                if i > 0 and abs(DIFF[i] + DIFF[i - 1]) < 0.3:
                    jjjj.append(i)

    jjj = [jjj[0], jjj[-1]]
    jjjj = [jjjj[0], jjjj[-1]]

    Right_sphenoid_fontanel_points = []
    for i in range(jj[-1], 100):
        Right_sphenoid_fontanel_points.append(Suture1[i, :])
    for i in range(0, jjjj[0] + 1):
        Right_sphenoid_fontanel_points.append(Suture6[i, :])
    for i in range(DISTANCE_ID2[jjjj[0]], DISTANCE_ID1[jj[-1]] + 1):
        Right_sphenoid_fontanel_points.append(Suture4[i, :])
    Right_sphenoid_fontanel_points.append(Suture1[jj[-1], :])

    Right_sphenoid_fontanel_points = np.array(Right_sphenoid_fontanel_points)
    points = Right_sphenoid_fontanel_points
    import matlab.engine
    engine = matlab.engine.start_matlab()  # start matlab process
    # engine = matlab.engine.start_matlab("-desktop")  # start matlab process with graphic UI
    # needs to transfer the data format from python to matlab
    m_points = matlab.double(points.tolist())
    Right_Sphenoid_fontanel_area = engine.Area_YZ_plane_right(m_points, 0,parameter)
    return Left_Sphenoid_fontanel_area, Right_Sphenoid_fontanel_area

def Area_Mastoid(PCA_mean_rebuild,ID,parameter):
    # Left
    # left squamosal suture
    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # Left Coronal Suture
    DISTANCE = []
    DISTANCE_ID = []
    DIFF = []
    for i in range(np.shape(Suture2)[0]):  # 计算每个suture2的点到suture3的距离
        POINT = Suture2[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID = np.array(DISTANCE_ID)
    DIFF = np.array(DIFF)
    j = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                j.append(i)
    POINT = Suture3[DISTANCE_ID[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture2)
    j.append(POINTB_ID)

    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    for i in range(np.shape(Suture1)[0]):  # 计算每个suture1的点到suture4的距离
        POINT = Suture1[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    jj = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                jj.append(i)
    POINT = Suture4[DISTANCE_ID1[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture1)
    jj.append(POINTB_ID)
    j = [j[0], j[-1]]
    jj = [jj[0], jj[-1]]

    POINT_3 = Suture3[DISTANCE_ID[j[-1]], :]
    Suture5 = []
    Suture6 = []
    for i in range(50):
        Suture5.append([PCA_mean_rebuild[3 * i + 1800], PCA_mean_rebuild[3 * i + 1801], PCA_mean_rebuild[3 * i + 1802]])
        Suture6.append([PCA_mean_rebuild[3 * i + 1950], PCA_mean_rebuild[3 * i + 1951], PCA_mean_rebuild[3 * i + 1952]])
    Suture5 = np.array(Suture5)
    Suture6 = np.array(Suture6)
    useless, POINTB_ID = Find_point_in_line(POINT_3, Suture5)
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    jjj = []
    for i in range(0, 50):  # 计算每个suture5的点到suture3的距离
        POINT = Suture5[i, :]
        temp1, POINTB_ID1 = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID1)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    for i in range(49):
        if DISTANCE[i] < DISTANCE[0] / 2:
            if abs(DIFF[i]) < 0.3:
                jjj.append(i)
            else:
                if i > 0 and abs(DIFF[i] + DIFF[i - 1]) < 0.3:
                    jjj.append(i)

    # Left lambdoidal suture
    Suture7 = []
    for i in range(100):
        Suture7.append(
            [PCA_mean_rebuild[3 * i + 2100], PCA_mean_rebuild[3 * i + 2101], PCA_mean_rebuild[3 * i + 2102]])
    Suture7 = np.array(Suture7)
    DISTANCE_L = []
    DISTANCE_IDL = []
    DIFF_L = []
    j_L = []
    for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
        POINT_L = Suture3[i, :]
        temp1_L, POINTB_L = Find_point_in_line(POINT_L, Suture7)
        DISTANCE_L.append(temp1_L)
        DISTANCE_IDL.append(POINTB_L)
        if i > 0:
            diff111 = DISTANCE_L[i] - DISTANCE_L[i - 1]
            DIFF_L.append(diff111)
    DISTANCE_L = np.array(DISTANCE_L)
    DISTANCE_IDL = np.array(DISTANCE_IDL)
    DIFF_L = np.array(DIFF_L)
    for i in range(199):
        if DISTANCE_L[i] < 10:
            if abs(DIFF_L[i]) < 0.4:
                j_L.append(i)
    j_L = [j_L[0], j_L[-1]]

    Left_mastoid_fontanel = []
    for i in range(jjj[-1], 50):
        Left_mastoid_fontanel.append(Suture5[i, :])
    for i in range(0, DISTANCE_IDL[j_L[-1]] + 1):
        Left_mastoid_fontanel.append(Suture7[i, :])
    for i in range(j_L[-1], DISTANCE_ID1[jjj[-1]] + 1):
        Left_mastoid_fontanel.append(Suture3[i, :])

    Left_mastoid_fontanel = np.array(Left_mastoid_fontanel)
    points = Left_mastoid_fontanel
    import matlab.engine
    engine = matlab.engine.start_matlab()  # start matlab process
    # engine = matlab.engine.start_matlab("-desktop")  # start matlab process with graphic UI
    # needs to transfer the data format from python to matlab
    m_points = matlab.double(points.tolist())
    Left_mastoid_fontanel_area = engine.Area_YZ_plane_left(m_points, 0,parameter)
    # Right mastoid

    # right squamosal suture
    jjjj = []
    DISTANCE_ID2 = []
    DISTANCE = []
    DIFF = []
    for i in range(0, 50):  # 计算每个suture6的点到suture4的距离
        POINT = Suture6[i, :]
        temp1, POINTB_ID1 = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID2.append(POINTB_ID1)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID2 = np.array(DISTANCE_ID2)
    DIFF = np.array(DIFF)
    for i in range(49):
        if DISTANCE[i] < DISTANCE[0] / 2:
            if abs(DIFF[i]) < 0.3:
                jjjj.append(i)
            else:
                if i > 0 and abs(DIFF[i] + DIFF[i - 1]) < 0.3:
                    jjjj.append(i)

    jjj = [jjj[0], jjj[-1]]
    jjjj = [jjjj[0], jjjj[-1]]

    # Right lambdoidal suture
    DISTANCE_R = []
    DISTANCE_IDR = []
    DIFF_R = []
    j_R = []
    for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
        POINT_R = Suture4[i, :]
        temp1_R, POINTB_R = Find_point_in_line(POINT_R, Suture7)
        DISTANCE_R.append(temp1_R)
        DISTANCE_IDR.append(POINTB_R)
        if i > 0:
            diff111 = DISTANCE_R[i] - DISTANCE_R[i - 1]
            DIFF_R.append(diff111)
    DISTANCE_R = np.array(DISTANCE_R)
    DISTANCE_IDR = np.array(DISTANCE_IDR)
    DIFF_R = np.array(DIFF_R)
    for i in range(199):
        if DISTANCE_R[i] < 10:
            if abs(DIFF_R[i]) < 0.4:
                j_R.append(i)
    j_R = [j_R[0], j_R[-1]]

    Right_mastoid_fontanel = []
    for i in range(jjjj[-1], 50):
        Right_mastoid_fontanel.append(Suture6[i, :])
    for i in range(99, DISTANCE_IDR[j_R[-1]] - 1, -1):
        Right_mastoid_fontanel.append(Suture7[i, :])
    for i in range(j_R[-1], DISTANCE_ID2[jjjj[-1]] + 1):
        Right_mastoid_fontanel.append(Suture4[i, :])

    Right_mastoid_fontanel = np.array(Right_mastoid_fontanel)
    points = Right_mastoid_fontanel
    import matlab.engine
    engine = matlab.engine.start_matlab()  # start matlab process
    # engine = matlab.engine.start_matlab("-desktop")  # start matlab process with graphic UI
    # needs to transfer the data format from python to matlab
    m_points = matlab.double(points.tolist())
    Right_mastoid_fontanel_area = engine.Area_YZ_plane_right(m_points, 0,parameter)
    return Left_mastoid_fontanel_area,Right_mastoid_fontanel_area

def Curv_anterior_fontanel(PCA_mean_rebuild,ID):
    context = 'anterior fontanelle'
    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)
    # 2. Get the distance between two nodes
    Temp = []
    for i in range(100):
        Temp.append(distance(Suture1[i, :], Suture2[i, :]))
    Temp = np.array(Temp)
    diff = []
    j = []
    for i in range(99):
        diff.append(Temp[i + 1] - Temp[i])
        if Temp[1] > 5:
            if diff[i] > 0.15 and diff[i] > 1.5 * diff[i - 1] and diff[i] < 1 and Temp[i] < 12:
                j.append(i)
        else:
            # if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and diff[i] < 1 and Temp[i] < 3:   OLD
            if diff[i] > 0.15 and diff[i] > 1.5 * diff[i - 1] and diff[i] < 1 and Temp[i] < 3:
                # if i>1 and diff[i-1] < 1.5* diff[i-2]:
                j.append(i)
    marker_X = np.array([Suture1[j[-1], 0], Suture2[j[-1], 0]])
    marker_Y = np.array([Suture1[j[-1], 1], Suture2[j[-1], 1]])
    marker_Z = np.array([Suture1[j[-1], 2], Suture2[j[-1], 2]])
    diff = np.array(diff)
    Metopic_point_ID = j[-1]

    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # Left Coronal Suture
    DISTANCE = []
    DISTANCE_ID = []
    DIFF = []
    for i in range(np.shape(Suture2)[0]):  # 计算每个2suture的点到suture3的距离
        POINT = Suture2[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID = np.array(DISTANCE_ID)
    DIFF = np.array(DIFF)
    j = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                j.append(i)
    POINT = Suture3[DISTANCE_ID[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture2)
    j.append(POINTB_ID)

    # Right Coronal Suture
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    for i in range(np.shape(Suture1)[0]):  # 计算每个suture1的点到suture4的距离
        POINT = Suture1[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    jj = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                jj.append(i)
    POINT = Suture4[DISTANCE_ID1[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture1)
    jj.append(POINTB_ID)
    # 只保留首尾
    j = [j[0], j[-1]]
    jj = [jj[0], jj[-1]]

    Left_Coronal_point_ID2 = j[0]
    Left_Coronal_point_ID3 = DISTANCE_ID[j[0]]
    Right_Coronal_point_ID6 = DISTANCE_ID1[jj[0]]
    Right_Coronal_point_ID7 = jj[0]

    # Get anterior fontanel points
    Anterior_fontanel_points = []
    for i in range(Metopic_point_ID, Left_Coronal_point_ID2 + 1):
        Anterior_fontanel_points.append(Suture2[i, :])
    for i in range(Left_Coronal_point_ID3, 200):
        Anterior_fontanel_points.append(Suture3[i, :])
    for i in range(199, Right_Coronal_point_ID6 - 1, -1):
        Anterior_fontanel_points.append(Suture4[i, :])
    for i in range(Right_Coronal_point_ID7, Metopic_point_ID - 1, -1):
        Anterior_fontanel_points.append(Suture1[i, :])
    Anterior_fontanel_points.append(Suture2[Metopic_point_ID, :])
    Anterior_fontanel_points = np.array(Anterior_fontanel_points)

    points = Anterior_fontanel_points
    import matlab.engine

    engine = matlab.engine.start_matlab()  # start matlab process
    # engine = matlab.engine.start_matlab("-desktop")  # start matlab process with graphic UI
    # needs to transfer the data format from python to matlab
    m_points = matlab.double(points.tolist())
    Anterior_fontanel_curv = engine.Curv_ZX_plane_anterior(m_points, 0)

    return Anterior_fontanel_curv

def Curv_posterior_fontanel(PCA_mean_rebuild,ID):
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)

    # 2. Get the distance between two nodes ##### Here find the smallest distance
    Temp = []
    DISTANCE_ID = []
    for i in range(200):
        POINT = Suture3[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        Temp.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        #####################################
        # Temp.append(distance(Suture3[i, :], Suture4[i, :]))
    Temp = np.array(Temp)
    diff = []
    j = []
    for i in range(99):
        diff.append(Temp[i + 1] - Temp[i])
        if Temp[1] > 14.5:
            if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and Temp[i + 2] > Temp[i + 1] and Temp[i] < 24.5:
                j.append(i)
        else:
            if diff[i] > 0.15 and diff[i] > 1.8 * diff[i - 1] and Temp[i + 2] > Temp[i + 1] and Temp[i] < 8:
                j.append(i)
    Id_4 = j[-1]
    Id_5 = DISTANCE_ID[j[-1]]

    # Left Lambdoid Suture
    Suture7 = []
    for i in range(100):
        Suture7.append(
            [PCA_mean_rebuild[3 * i + 2100], PCA_mean_rebuild[3 * i + 2101], PCA_mean_rebuild[3 * i + 2102]])
    Suture7 = np.array(Suture7)
    DISTANCE_L = []
    DISTANCE_IDL = []
    DIFF_L = []
    j_L = []
    for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
        POINT_L = Suture3[i, :]
        temp1_L, POINTB_L = Find_point_in_line(POINT_L, Suture7)
        DISTANCE_L.append(temp1_L)
        DISTANCE_IDL.append(POINTB_L)
        if i > 0:
            diff111 = DISTANCE_L[i] - DISTANCE_L[i - 1]
            DIFF_L.append(diff111)
    DISTANCE_L = np.array(DISTANCE_L)
    DISTANCE_IDL = np.array(DISTANCE_IDL)
    DIFF_L = np.array(DIFF_L)
    for i in range(199):
        if DISTANCE_L[i] < 10:
            if abs(DIFF_L[i]) < 0.4:
                j_L.append(i)
    j_L = [j_L[0], j_L[-1]]
    ID_1 = DISTANCE_IDL[j_L[0]]
    ID_6 = j_L[0]
    # Right Lambdoid Suture
    DISTANCE_R = []
    DISTANCE_IDR = []
    DIFF_R = []
    j_R = []
    for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
        POINT_R = Suture4[i, :]
        temp1_R, POINTB_R = Find_point_in_line(POINT_R, Suture7)
        DISTANCE_R.append(temp1_R)
        DISTANCE_IDR.append(POINTB_R)
        if i > 0:
            diff111 = DISTANCE_R[i] - DISTANCE_R[i - 1]
            DIFF_R.append(diff111)
    DISTANCE_R = np.array(DISTANCE_R)
    DISTANCE_IDR = np.array(DISTANCE_IDR)
    DIFF_R = np.array(DIFF_R)
    for i in range(199):
        if DISTANCE_R[i] < 10:
            if abs(DIFF_R[i]) < 0.4:
                j_R.append(i)
    j_R = [j_R[0], j_R[-1]]
    ID_2 = DISTANCE_IDR[j_R[0]]
    ID_3 = j_R[0]
    # Get posterior fontanel points
    Posterior_fontanel_points = []
    for i in range(ID_1, ID_2 + 1):
        Posterior_fontanel_points.append(Suture7[i, :])
    for i in range(ID_3, Id_5 - 1, -1):
        Posterior_fontanel_points.append(Suture4[i, :])
    for i in range(Id_4, ID_6 + 1):
        Posterior_fontanel_points.append(Suture3[i, :])
    Posterior_fontanel_points.append(Suture7[ID_1, :])
    Posterior_fontanel_points = np.array(Posterior_fontanel_points)
    points = Posterior_fontanel_points
    import matlab.engine

    engine = matlab.engine.start_matlab()  # start matlab process
    # engine = matlab.engine.start_matlab("-desktop")  # start matlab process with graphic UI
    # needs to transfer the data format from python to matlab
    m_points = matlab.double(points.tolist())
    Posterior_fontanel_curv = engine.Curv_ZX_plane_posterior(m_points, 0)

    return Posterior_fontanel_curv

def Curv_Sphenoid(PCA_mean_rebuild,ID):
    # Left
    # Left coronal suture
    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # Left Coronal Suture
    DISTANCE = []
    DISTANCE_ID = []
    DIFF = []
    for i in range(np.shape(Suture2)[0]):  # 计算每个2suture的点到suture3的距离
        POINT = Suture2[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID = np.array(DISTANCE_ID)
    DIFF = np.array(DIFF)
    j = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                j.append(i)
    POINT = Suture3[DISTANCE_ID[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture2)
    j.append(POINTB_ID)

    # left squamosal suture
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    for i in range(np.shape(Suture1)[0]):  # 计算每个suture1的点到suture4的距离
        POINT = Suture1[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    jj = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                jj.append(i)
    POINT = Suture4[DISTANCE_ID1[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture1)
    jj.append(POINTB_ID)
    j = [j[0], j[-1]]
    jj = [jj[0], jj[-1]]

    POINT_3 = Suture3[DISTANCE_ID[j[-1]], :]
    Suture5 = []
    Suture6 = []
    for i in range(50):
        Suture5.append([PCA_mean_rebuild[3 * i + 1800], PCA_mean_rebuild[3 * i + 1801], PCA_mean_rebuild[3 * i + 1802]])
        Suture6.append([PCA_mean_rebuild[3 * i + 1950], PCA_mean_rebuild[3 * i + 1951], PCA_mean_rebuild[3 * i + 1952]])
    Suture5 = np.array(Suture5)
    Suture6 = np.array(Suture6)
    useless, POINTB_ID = Find_point_in_line(POINT_3, Suture5)
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    jjj = []
    for i in range(0, 50):  # 计算每个suture5的点到suture3的距离
        POINT = Suture5[i, :]
        temp1, POINTB_ID1 = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID1)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    for i in range(49):
        if DISTANCE[i] < DISTANCE[0] / 2:
            if abs(DIFF[i]) < 0.3:
                jjj.append(i)
            else:
                if i > 0 and abs(DIFF[i] + DIFF[i - 1]) < 0.3:
                    jjj.append(i)

    # Get Left Sphenoid suture points
    Left_Sphenoid_fontanel_points = []
    for i in range(j[-1], 100):
        Left_Sphenoid_fontanel_points.append(Suture2[i, :])
    for i in range(0, jjj[0] + 1):
        Left_Sphenoid_fontanel_points.append(Suture5[i, :])
    for i in range(DISTANCE_ID1[jjj[0]], DISTANCE_ID[j[-1]] + 1):
        Left_Sphenoid_fontanel_points.append(Suture3[i, :])
    Left_Sphenoid_fontanel_points.append(Suture2[j[-1], :])
    Left_Sphenoid_fontanel_points = np.array(Left_Sphenoid_fontanel_points)
    points = Left_Sphenoid_fontanel_points
    import matlab.engine
    engine = matlab.engine.start_matlab()  # start matlab process
    # engine = matlab.engine.start_matlab("-desktop")  # start matlab process with graphic UI
    # needs to transfer the data format from python to matlab
    m_points = matlab.double(points.tolist())
    Left_Sphenoid_fontanel_curv = engine.Curv_YZ_plane_left(m_points, 0)

    # Right sphenoid fontanel
    # Right coronal suture
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    for i in range(np.shape(Suture1)[0]):  # 计算每个suture1的点到suture4的距离
        POINT = Suture1[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    jj = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                jj.append(i)
    POINT = Suture4[DISTANCE_ID1[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture1)
    jj.append(POINTB_ID)
    # 只保留首尾
    j = [j[0], j[-1]]
    jj = [jj[0], jj[-1]]

    # right squamosal suture
    jjjj = []
    DISTANCE_ID2 = []
    DISTANCE = []
    DIFF = []
    for i in range(0, 50):  # 计算每个suture6的点到suture4的距离
        POINT = Suture6[i, :]
        temp1, POINTB_ID1 = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID2.append(POINTB_ID1)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID2 = np.array(DISTANCE_ID2)
    DIFF = np.array(DIFF)
    for i in range(49):
        if DISTANCE[i] < DISTANCE[0] / 2:
            if abs(DIFF[i]) < 0.3:
                jjjj.append(i)
            else:
                if i > 0 and abs(DIFF[i] + DIFF[i - 1]) < 0.3:
                    jjjj.append(i)

    jjj = [jjj[0], jjj[-1]]
    jjjj = [jjjj[0], jjjj[-1]]

    Right_sphenoid_fontanel_points = []
    for i in range(jj[-1], 100):
        Right_sphenoid_fontanel_points.append(Suture1[i, :])
    for i in range(0, jjjj[0] + 1):
        Right_sphenoid_fontanel_points.append(Suture6[i, :])
    for i in range(DISTANCE_ID2[jjjj[0]], DISTANCE_ID1[jj[-1]] + 1):
        Right_sphenoid_fontanel_points.append(Suture4[i, :])
    Right_sphenoid_fontanel_points.append(Suture1[jj[-1], :])

    Right_sphenoid_fontanel_points = np.array(Right_sphenoid_fontanel_points)
    points = Right_sphenoid_fontanel_points
    import matlab.engine
    engine = matlab.engine.start_matlab()  # start matlab process
    # engine = matlab.engine.start_matlab("-desktop")  # start matlab process with graphic UI
    # needs to transfer the data format from python to matlab
    m_points = matlab.double(points.tolist())
    Right_Sphenoid_fontanel_curv = engine.Curv_YZ_plane_right(m_points, 0)
    return Left_Sphenoid_fontanel_curv, Right_Sphenoid_fontanel_curv

def Curv_Mastoid(PCA_mean_rebuild,ID):
    # Left
    # left squamosal suture
    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)
    Suture3 = []
    Suture4 = []
    for i in range(200):
        Suture3.append([PCA_mean_rebuild[3 * i + 600], PCA_mean_rebuild[3 * i + 601], PCA_mean_rebuild[3 * i + 602]])
        Suture4.append([PCA_mean_rebuild[3 * i + 1200], PCA_mean_rebuild[3 * i + 1201], PCA_mean_rebuild[3 * i + 1202]])
    Suture3 = np.array(Suture3)
    Suture4 = np.array(Suture4)
    # Left Coronal Suture
    DISTANCE = []
    DISTANCE_ID = []
    DIFF = []
    for i in range(np.shape(Suture2)[0]):  # 计算每个suture2的点到suture3的距离
        POINT = Suture2[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID = np.array(DISTANCE_ID)
    DIFF = np.array(DIFF)
    j = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                j.append(i)
    POINT = Suture3[DISTANCE_ID[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture2)
    j.append(POINTB_ID)

    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    for i in range(np.shape(Suture1)[0]):  # 计算每个suture1的点到suture4的距离
        POINT = Suture1[i, :]
        temp1, POINTB_ID = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    jj = []
    for i in range(99):
        if DISTANCE[i] < 10:
            if abs(DIFF[i]) < 0.4:
                jj.append(i)
    POINT = Suture4[DISTANCE_ID1[-1], :]
    temp1, POINTB_ID = Find_point_in_line(POINT, Suture1)
    jj.append(POINTB_ID)
    j = [j[0], j[-1]]
    jj = [jj[0], jj[-1]]

    POINT_3 = Suture3[DISTANCE_ID[j[-1]], :]
    Suture5 = []
    Suture6 = []
    for i in range(50):
        Suture5.append([PCA_mean_rebuild[3 * i + 1800], PCA_mean_rebuild[3 * i + 1801], PCA_mean_rebuild[3 * i + 1802]])
        Suture6.append([PCA_mean_rebuild[3 * i + 1950], PCA_mean_rebuild[3 * i + 1951], PCA_mean_rebuild[3 * i + 1952]])
    Suture5 = np.array(Suture5)
    Suture6 = np.array(Suture6)
    useless, POINTB_ID = Find_point_in_line(POINT_3, Suture5)
    DISTANCE = []
    DISTANCE_ID1 = []
    DIFF = []
    jjj = []
    for i in range(0, 50):  # 计算每个suture5的点到suture3的距离
        POINT = Suture5[i, :]
        temp1, POINTB_ID1 = Find_point_in_line(POINT, Suture3)
        DISTANCE.append(temp1)
        DISTANCE_ID1.append(POINTB_ID1)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID1 = np.array(DISTANCE_ID1)
    DIFF = np.array(DIFF)
    for i in range(49):
        if DISTANCE[i] < DISTANCE[0] / 2:
            if abs(DIFF[i]) < 0.3:
                jjj.append(i)
            else:
                if i > 0 and abs(DIFF[i] + DIFF[i - 1]) < 0.3:
                    jjj.append(i)

    # Left lambdoidal suture
    Suture7 = []
    for i in range(100):
        Suture7.append(
            [PCA_mean_rebuild[3 * i + 2100], PCA_mean_rebuild[3 * i + 2101], PCA_mean_rebuild[3 * i + 2102]])
    Suture7 = np.array(Suture7)
    DISTANCE_L = []
    DISTANCE_IDL = []
    DIFF_L = []
    j_L = []
    for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
        POINT_L = Suture3[i, :]
        temp1_L, POINTB_L = Find_point_in_line(POINT_L, Suture7)
        DISTANCE_L.append(temp1_L)
        DISTANCE_IDL.append(POINTB_L)
        if i > 0:
            diff111 = DISTANCE_L[i] - DISTANCE_L[i - 1]
            DIFF_L.append(diff111)
    DISTANCE_L = np.array(DISTANCE_L)
    DISTANCE_IDL = np.array(DISTANCE_IDL)
    DIFF_L = np.array(DIFF_L)
    for i in range(199):
        if DISTANCE_L[i] < 10:
            if abs(DIFF_L[i]) < 0.4:
                j_L.append(i)
    j_L = [j_L[0], j_L[-1]]

    Left_mastoid_fontanel = []
    for i in range(jjj[-1], 50):
        Left_mastoid_fontanel.append(Suture5[i, :])
    for i in range(0, DISTANCE_IDL[j_L[-1]] + 1):
        Left_mastoid_fontanel.append(Suture7[i, :])
    for i in range(j_L[-1], DISTANCE_ID1[jjj[-1]] + 1):
        Left_mastoid_fontanel.append(Suture3[i, :])

    Left_mastoid_fontanel = np.array(Left_mastoid_fontanel)
    points = Left_mastoid_fontanel
    import matlab.engine
    engine = matlab.engine.start_matlab()  # start matlab process
    # engine = matlab.engine.start_matlab("-desktop")  # start matlab process with graphic UI
    # needs to transfer the data format from python to matlab
    m_points = matlab.double(points.tolist())
    Left_mastoid_fontanel_curv = engine.Curv_YZ_plane_left(m_points, 0)
    # Right mastoid

    # right squamosal suture
    jjjj = []
    DISTANCE_ID2 = []
    DISTANCE = []
    DIFF = []
    for i in range(0, 50):  # 计算每个suture6的点到suture4的距离
        POINT = Suture6[i, :]
        temp1, POINTB_ID1 = Find_point_in_line(POINT, Suture4)
        DISTANCE.append(temp1)
        DISTANCE_ID2.append(POINTB_ID1)
        if i > 0:
            diff11 = DISTANCE[i] - DISTANCE[i - 1]
            DIFF.append(diff11)
    DISTANCE = np.array(DISTANCE)
    DISTANCE_ID2 = np.array(DISTANCE_ID2)
    DIFF = np.array(DIFF)
    for i in range(49):
        if DISTANCE[i] < DISTANCE[0] / 2:
            if abs(DIFF[i]) < 0.3:
                jjjj.append(i)
            else:
                if i > 0 and abs(DIFF[i] + DIFF[i - 1]) < 0.3:
                    jjjj.append(i)

    jjj = [jjj[0], jjj[-1]]
    jjjj = [jjjj[0], jjjj[-1]]

    # Right lambdoidal suture
    DISTANCE_R = []
    DISTANCE_IDR = []
    DIFF_R = []
    j_R = []
    for i in range(0, 200):  # 计算每个suture3的点到suture7的距离
        POINT_R = Suture4[i, :]
        temp1_R, POINTB_R = Find_point_in_line(POINT_R, Suture7)
        DISTANCE_R.append(temp1_R)
        DISTANCE_IDR.append(POINTB_R)
        if i > 0:
            diff111 = DISTANCE_R[i] - DISTANCE_R[i - 1]
            DIFF_R.append(diff111)
    DISTANCE_R = np.array(DISTANCE_R)
    DISTANCE_IDR = np.array(DISTANCE_IDR)
    DIFF_R = np.array(DIFF_R)
    for i in range(199):
        if DISTANCE_R[i] < 10:
            if abs(DIFF_R[i]) < 0.4:
                j_R.append(i)
    j_R = [j_R[0], j_R[-1]]

    Right_mastoid_fontanel = []
    for i in range(jjjj[-1], 50):
        Right_mastoid_fontanel.append(Suture6[i, :])
    for i in range(99, DISTANCE_IDR[j_R[-1]] - 1, -1):
        Right_mastoid_fontanel.append(Suture7[i, :])
    for i in range(j_R[-1], DISTANCE_ID2[jjjj[-1]] + 1):
        Right_mastoid_fontanel.append(Suture4[i, :])

    Right_mastoid_fontanel = np.array(Right_mastoid_fontanel)
    points = Right_mastoid_fontanel
    import matlab.engine
    engine = matlab.engine.start_matlab()  # start matlab process
    # engine = matlab.engine.start_matlab("-desktop")  # start matlab process with graphic UI
    # needs to transfer the data format from python to matlab
    m_points = matlab.double(points.tolist())
    Right_mastoid_fontanel_curv = engine.Curv_YZ_plane_right(m_points, 0)
    return Left_mastoid_fontanel_curv,Right_mastoid_fontanel_curv

def Metopic_Suture_pred(PCA_mean_rebuild,ID):
    Suture1 = []
    Suture2 = []
    for i in range(100):
        Suture1.append([PCA_mean_rebuild[3 * i], PCA_mean_rebuild[3 * i + 1], PCA_mean_rebuild[3 * i + 2]])
        Suture2.append([PCA_mean_rebuild[3 * i + 300], PCA_mean_rebuild[3 * i + 301], PCA_mean_rebuild[3 * i + 302]])
    Suture1 = np.array(Suture1)
    Suture2 = np.array(Suture2)
    # 2. Get the distance between two nodes
    Temp = []
    for i in range(100):
        Temp.append(distance(Suture1[i, :], Suture2[i, :]))
    Temp = np.array(Temp)
    diff = []
    # Measure the Metopic suture
    # Width
    Metopic_Suture_width = np.mean(Temp[1:ID + 1])
    # Length
    Metopic_Suture_length1 = 0
    Metopic_Suture_length2 = 0
    for i in range(ID):
        Metopic_Suture_length1 = Metopic_Suture_length1 + distance(Suture1[i, :], Suture1[i + 1, :])
        Metopic_Suture_length2 = Metopic_Suture_length2 + distance(Suture2[i, :], Suture2[i + 1, :])
    Metopic_Suture_length = (Metopic_Suture_length2 + Metopic_Suture_length1) / 2
    Begining_point = np.mean([Suture1[0, :], Suture2[0, :]], axis=0)
    Ending_point = np.mean([Suture1[ID, :], Suture2[ID, :]], axis=0)
    Metopic_chord_length = distance(Begining_point, Ending_point)
    SI = Metopic_Suture_length/Metopic_chord_length
    return Metopic_Suture_length, Metopic_Suture_width,SI