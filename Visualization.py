import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D

def scatter_(data, i ,j,CONTEXT):
    i_sample = data[i, :]
    X = np.zeros((j))
    Y = np.zeros((j))
    Z = np.zeros((j))

    for i in range(j):
        X[i] = i_sample[i * 3 + 0]
        Y[i] = i_sample[i * 3 + 1]
        Z[i] = i_sample[i * 3 + 2]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(X, Y, Z, marker='o', color='c', alpha=0.5, linewidths = 0.1)
    ax.set_title(CONTEXT)
    plt.show()



def single_line(data, i, j, CONTEXT):
    if data.shape[0] == 2400:
        i_sample = data
    else:
        i_sample = data[i, :]
    X = np.zeros((j))
    Y = np.zeros((j))
    Z = np.zeros((j))

    for iii in range(j):
        X[iii] = i_sample[iii * 3 + 0]
        Y[iii] = i_sample[iii * 3 + 1]
        Z[iii] = i_sample[iii * 3 + 2]
    X1_new = X[0:100]
    X2_new = X[100:200]
    X3_new = X[200:400]
    X4_new = X[400:600]
    X5_new = X[600:650]
    X6_new = X[650:700]
    X7_new = X[700:800]
    X_new = np.hstack((X2_new, X5_new, X7_new, X6_new[::-1], X1_new[::-1], X2_new[0]))  #
    Y1_new = Y[0:100]
    Y2_new = Y[100:200]
    Y3_new = Y[200:400]
    Y4_new = Y[400:600]
    Y5_new = Y[600:650]
    Y6_new = Y[650:700]
    Y7_new = Y[700:800]

    Y_new = np.hstack((Y2_new, Y5_new, Y7_new, Y6_new[::-1], Y1_new[::-1], Y2_new[0]))
    Z1_new = Z[0:100]
    Z2_new = Z[100:200]
    Z3_new = Z[200:400]
    Z4_new = Z[400:600]
    Z5_new = Z[600:650]
    Z6_new = Z[650:700]
    Z7_new = Z[700:800]
    Z_new = np.hstack((Z2_new, Z5_new, Z7_new, Z6_new[::-1], Z1_new[::-1], Z2_new[0]))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca(projection='3d')
            ## Axes3D doesn't have the title
            # ax = Axes3D(fig)
    # plt.figure('SINGLE_LINE')
    ax.plot(X_new, Y_new, Z_new, color="blue")
    ax.plot(X3_new, Y3_new, Z3_new, color="blue")
    ax.plot(X4_new, Y4_new, Z4_new, color="blue")

    ax.set_title(CONTEXT)
    # plt.show()
def getXYZ(A):
    X_A = A[::3]
    Y_A = A[1::3]
    Z_A = A[2::3]
    return X_A,Y_A,Z_A

def line_(data1, i, j, CONTEXT):
    count = 0
    for ii in range(len(data1)):
        data = data1[ii]
        if data.shape[0] == 2400:
            i_sample = data
        else:
            i_sample = data[i, :]
        X = np.zeros((j))
        Y = np.zeros((j))
        Z = np.zeros((j))
        for iii in range(j):
            X[iii] = i_sample[iii * 3 + 0]
            Y[iii] = i_sample[iii * 3 + 1]
            Z[iii] = i_sample[iii * 3 + 2]
        X1_new = X[0:100]
        X2_new = X[100:200]
        X3_new = X[200:400]
        X4_new = X[400:600]
        X5_new = X[600:650]
        X6_new = X[650:700]
        X7_new = X[700:800]
        X_new = np.hstack((X2_new, X5_new, X7_new, X6_new[::-1], X1_new[::-1], X2_new[0]))  #
        Y1_new = Y[0:100]
        Y2_new = Y[100:200]
        Y3_new = Y[200:400]
        Y4_new = Y[400:600]
        Y5_new = Y[600:650]
        Y6_new = Y[650:700]
        Y7_new = Y[700:800]
        Y_new = np.hstack((Y2_new, Y5_new, Y7_new, Y6_new[::-1], Y1_new[::-1], Y2_new[0]))
        Z1_new = Z[0:100]
        Z2_new = Z[100:200]
        Z3_new = Z[200:400]
        Z4_new = Z[400:600]
        Z5_new = Z[600:650]
        Z6_new = Z[650:700]
        Z7_new = Z[700:800]
        Z_new = np.hstack((Z2_new, Z5_new, Z7_new, Z6_new[::-1], Z1_new[::-1], Z2_new[0]))
        if count == 0:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.gca(projection='3d')
            ax.view_init(azim=-50, elev=15)
            ax.plot(X_new, Y_new, Z_new, color="blue")  # yellow green
            ax.plot(X3_new, Y3_new, Z3_new, color="blue")
            ax.plot(X4_new, Y4_new, Z4_new, color="blue")
            # ax.plot(X_new, Y_new, Z_new, color="red")
            # ax.plot(X3_new, Y3_new, Z3_new, color="red")
            # ax.plot(X4_new, Y4_new, Z4_new, color="red")
            # ax.plot(X_new, Y_new, Z_new, color="orange")
            # ax.plot(X3_new, Y3_new, Z3_new, color="orange")
            # ax.plot(X4_new, Y4_new, Z4_new, color="orange")
            # count = count + 1
        if count == 1:
            ax.plot(X_new, Y_new, Z_new, color="deeppink")  # yellow green
            ax.plot(X3_new, Y3_new, Z3_new, color="deeppink")
            ax.plot(X4_new, Y4_new, Z4_new, color="deeppink")
            # ax.plot(X_new, Y_new, Z_new, color="green")
            # ax.plot(X3_new, Y3_new, Z3_new, color="green")
            # ax.plot(X4_new, Y4_new, Z4_new, color="green")  # green blue
            # ax.plot(X_new, Y_new, Z_new, color="red")
            # ax.plot(X3_new, Y3_new, Z3_new, color="red")
            # ax.plot(X4_new, Y4_new, Z4_new, color="red")
            # ax.plot(X_new, Y_new, Z_new, color="yellow") #yellow green
            # ax.plot(X3_new, Y3_new, Z3_new, color="yellow")
            # ax.plot(X4_new, Y4_new, Z4_new, color="yellow")
            # count = count + 1
        if count == 2:
            # ax.plot(X_new, Y_new, Z_new, color="yellow") #yellow green
            # ax.plot(X3_new, Y3_new, Z3_new, color="yellow")
            # ax.plot(X4_new, Y4_new, Z4_new, color="yellow")
            # ax.plot(X_new, Y_new, Z_new, color="blue")  # yellow green
            # ax.plot(X3_new, Y3_new, Z3_new, color="blue")
            # ax.plot(X4_new, Y4_new, Z4_new, color="blue")
            ax.plot(X_new, Y_new, Z_new, color="green")
            ax.plot(X3_new, Y3_new, Z3_new, color="green")
            ax.plot(X4_new, Y4_new, Z4_new, color="green")
            # count = count + 1
        if count == 3:
            ax.plot(X_new, Y_new, Z_new, color="blue")  # yellow green
            ax.plot(X3_new, Y3_new, Z3_new, color="blue")
            ax.plot(X4_new, Y4_new, Z4_new, color="blue")
            # ax.plot(X_new, Y_new, Z_new, color="purple")
            # ax.plot(X3_new, Y3_new, Z3_new, color="purple")
            # ax.plot(X4_new, Y4_new, Z4_new, color="purple")
            # count = count + 1
        if count == 4:
            # ax.plot(X_new, Y_new, Z_new, color="green")
            # ax.plot(X3_new, Y3_new, Z3_new, color="green")
            # ax.plot(X4_new, Y4_new, Z4_new, color="green")
            ax.plot(X_new, Y_new, Z_new, color="purple")
            ax.plot(X3_new, Y3_new, Z3_new, color="purple")
            ax.plot(X4_new, Y4_new, Z4_new, color="purple")
            # count = count + 1
        count = count + 1


    labels = ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.tick_params(labelsize=16)
    plt.tick_params(pad=0.05)  # 通过pad参数调整距离

    # 显示图形
    # plt.show()

    ax.set_title(CONTEXT)
    # plt.show()

def single_line_nodes(data, i, j, CONTEXT,marker_X, marker_Y, marker_Z):
    if data.shape[0] == 2400:
        i_sample = data
    else:
        i_sample = data[i, :]
    X = np.zeros((j))
    Y = np.zeros((j))
    Z = np.zeros((j))

    for iii in range(j):
        X[iii] = i_sample[iii * 3 + 0]
        Y[iii] = i_sample[iii * 3 + 1]
        Z[iii] = i_sample[iii * 3 + 2]
    X1_new = X[0:100]
    X2_new = X[100:200]
    X3_new = X[200:400]
    X4_new = X[400:600]
    X5_new = X[600:650]
    X6_new = X[650:700]
    X7_new = X[700:800]
    X_new = np.hstack((X2_new, X5_new, X7_new, X6_new[::-1], X1_new[::-1], X2_new[0]))  #
    Y1_new = Y[0:100]
    Y2_new = Y[100:200]
    Y3_new = Y[200:400]
    Y4_new = Y[400:600]
    Y5_new = Y[600:650]
    Y6_new = Y[650:700]
    Y7_new = Y[700:800]

    Y_new = np.hstack((Y2_new, Y5_new, Y7_new, Y6_new[::-1], Y1_new[::-1], Y2_new[0]))
    Z1_new = Z[0:100]
    Z2_new = Z[100:200]
    Z3_new = Z[200:400]
    Z4_new = Z[400:600]
    Z5_new = Z[600:650]
    Z6_new = Z[650:700]
    Z7_new = Z[700:800]
    Z_new = np.hstack((Z2_new, Z5_new, Z7_new, Z6_new[::-1], Z1_new[::-1], Z2_new[0]))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca(projection='3d')
            ## Axes3D doesn't have the title
            # ax = Axes3D(fig)
    # plt.figure('SINGLE_LINE')
    font = {'family': 'timesnewroman',
            'color': 'black',
            'weight': 'normal',
            'size': 14}
    ax.plot(X_new, Y_new, Z_new, color="deeppink")
    ax.plot(X3_new, Y3_new, Z3_new, color="deeppink")
    ax.plot(X4_new, Y4_new, Z4_new, color="deeppink")
    ax.set_xlabel('', fontdict=font)
    ax.set_ylabel( '', fontdict=font)
    ax.set_zlabel(' ', fontdict=font)
    if marker_X.size < 15:
        ax.scatter(marker_X, marker_Y, marker_Z, marker='o', c='blue', s=50)
    else:
        ax.scatter(np.concatenate((marker_X[0:2], marker_X[18:22], marker_X[30:32])),
                   np.concatenate((marker_Y[0:2], marker_Y[18:22], marker_Y[30:32])),
                   np.concatenate((marker_Z[0:2], marker_Z[18:22], marker_Z[30:32])),
                   marker='o', c='red', s=50)
        ax.scatter(np.concatenate((marker_X[2:18], marker_X[22:30])),
                   np.concatenate((marker_Y[2:18], marker_Y[22:30])),
                   np.concatenate((marker_Z[2:18], marker_Z[22:30])),
                   marker='o', c='blue', s=50)

    ax.set_title(CONTEXT)

