# Read the csv file
import pandas as pd
import relevant
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import Visualization
import Evaluation
import VAE_function
import os

#  Read the data
df = pd.read_csv('X:/Siyuanch/Project2/DATA.csv')
# Relevance analysis (features engineering process)
data = relevant.analysis(df, 'Record_ID', False)  # discard the column of Record_ID

# Sliding data
## Replace the data with slided semilandmark for PCA
slided_data = pd.read_csv('X:/Siyuanch/Project2/COMMENTS_FROM_JOURNAL_ANATOMY/Python_code/new_data_after_sliding.csv')
slided_data = slided_data.to_numpy()
data = slided_data


##
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
       64,30,18,38,5,40,62,29,32,34,3,67,8,10,19,39,47,15,20,16,44,55]#andom order

new_data = relevant.random(Re_data,per)

## Data standardization
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()# Same with scale
data_s = standardizer.fit_transform(new_data)
################################
Error_LSD_train = []
Error_LMD_train = []
Error_LSD_test = []
Error_LMD_test = []

E_LSD_train_List = []
E_LMD_train_List = []
E_LSD_test_List = []
E_LMD_test_List = []

## Ten floder



n = 2400

latent_dim = 30
beta = 1e-3
base = 16
conv_size = 3
padding_size = 2
st = 1
epoch = 4000
act = 'tanh'
lr = 0.0001
name = 'VAE'
rank = 'TRY'
model_name = f'_{name}.h5'
act = 'tanh'

Epochs_LSD_train_List = []
Epochs_LMD_train_List = []
Epochs_LSD_test_List = []
Epochs_LMD_test_List = []


for i in range(10):
    [train, test] = relevant.ten_fold(data_s, i)
    folder_name = f'VAE_{i}'
    wdir = f'./{folder_name}/'
    # encoder = models.load_model('en_try_VAE.h5', custom_objects={'sampling': sampling})
    # print(encoder.summary())

    inp = layers.Input(shape=n)
    x = layers.Dense(64, activation='relu')(inp)
    # x = layers.Dense(base * 8, activation='relu')(x)  # 128
    z_mean = layers.Dense(latent_dim)(x)
    z_log_sigma = layers.Dense(latent_dim)(x)
    z = layers.Lambda(VAE_function.sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

    ##########Encoder
    encoder = models.Model(inp, z)
    print(encoder.summary())
    ############
      ######## Decoder
    code_i = layers.Input(shape=latent_dim)
    x = layers.Dense(64, activation=act)(code_i)
    out = layers.Dense(2400, activation='sigmoid')(x)
    decoder = models.Model(code_i, out)
    print(decoder.summary())

    # print(K.reshape(inp, (-1,)))

    out_d = decoder(encoder(inp))

    model = models.Model(inp, out_d)
    print(model.summary())

##### LOSS
    rec_loss = losses.mse(K.reshape(inp, (-1,)), K.reshape(out_d, (-1,)))
    # rec_loss = losses.MeanSquaredError(inp,out_d)
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)  ############ sum or mean
    kl_loss *= -0.5
    vae_loss = K.mean(rec_loss + beta * kl_loss)
    model.add_loss(vae_loss)
    model.add_metric(rec_loss, name='rec_loss',
                     aggregation='mean')  ## metric is just the print metric not involved in the training
    model.add_metric(kl_loss, name='kl_loss', aggregation='mean')

    opt = optimizers.Adam(learning_rate=lr)
    # model.compile(optimizer = opt)
    model.compile(optimizer=opt)

    train_data = train[:, :]
    test_data = test[:,:]
    hist = model.fit(x=train_data, y=None, epochs=epoch)  # train_data

    os.makedirs(wdir, exist_ok=True)
    savemat(wdir + f'CNNVAEloss_{rank}.mat', hist.history)

    encoder.save(wdir + 'en' + model_name)
    decoder.save(wdir + 'de' + model_name)

## -------------------------------------After training -------------------------------------------------------------------
    # Loss
    # plt.figure(figsize=(4, 4))
    # plt.title("Learning curve")
    # plt.plot(hist.history["loss"], label="loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss Value")
    # plt.legend()
    # plt.show()

    train_encode = encoder.predict(train_data)
    train_predict = decoder.predict(train_encode)
    train_predict = np.squeeze(train_predict)
    rebuild_train = standardizer.inverse_transform(train_predict)

    test_encode = encoder.predict(test_data)
    test_predict = decoder.predict(test_encode)
    test_predict = np.squeeze(test_predict)
    rebuild_test = standardizer.inverse_transform(test_predict)

    real_test = standardizer.inverse_transform(test)
    real_train = standardizer.inverse_transform(train)
    ### Evaluate
    LSD_test = Evaluation.LSD(real_test, rebuild_test)
    LMD_test = Evaluation.LMD(real_test, rebuild_test)
    LSD_train = Evaluation.LSD(real_train, rebuild_train)
    LMD_train = Evaluation.LMD(real_train, rebuild_train)

    Mean_LSD_train = np.mean(LSD_train)
    Mean_LMD_train = np.mean(LMD_train)
    Mean_LSD_test = np.mean(LSD_test)
    Mean_LMD_test = np.mean(LMD_test)

    Error_LSD_train = np.append(Error_LSD_train, Mean_LSD_train)
    Error_LMD_train = np.append(Error_LMD_train, Mean_LMD_train)
    Error_LSD_test = np.append(Error_LSD_test, Mean_LSD_test)
    Error_LMD_test = np.append(Error_LMD_test, Mean_LMD_test)
# Plot
print('mean LMD train error = %f' % np.mean(Error_LMD_train))
print('mean LMD test error = %f' % np.mean(Error_LMD_test))
print('done')



