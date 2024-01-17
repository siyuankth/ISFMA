import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import Evaluation
from sklearn.decomposition import PCA, KernelPCA
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras import backend as K
import VAE_function
import os

def analysis(df,column_dp_name,PIC):
    data = df.drop(columns=[column_dp_name])
    # visualization to check the relevance
    corr = data.corr()
    corr_value = corr.values
    print('The maximum relevant coefficient is %f and the minimum value is %f' %(np.min(corr_value),np.max(corr_value)))
    upper_tri = np.tril(corr_value)
    standard = 0.7
    sum = (upper_tri > standard).sum()
    temp = upper_tri.shape[0]
    total_corr = (1 + temp) * temp / 2 - temp
    print('The number of the values of relevant coefficient over than %f is %d' % (standard, sum))
    print('The ratio of the coefficient values over %f is %f' % (standard, sum / total_corr))
    if PIC:
        sns.heatmap(data.corr(), annot=True)
        plt.show()
    return data

def random(data,per):
    # per = np.random.permutation(data.shape[0])  # 打乱后的行号
    # pseudorandom keep the per same and known
    new_data = data[per, :]
    return new_data

def ten_fold(new_data,current_index):
    K = 10
    n = len(new_data)
    if np.ndim(new_data) == 2:
        test = new_data[int(current_index*n/K):int((current_index+1)*n/K),:]
        train = np.vstack((new_data[:int(current_index * n / K), :],new_data[int((current_index + 1) * n / K):, :]))
    elif np.ndim(new_data) == 1:
        test = new_data[int(current_index * n / K):int((current_index + 1) * n / K)]
        train = np.hstack((new_data[:int(current_index * n / K)], new_data[int((current_index + 1) * n / K):]))
    print('The current index is %d' %current_index)
    print('The number of test data is %f' %(len(test)))
    print('The number of train data is %f' %(len(train)))
    return train, test

def one_fold(new_data,i):
    K = 1
    n = len(new_data)
    if np.ndim(new_data) == 2:
        test = new_data[i,:]
        train = np.vstack((new_data[:i, :],new_data[(i+1):, :]))
    elif np.ndim(new_data) == 1:
        test = new_data[i]
        train = np.hstack((new_data[:i], new_data[(i+1):]))
    print('The current index is %d' %i)
    # print('The number of test data is %f' %(len(test)))
    print('The number of train data is %f' %(len(train)))
    return train, test

def Impact_PCs(train,test,PCs_nr,standardizer):
    pca = PCA(n_components = PCs_nr)
    X = pca.fit_transform(train)
    Y_train = pca.transform(train)
    train_pred = pca.inverse_transform(Y_train)
    real_train = standardizer.inverse_transform(train)
    rebuild_train = standardizer.inverse_transform(train_pred)
    LSD_train = Evaluation.LSD(real_train, rebuild_train)
    LMD_train = Evaluation.LMD(real_train, rebuild_train)
    Y_test = pca.transform(test)
    test_pred = pca.inverse_transform(Y_test)
    # LSD and LMD
    real_test = standardizer.inverse_transform(test)
    rebuild_test = standardizer.inverse_transform(test_pred)
    LSD_test = Evaluation.LSD(real_test, rebuild_test)
    LMD_test = Evaluation.LMD(real_test, rebuild_test)
    ## Get the mean of error
    Mean_LSD_train = np.mean(LSD_train)
    Mean_LMD_train = np.mean(LMD_train)
    Mean_LSD_test = np.mean(LSD_test)
    Mean_LMD_test = np.mean(LMD_test)
    return Mean_LSD_train, Mean_LMD_train, Mean_LSD_test, Mean_LMD_test

def Impact_VAE(train,test,standardizer,latent_dim,beta,base,conv_size,padding_size,st,epoch,act,lr,n):

    inp = layers.Input(shape=(n, 1))
    x = layers.Conv1D(base, conv_size, activation=act, strides=st, padding='same')(inp)
    x = layers.MaxPooling1D(padding_size, padding='same')(x)
    x = layers.Conv1D(base * 2, conv_size, activation=act, strides=st, padding='same')(x)
    x = layers.MaxPooling1D(padding_size, padding='same')(x)
    x = layers.Conv1D(base * 4, conv_size, activation=act, strides=st, padding='same')(x)
    x = layers.MaxPooling1D(padding_size, padding='same')(x)
    x = layers.Conv1D(base * 8, conv_size, activation=act, strides=st, padding='same')(x)
    x = layers.MaxPooling1D(padding_size, padding='same')(x)
    x = layers.Conv1D(base * 16, conv_size, activation=act, strides=st, padding='same')(x)
    x = layers.MaxPooling1D(padding_size, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(base * 8, activation=act)(x)  # 128
    z_mean = layers.Dense(latent_dim)(x)
    z_log_sigma = layers.Dense(latent_dim)(x)
    z = layers.Lambda(VAE_function.sampling)([z_mean, z_log_sigma])
   ##########Encoder
    encoder = models.Model(inp, z)
    # print(encoder.summary())
    ############

    code_i = layers.Input(shape=(latent_dim,))
    x = layers.Dense(base * 8, activation=act)(code_i)
    x = layers.Dense(int(base * 16 * n / (2 ** 5)), activation=act)(x)  # 5 means the number of the conv1D
    x = layers.Reshape((int(n / (2 ** 5)), base * 16))(x)
    x = layers.UpSampling1D(padding_size)(x)
    x = layers.Conv1D(base * 16, conv_size, activation=act, strides=st, padding='same')(x)
    x = layers.UpSampling1D(padding_size)(x)
    x = layers.Conv1D(base * 8, conv_size, activation=act, strides=st, padding='same')(x)
    x = layers.UpSampling1D(padding_size)(x)
    x = layers.Conv1D(base * 4, conv_size, activation=act, strides=st, padding='same')(x)
    x = layers.UpSampling1D(padding_size)(x)
    x = layers.Conv1D(base * 2, conv_size, activation=act, strides=st, padding='same')(x)
    x = layers.UpSampling1D(padding_size)(x)
    x = layers.Conv1D(base * 1, conv_size, activation=act, strides=st, padding='same')(x)
    out = layers.Conv1D(1, conv_size, strides=st, padding='same')(x)
    decoder = models.Model(code_i, out)

    out_d = decoder(encoder(inp))

    model = models.Model(inp, out_d)
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

    train_data = train[:, :, np.newaxis]
    test_data = test[:, :, np.newaxis]
    hist = model.fit(x=train_data, y=None, epochs=epoch)  # train_data

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

    return Mean_LSD_train, Mean_LMD_train, Mean_LSD_test, Mean_LMD_test

def Impact_KPCs(train,test,PCs_nr,standardizer):

    pca = KernelPCA(n_components = PCs_nr,fit_inverse_transform=True)

    X = pca.fit_transform(train)
    Y_train = pca.transform(train)
    train_pred = pca.inverse_transform(Y_train)
    real_train = standardizer.inverse_transform(train)
    rebuild_train = standardizer.inverse_transform(train_pred)
    LSD_train = Evaluation.LSD(real_train, rebuild_train)
    LMD_train = Evaluation.LMD(real_train, rebuild_train)
    Y_test = pca.transform(test)
    test_pred = pca.inverse_transform(Y_test)
    # LSD and LMD
    real_test = standardizer.inverse_transform(test)
    rebuild_test = standardizer.inverse_transform(test_pred)
    LSD_test = Evaluation.LSD(real_test, rebuild_test)
    LMD_test = Evaluation.LMD(real_test, rebuild_test)
    ## Get the mean of error
    Mean_LSD_train = np.mean(LSD_train)
    Mean_LMD_train = np.mean(LMD_train)
    Mean_LSD_test = np.mean(LSD_test)
    Mean_LMD_test = np.mean(LMD_test)
    return Mean_LSD_train, Mean_LMD_train, Mean_LSD_test, Mean_LMD_test

def Impact_KPCs_cos(train,test,PCs_nr,standardizer):

    pca = KernelPCA(n_components = PCs_nr, kernel='cosine', fit_inverse_transform=True)

    X = pca.fit_transform(train)
    Y_train = pca.transform(train)
    train_pred = pca.inverse_transform(Y_train)
    real_train = standardizer.inverse_transform(train)
    rebuild_train = standardizer.inverse_transform(train_pred)
    LSD_train = Evaluation.LSD(real_train, rebuild_train)
    LMD_train = Evaluation.LMD(real_train, rebuild_train)
    Y_test = pca.transform(test)
    test_pred = pca.inverse_transform(Y_test)
    # LSD and LMD
    real_test = standardizer.inverse_transform(test)
    rebuild_test = standardizer.inverse_transform(test_pred)
    LSD_test = Evaluation.LSD(real_test, rebuild_test)
    LMD_test = Evaluation.LMD(real_test, rebuild_test)
    ## Get the mean of error
    Mean_LSD_train = np.mean(LSD_train)
    Mean_LMD_train = np.mean(LMD_train)
    Mean_LSD_test = np.mean(LSD_test)
    Mean_LMD_test = np.mean(LMD_test)
    return Mean_LSD_train, Mean_LMD_train, Mean_LSD_test, Mean_LMD_test

def Impact_KPCs_sig(train,test,PCs_nr,standardizer):

    pca = KernelPCA(n_components = PCs_nr, kernel='sigmoid', fit_inverse_transform=True)

    X = pca.fit(train)
    Y_train = pca.transform(train)
    train_pred = pca.inverse_transform(Y_train)
    real_train = standardizer.inverse_transform(train)
    rebuild_train = standardizer.inverse_transform(train_pred)
    LSD_train = Evaluation.LSD(real_train, rebuild_train)
    LMD_train = Evaluation.LMD(real_train, rebuild_train)
    Y_test = pca.transform(test)
    test_pred = pca.inverse_transform(Y_test)
    # LSD and LMD
    real_test = standardizer.inverse_transform(test)
    rebuild_test = standardizer.inverse_transform(test_pred)
    LSD_test = Evaluation.LSD(real_test, rebuild_test)
    LMD_test = Evaluation.LMD(real_test, rebuild_test)
    ## Get the mean of error
    Mean_LSD_train = np.mean(LSD_train)
    Mean_LMD_train = np.mean(LMD_train)
    Mean_LSD_test = np.mean(LSD_test)
    Mean_LMD_test = np.mean(LMD_test)
    return Mean_LSD_train, Mean_LMD_train, Mean_LSD_test, Mean_LMD_test

def Impact_KPCs_rbf(train,test,PCs_nr,standardizer):

    pca = KernelPCA(n_components = PCs_nr, kernel='rbf', fit_inverse_transform=True)

    X = pca.fit_transform(train)
    Y_train = pca.transform(train)
    train_pred = pca.inverse_transform(Y_train)
    real_train = standardizer.inverse_transform(train)
    rebuild_train = standardizer.inverse_transform(train_pred)
    LSD_train = Evaluation.LSD(real_train, rebuild_train)
    LMD_train = Evaluation.LMD(real_train, rebuild_train)
    Y_test = pca.transform(test)
    test_pred = pca.inverse_transform(Y_test)
    # LSD and LMD
    real_test = standardizer.inverse_transform(test)
    rebuild_test = standardizer.inverse_transform(test_pred)
    LSD_test = Evaluation.LSD(real_test, rebuild_test)
    LMD_test = Evaluation.LMD(real_test, rebuild_test)
    ## Get the mean of error
    Mean_LSD_train = np.mean(LSD_train)
    Mean_LMD_train = np.mean(LMD_train)
    Mean_LSD_test = np.mean(LSD_test)
    Mean_LMD_test = np.mean(LMD_test)
    return Mean_LSD_train, Mean_LMD_train, Mean_LSD_test, Mean_LMD_test

def Impact_KPCs_laplacian(train,test,PCs_nr,standardizer):

    pca = KernelPCA(n_components = PCs_nr, kernel='laplacian', fit_inverse_transform=True)

    X = pca.fit_transform(train)
    Y_train = pca.transform(train)
    train_pred = pca.inverse_transform(Y_train)
    real_train = standardizer.inverse_transform(train)
    rebuild_train = standardizer.inverse_transform(train_pred)
    LSD_train = Evaluation.LSD(real_train, rebuild_train)
    LMD_train = Evaluation.LMD(real_train, rebuild_train)
    Y_test = pca.transform(test)
    test_pred = pca.inverse_transform(Y_test)
    # LSD and LMD
    real_test = standardizer.inverse_transform(test)
    rebuild_test = standardizer.inverse_transform(test_pred)
    LSD_test = Evaluation.LSD(real_test, rebuild_test)
    LMD_test = Evaluation.LMD(real_test, rebuild_test)
    ## Get the mean of error
    Mean_LSD_train = np.mean(LSD_train)
    Mean_LMD_train = np.mean(LMD_train)
    Mean_LSD_test = np.mean(LSD_test)
    Mean_LMD_test = np.mean(LMD_test)
    return Mean_LSD_train, Mean_LMD_train, Mean_LSD_test, Mean_LMD_test

def Impact_KPCs_poly(train,test,PCs_nr,standardizer):

    pca = KernelPCA(n_components = PCs_nr, kernel='polynomial', degree=2, gamma=1, fit_inverse_transform=True)

    X = pca.fit_transform(train)
    Y_train = pca.transform(train)
    train_pred = pca.inverse_transform(Y_train)
    real_train = standardizer.inverse_transform(train)
    rebuild_train = standardizer.inverse_transform(train_pred)
    LSD_train = Evaluation.LSD(real_train, rebuild_train)
    LMD_train = Evaluation.LMD(real_train, rebuild_train)
    Y_test = pca.transform(test)
    test_pred = pca.inverse_transform(Y_test)
    # LSD and LMD
    real_test = standardizer.inverse_transform(test)
    rebuild_test = standardizer.inverse_transform(test_pred)
    LSD_test = Evaluation.LSD(real_test, rebuild_test)
    LMD_test = Evaluation.LMD(real_test, rebuild_test)
    ## Get the mean of error
    Mean_LSD_train = np.mean(LSD_train)
    Mean_LMD_train = np.mean(LMD_train)
    Mean_LSD_test = np.mean(LSD_test)
    Mean_LMD_test = np.mean(LMD_test)
    return Mean_LSD_train, Mean_LMD_train, Mean_LSD_test, Mean_LMD_test


def Impact_dense_VAE(train,test,standardizer,latent_dim,beta,base,conv_size,padding_size,st,epoch,act,lr,n):

    inp = layers.Input(shape=n)
    x = layers.Dense(64, activation='relu')(inp)
    # x = layers.Dense(base * 8, activation='relu')(x)  # 128
    z_mean = layers.Dense(latent_dim)(x)
    z_log_sigma = layers.Dense(latent_dim)(x)
    z = layers.Lambda(VAE_function.sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

    ##########Encoder
    encoder = models.Model(inp, z)
    # print(encoder.summary())
    ############
    ######## Decoder
    code_i = layers.Input(shape=latent_dim)
    x = layers.Dense(64, activation=act)(code_i)
    out = layers.Dense(2400, activation='sigmoid')(x)
    decoder = models.Model(code_i, out)
    # print(decoder.summary())
    out_d = decoder(encoder(inp))

    model = models.Model(inp, out_d)

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
    test_data = test[:, :]
    hist = model.fit(x=train_data, y=None, epochs=epoch)  # train_data

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

    return Mean_LSD_train, Mean_LMD_train, Mean_LSD_test, Mean_LMD_test
