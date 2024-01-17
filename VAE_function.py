from tensorflow.keras import backend as K
import numpy as np
def sampling(args):
    z_mean = args[0]
    z_log_sigma = args[1]
    # z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 30),
                              mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_sigma) * epsilon