import cirq
import sympy
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq

# visualization tools
%matplotlib inline
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit
np.random.seed(1234)

import scipy.io
import numpy as np

EC=[]
x=[]
for i in range (2, 10):
  EC.append(scipy.io.loadmat('SPIS-Resting-State-Dataset/Pre-SART EEG/S0'+str(i)+'_restingPre_EC.mat'))
EC.append(scipy.io.loadmat('SPIS-Resting-State-Dataset/Pre-SART EEG/S10_restingPre_EC.mat'))
EC.append(scipy.io.loadmat('SPIS-Resting-State-Dataset/Pre-SART EEG/S11_restingPre_EC.mat'))
for j in range (0, 10):
  array = list(EC[j].items())
  x.append(array[3][1])
  
EO=[]
for i in range (2, 10):
  EO.append(scipy.io.loadmat('SPIS-Resting-State-Dataset/Pre-SART EEG/S0'+str(i)+'_restingPre_EO.mat'))
EO.append(scipy.io.loadmat('SPIS-Resting-State-Dataset/Pre-SART EEG/S10_restingPre_EO.mat'))
EO.append(scipy.io.loadmat('SPIS-Resting-State-Dataset/Pre-SART EEG/S11_restingPre_EO.mat'))
for j in range (0, 10):
  array = list(EO[j].items())
  x.append(array[3][1])
  print (x[j].shape)

def truncate_x(x_train, n_components=10):
  """Perform PCA on image dataset keeping the top `n_components` components."""
  n_points_train = tf.gather(tf.shape(x_train), 0)

  # Flatten to 1D
  x_train = tf.reshape(x_train, [n_points_train, -1])
  # Normalize.
  feature_mean = tf.reduce_mean(x_train, axis=0)
  x_train_normalized = x_train - feature_mean
  print("REEM")

  # Truncate.
  
  return x_train_normalized

DATASET_DIM = 10
x=np.array(x)
print(x.shape)
x = x.astype(np.float32)
x=x/np.max(x)
print("Number of original training examples:", len(x))
x_train = truncate_x(x, n_components=DATASET_DIM)
print(f'New datapoint dimension:', len(x_train[0]))