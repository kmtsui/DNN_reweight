from __future__ import absolute_import, division, print_function

import math

import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping

print(tf.__version__)

# Use uproot for ROOT I/O 
import uproot
def getarray(filename):
    f = uproot.open(filename)
    t = f["RES"]
    array = np.transpose(np.asarray(t.arrays(["Enu","Q2","p_mu","costh_mu",
                          "W_had","gamma_had","mode","norm"], library="np", how=tuple)))
    weight = array[0,-1] # cross-section normalization which is the same for all events
    array=array[array[:,-2]==11] # select events equivalent to NEUT mode 11
    return array[:,:6], weight # only use the first 6 variables in training

array11_Eb0, w0 =getarray('nuwro_res_C_Eb0.root')
array11_Eb27, w27=getarray('nuwro_res_C_Eb27.root')

print("Number of events = ",len(array11_Eb0), "weight = ",w0)
print("Number of events = ",len(array11_Eb27), "weight = ",w27)

# Normalization layer for input variables
from tensorflow.keras.layers.experimental import preprocessing
normalizer = preprocessing.Normalization(axis=-1)
normalizer.adapt(array11_Eb0)

# set up the DNN model for training
ndim = len(array11_Eb0[0])
inputs = Input((ndim, )) 
normalized_layer = normalizer(inputs)
hidden_layer_1 = Dense(50, activation='relu')(normalized_layer)
dropout_layer_1 = Dropout(0.5)(hidden_layer_1)
hidden_layer_2 = Dense(50, activation='relu')(dropout_layer_1)
dropout_layer_2 = Dropout(0.5)(hidden_layer_2)
hidden_layer_3 = Dense(50, activation='relu')(dropout_layer_2)
dropout_layer_3 = Dropout(0.5)(hidden_layer_3)
outputs = Dense(1, activation='sigmoid')(dropout_layer_3)

model = Model(inputs=inputs, outputs=outputs)

earlystopping = EarlyStopping(patience=10, 
                              verbose=1,
                              restore_best_weights=True)

# Just use a fraction of data for training
nevents = int(len(array11_Eb0)/5)
batch_size_defined = 100 

x_data_and_MCback = np.concatenate([array11_Eb0[0:nevents],array11_Eb27[0:nevents]])
    
y_data_and_MCback = np.concatenate([np.zeros(nevents),
                                    np.ones(nevents)])
    
W_data_and_MCback = np.concatenate([ np.ones(nevents), np.ones(nevents) ])

X_train_1, X_test_1, Y_train_1, Y_test_1, w_train_1, w_test_1 = train_test_split(
    x_data_and_MCback, y_data_and_MCback, W_data_and_MCback)

# Optimizer that implements the Adam algorithm with weight decay
from tensorflow_addons.optimizers import AdamW
optimizer = AdamW(weight_decay=0.001,learning_rate=0.001)

model.compile(loss='binary_crossentropy',
              optimizer= optimizer,
              metrics=['accuracy'])


model.fit(X_train_1,
          Y_train_1,
          sample_weight=w_train_1,
          epochs=200,
          batch_size=batch_size_defined,
          validation_data=(X_test_1, Y_test_1, w_test_1),
          callbacks=[earlystopping],
          verbose=1)

# save the model for future use
model.save('DNN_Eb27', save_format='tf')

print("Finish training!!!")