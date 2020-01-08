#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:28:55 2019

@author: kkottmann
"""

import numpy as np
from matplotlib import pyplot as plt
from tenpy.networks.mps import MPS
from tenpy.models.hubbard import BoseHubbardChain
from tenpy.algorithms import dmrg

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

import datetime # datetime.datetime.now()   


__all__ = [
    "hubbard_dmrg","training","norm2","eval_loss"
]

def hubbard_dmrg(L,U,V,init="11",t=1.,mu=0,n_max=3,conserve="N",chi_max=100,bc="finite",extra_fill=0):
    """ Run DMRG for the Bose Hubbard model """
    # Setting the initial state
    where = int(L/3)
    if where % 2:
        where += 1
    if where > L:
        where = 1
    if init == "11":
        init_config = [1] * L
        init_config[where] += extra_fill
    if init == "20":
        init_config = [2,0] * int(L/2)
        init_config[where] += extra_fill
    if init == "02":
        init_config = [0,2] * int(L/2)
        init_config[where] += extra_fill
    filling = np.sum(init_config)/L
    t0 = datetime.datetime.now()
    model_params = dict(
        filling = filling,
        n_max = n_max,
        t = t,
        U = U,
        V = V,
        mu = mu,
        L=L,
        bc_MPS=bc,
        conserve = conserve,
        verbose=0)
    M = BoseHubbardChain(model_params)
    psi = MPS.from_product_state(M.lat.mps_sites(), init_config, bc=bc)
    dmrg_params = {
        'mixer': None,
        'trunc_params': {
            'chi_max': chi_max,
            'svd_min': 1.e-10
        },
        'max_E_err': 1.e-10,
        'verbose': 0,
        "norm_tol":1e-5
    }
    eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    eng.reset_stats()
    eng.trunc_params["chi_max"] = chi_max
    eng.run()
    E = np.sum(psi.expectation_value(M.H_bond[1:]))
    print("E = {E:.13f}".format(E=E))
    #print("final bond dimensions: ", psi.chi)
    time = datetime.datetime.now() - t0
    print(time)
    params = model_params
    params["chi_max"] = chi_max
    params["chis"] = psi.chi
    params["E"] = E
    params["time"] = time
    params["init_config"] = init_config
    params["init"] = init
    return psi, params

def norm2(y_true,y_pred):
    return np.sqrt(np.sum(np.abs(y_true - y_pred)**2))

def eval_loss(x_batch,y_batch,norm=norm2):
    a = []
    for i in range(x_batch.shape[0]):
        a.append(norm(x_batch[i],y_batch[i]))
    return np.array(a)

def training(x_train,choose_cnn, name = "", provide_cnn = False,
             load_prev=False, num_epochs = 10,  verbose_val=1, batch_size = 128, shuffle=True, early=False,
             loss="mse", activation0 = 'relu', activation = 'linear', optimizer = "adam"):
    """
    provide_cnn is to continue the training of a cnn
    load_prev is to load a previously trained network under the same paramters
    training:: [(Vmin,Vmax),(Umin,Umax)] for training
    """
        
    name_string = choose_cnn.__name__ + "_" + str(activation0) + "_" + str(activation) + "_" + str(optimizer) + "_" + name
    CNN_filepath= 'CNN_data/' + name_string + 'weights.hdf5'

    cnn = choose_cnn(loss,optimizer,activation0,activation,x_train.shape[1:])
    if load_prev:
        cnn.load_weights(CNN_filepath)
    else:
        if provide_cnn:
            cnn = provide_cnn
        # checkpoint
        checkpoint = ModelCheckpoint(CNN_filepath, monitor='val_loss', verbose=0, save_best_only=False, mode='max')
        callbacks_list = [checkpoint]
        if early:
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto', baseline=None)#, restore_best_weights=True)
            callbacks_list.append(early_stop)

        history=cnn.fit(x_train, x_train, epochs=num_epochs, batch_size=batch_size,validation_data=(x_train,x_train),callbacks=callbacks_list,verbose=verbose_val,shuffle=shuffle)
        np.savez('CNN_data/history_cnn_' + name_string + '.npz',loss=history.history['loss'],val_loss=history.history['val_loss'])#,acc=history.history['acc'],val_acc=history.history['val_acc'])

    # history
    plotname = 'plots/training_history' +  name_string
    if not load_prev:
        plt.plot(history.history['loss'], linewidth=2, label='Train')
        plt.plot(history.history['val_loss'], linewidth=2, label='Val')
        plt.legend(loc='upper right')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        #plt.ylim(ymin=0.70,ymax=1)
        plt.savefig(plotname + 'training.png', format="png")
        plt.show()
    return cnn