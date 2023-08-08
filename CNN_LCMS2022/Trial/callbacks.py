# -*- coding: utf-8 -*-
"""
Created on Mon May 15 22:12:49 2023

@author: alvak
"""

import tensorflow as tf

class SaveModelWeightsCallback(tf.keras.callbacks.Callback):
    
    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(f'cnn_weights_{epoch:03d}.h5')