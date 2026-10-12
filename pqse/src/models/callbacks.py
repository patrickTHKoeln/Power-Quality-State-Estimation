import os

import numpy as np
import tensorflow as tf
import math


def create_callbacks(callbacks, min_lr=1e-5, max_lr=1e-3, checkpoint_path=""):
    """
    Helper method to convert a list of callback names to a list of callback objects with default parameters
    :param callbacks: list of callback names
    :param min_lr: minimum learning rate for cyclic learning rate and reduce_lr_on_plateau
    :param max_lr: maximum learning rate for cyclic learning rate
    :param checkpoint_path: checkpoint path for saving the model weights
    :return:
    """
    cbs = []
    if 'early_stopping' in callbacks:
        cbs.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25))
    if 'reduce_lr_on_plateau' in callbacks:
        cbs.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, min_lr=min_lr))
    if 'cyclic_lr' in callbacks:
        clr = CyclicLR(min_lr=min_lr, max_lr=max_lr, step_size=200., mode='triangular')
        cbs.append(clr)
    if 'step_decay' in callbacks:
        def step_decay(epoch):
            initial_lr = 0.1
            drop = 0.5
            epochs_drop = 10
            lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
            return lr

        cbs.append(tf.keras.callbacks.LearningRateScheduler(step_decay))
    if 'minimal_loss_save' in callbacks:
        cp_callback = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                                      save_weights_only=True, initial_value_threshold=1e-4)
        cbs.append(cp_callback)
    if 'epoch_save' in callbacks:
        cp_callback = ModelCheckpoint(filepath=os.path.join(exp_path, 'weights', 'cp.ckpt'),
                                      monitor='val_loss', save_freq=50)
        cbs.append(cp_callback)
    return cbs


class CyclicLR(tf.keras.callbacks.Callback):
    def __init__(self, min_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular', gamma=0.99994):
        """
        Cyclical Learning Rate (CLR) Callback
        :param min_lr: Minimum learning rate
        :param max_lr: Maximum learning rate
        :param step_size:
        :param mode: One of {triangular, exp_range}
        :param gamma: Value for exp_range mode that determines the decay
        """
        super(CyclicLR, self).__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        base_lr = self.min_lr + (self.max_lr - self.min_lr) * np.maximum(0, (1 - x))

        if self.mode == 'triangular':
            return base_lr
        elif self.mode == 'exp_range':
            return base_lr * (self.gamma ** self.clr_iterations)

    def on_train_begin(self, logs=None):
        logs = logs or {}
        initial_lr = self.clr()
        tf.keras.backend.set_value(self.model.optimizer.lr, initial_lr)
        logs['lr'] = initial_lr  # Log the initial learning rate

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        new_lr = self.clr()
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        logs['lr'] = new_lr  # Update the learning rate in logs at the end of each batch

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Ensure the latest learning rate is logged at the end of each epoch
        current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        logs['lr'] = current_lr
