import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import json
from datetime import datetime

def graph_from_training_data(file, save=False, save_prefix=''):

  data = json.loads(open(file, "r").read())

  for index, epoch in enumerate(data['epochs']):

    metric_count = len(epoch["metrics"][0])

    fig, ax = plt.subplots(metric_count, 1)

    fig.suptitle(f'Epoch {index}')
    #fig.tight_layout(pad=3.0)
    fig.set_figheight(4 * metric_count)
    fig.set_figwidth(10)

    for k, metric in enumerate(epoch["metrics"][0]):
      
      s = map(lambda item: item[metric], epoch["metrics"])
      s = list(s)

      ax[k].plot(s)
      ax[k].plot([epoch['val_' + metric] for x in range(len(s))])
      ax[k].set(xlabel='batch count', ylabel=metric)
    
      #ax[k].grid()

    if save:
      fig.savefig(f"{save_prefix}epoch_{index}.png")
    
    plt.show()

class SaveCallback(keras.callbacks.Callback):

  def __init__(self, filename, step, epochs, batch_size, learning_rate, log_interval):


    self.step    = step    
    self.counter = 0
    self.epochs  = epochs
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.log_interval = log_interval
    self.file    = open(filename, 'a')


  def on_train_begin(self, logs=None):

    timestamp = datetime.now().timestamp()

    self.file.write(f'{"{"}"timestamp": {timestamp}, "log-interval": {self.log_interval}, "learning-rate": {self.learning_rate}, "batch-size": {self.batch_size}, "epochs": [')


  def on_train_end(self, logs=None):

    self.file.write(']}')


  def on_epoch_begin(self, epoch, logs=None):

    self.file.write('{"metrics": [')


  def on_epoch_end(self, epoch, logs=None):

    val_loss = logs['val_loss']
    val_accuracy = logs['val_accuracy']

    self.counter = 0
    self.file.write(json.dumps(logs) + f'], "val_loss": {val_loss}, "val_accuracy": {val_accuracy}' + '}')

    if epoch != self.epochs-1:
      self.file.write(',')


  def on_train_batch_end(self, batch, logs=None):

    if self.counter == 0:

      self.file.write(json.dumps(logs) + ', ')
      self.counter = 0

    else:

      self.counter = (self.counter + 1) % self.step