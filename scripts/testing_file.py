'''
@author: Robin Schumacher

In diesem Programm können einzelne Programmabschnitte getestet werden. Hat keine weitere relevanz
'''

from tensorflow.python.framework.op_def_library import value_to_attr_value
#import config
from trainModel import modeltrainer as mt
from loader import Loader as ld
#from plotter import preperator as prep
#from modeltrainer import Trainer
#import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
#import seaborn as sns
import numpy as np
from tensorflow.keras import models
import config

class preperator:

    def __init__(self) -> None:
        #pass
        self.val_ds = mt().Preprocessing().imageImporterValidation()

    def accuracy(self, history):
        acc = history['accuracy']
        return acc
    
    def validatedAccuracy(self, history):
        val_acc = history['val_accuracy']
        return val_acc

    def loss(self, history):
        loss = history['loss']
        return loss

    def validatedLoss(self, history):
        val_loss = history['val_loss']
        return val_loss

    def epochsRange(self, epochs = config.EPOCHS):
        epochs_range = range(epochs)
        return epochs_range

    def predict(self, m):
        predict = m.predict
        (self.val_ds, steps = 1298)
        y = np.concatenate([y for x, y in self.val_ds],  axis=0)
        confm = confusion_matrix(y, np.argmax(predict, axis=1))
        return confm


if __name__=="__main__":  
    #Evaluator().evaluateHistory('prebuild_model')
    model = ld('DenseNet').loadModel('prebuild_model')
    #history = ld('DenseNet').loadHistory('trainHistory')
    confm = preperator().confusionMatrix(model)
    #print(history)
    #acc = preperator().accuracy(history)
    #val_acc = preperator().validatedAccuracy(history)
    #loss = preperator().loss(history)
    #val_loss = preperator().validatedLoss(history)
    #epoch = preperator().epochsRange()
    print(model)
    #print(history)
    #preperator().accuracy(history)