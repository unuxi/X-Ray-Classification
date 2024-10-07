'''
@author: Robin Schumacher und Jakob Poley

Mit diesem Programm können Modelle evaluiert werden und zugehörige Plots werden erstellt. 
Diese werden anschließend abgespeichert.
'''
from tensorflow.python.framework.op_def_library import value_to_attr_value
import sys

sys.path.insert(0, 'C:/Users/robsc/Documents/Studium/Semester 6/Data Science/Portfolio/skripte')

from trainModel import modeltrainer as mt
from loader import Loader as ld
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
from tensorflow.keras import models
from outputmanager import OutputManager as om

class Preperator:

    def __init__(self, modelId):
        self.val_ds = mt.Preprocessing().imageImporterValidation()
        self.config = ld(modelId).loadModelConfig()
        
        #Zu beachten ist, dass die steps der Anzahl der Bilder (930) /Batch_Size entspricht
    def confusionMatrix(self, model):
        predict = model.predict(self.val_ds, steps = 930/self.config['BATCH_SIZE'])
        y = np.concatenate([y for x, y in self.val_ds], axis=0)
        confm = confusion_matrix(y, np.argmax(predict,axis=1))
        print(confm)
        return confm

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
        
    def epochsRange(self):
        epochs = self.config['EPOCHS']
        epochs_range = range(epochs)
        return epochs_range




class Plotter:

    def __init__(self, modelId):
        self.preperator = Preperator(modelId)

    def plotHistory(self, history):
        plt.figure(figsize=(20, 15))
        plt.subplot(1,2,1)
        plt.plot(self.preperator.epochsRange(), self.preperator.accuracy(history), label='Training Accuracy')
        plt.plot(self.preperator.epochsRange(), self.preperator.validatedAccuracy(history), label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy', fontsize=20)
        plt.gca().set_xlabel('Epoch', fontsize=15)
        plt.gca().set_ylabel('Accuracy', fontsize=15)
        plt.gca().tick_params(labelsize=10)

        plt.subplot(1,2,2)
        plt.plot(self.preperator.epochsRange(), self.preperator.loss(history), label='Training Loss')
        plt.plot(self.preperator.epochsRange(), self.preperator.validatedLoss(history), label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss', fontsize=20)
        plt.gca().set_xlabel('Epoch', fontsize=15)
        plt.gca().set_ylabel('Accuracy', fontsize=15)
        plt.gca().tick_params(labelsize=10)

    def plotConfusionMatrix(self, model):
        plt.figure(figsize = (10,6))
        sns.heatmap(self.preperator.confusionMatrix(model), annot = True, yticklabels=["Covid 19","Normal","Pneumonia"], xticklabels=["Covid 19","Normal","Pneumonia"], fmt="d",cmap = "Blues")

class Evaluator:

    def __init__(self) -> None:
        pass

    def evaluateHistory(self, modelId, historyname):
        history = ld(modelId).loadHistory(historyname)
        Plotter(modelId).plotHistory(history)
        om(modelId).saveFig(historyname)

    def evaluateModel(self, modelId, modelname):
        model = ld(modelId).loadModel(modelname)
        print(model)
        Plotter(modelId).plotConfusionMatrix(model)
        om(modelId).saveFig(modelname)

if __name__=="__main__":  
    #Evaluator().evaluateHistory('VGG', 'VGG_history')
    #Evaluator().evaluateHistory('DenseNet', 'DenseNet_history')
    #Evaluator().evaluateHistory('Sequential', 'Sequential_history')
    #Evaluator().evaluateModel('DenseNet', 'DenseNet_model')
    Evaluator().evaluateModel('VGG', 'VGG_model')
    #Evaluator().evaluateModel('Sequential', 'Sequential_model')

