'''
@author: Robin Schumacher

Programm, um gespeicherte Modelle, deren Historie und Config zu laden.
'''
from tensorflow import keras
import pickle
import json

class Loader:
    def __init__(self, modelId):
        self.folderModel = "C:/Users/robsc/Documents/Studium/Semester 6/Data Science/Portfolio/output/" + modelId

    def loadModel(self, modelname):
        model = keras.models.load_model(self.folderModel + '/' + modelname + '.h5')
        return model 

    def loadHistory(self, historyname):
        with open(self.folderModel + '/' + historyname, 'rb') as file:
            history = pickle.load(file)
            return history

    def loadModelConfig(self):
        with open(self.folderModel + '/config.json', 'r') as openfile:
            config = json.load(openfile)
            return config
