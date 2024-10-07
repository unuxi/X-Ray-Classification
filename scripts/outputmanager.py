'''
@author:Robin Schumacher

Mit diesem programm können Historien, Modelle, Plots und Configs gespeichert werden.

'''
import pickle
import matplotlib.pyplot as plt
import json
from trainModel import config

class OutputManager:
    def __init__(self, modelId):
        self.outputfolder = "C:/Users/robsc/Documents/Studium/Semester 6/Data Science/Portfolio/output/" + modelId

    def saveFig(self, figurename):
        figure=plt.gcf()
        figure.savefig(self.outputfolder + "/" + figurename + ".png")

    def saveModel(self, model, modelname):
        model.save(self.outputfolder + "/" + modelname + ".h5")

    def saveHistory(self, historyname, history):
        with open(self.outputfolder +'/' + historyname, 'wb') as file:
            pickle.dump(history.history, file, pickle.HIGHEST_PROTOCOL)

    def writeConfigModel(self):
        dict = {
            'EPOCHS': config.EPOCHS, 
            'BATCH_SIZE': config.BATCH_SIZE,
            'IMG_SIZE': config.IMG_SIZE,
        }
        with open(self.outputfolder + '/config.json', "w") as outfile:
            json.dump(dict, outfile)

