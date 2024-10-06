'''
@author: Robin Schumacher 

Mit diesem Programm sollen trainierte Modelle genutzt werden, um die Klassen von Bildern vorherzusagen
'''

from skimage.segmentation import mark_boundaries
import lime
from lime import lime_image
import cv2
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import os
import random

import sys
sys.path.insert(0, 'C:/Users/robsc/Documents/Studium/Semester 6/Data Science/Portfolio/skripte')
from loader import Loader as ld
from outputmanager import OutputManager as om
from trainModel import config


class Preperate:
    def __init__(self):
        self.class_dict = {0:'COVID19', 1:'NORMAL', 2:'PNEUMONIA'}
        self.data_path = config.test_path

    def findtrueClass(self, file_path):
        true_class = None
        if 'COVID19' in file_path:
            true_class = 'COVID19'
        elif 'PNEUMONIA' in file_path:
            true_class = 'PNEUMONIA'
        elif 'NORMAL' in file_path:
            true_class = 'NORMAL'
        return true_class

    def visualize(self, modelId, model, file_path, ax, text_loc, fig):
        image_size = ld(modelId).loadModelConfig()['IMG_SIZE']
        test_image = cv2.imread(self.data_path + file_path)
        test_image = cv2.resize(test_image, image_size, interpolation=cv2.INTER_NEAREST)
        test_image = np.expand_dims(test_image, axis=0)
        probs = model.predict(test_image)
        pred_class = np.argmax(probs)
        pred_class = self.class_dict[pred_class]

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(test_image[0], model.predict, top_labels=5, hide_color=0, num_samples=1000)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)

        ax.imshow(mark_boundaries(temp, mask))
        fig.text(text_loc, 0.9, "Predicted Class: " + pred_class , fontsize=13)
        true_class = Preperate().findtrueClass(file_path)
        if true_class is not None:
            fig.text(text_loc, 0.86, "Actual Class: " + true_class , fontsize=13)

class Predictor:

    def __init__(self):
        self.covid = random.choice(os.listdir(config.test_path + '/COVID19/'))
        self.normal = random.choice(os.listdir(config.test_path + '/NORMAL/'))
        self.pneumonia = random.choice(os.listdir(config.test_path + '/PNEUMONIA/'))

    def predict(self, modelId, model):
        fig,ax = plt.subplots(1,3,figsize=(18,6))
        Preperate().visualize(modelId, model, '/COVID19/' + self.covid,ax[0],0.15, fig)
        Preperate().visualize(modelId, model, '/NORMAL/' + self.normal,ax[1],0.4, fig)
        Preperate().visualize(modelId, model, '/PNEUMONIA/' + self.pneumonia,ax[2],0.7, fig)

class ProcessorManually:
    def __init__(self, modelId, modelname):
        self.modelId = modelId
        self.output = om(modelId)
        self.model = ld(modelId).loadModel(modelname)

    def execute(self):
        Predictor().predict(self.modelId, self.model)
        self.output.saveFig('predicted')


if __name__=="__main__":  
    #modelId1, modelname1 = 'DenseNet', 'DenseNet_model'
    modelId2, modelname2 = 'Sequential', 'Sequential_model'
    modelId3, modelname3 = 'VGG', 'VGG_model'


    processor = ProcessorManually(modelId3, modelname3)
    processor.execute()

