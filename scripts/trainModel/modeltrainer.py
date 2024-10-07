'''
@autor: Robin Schumacher und Jakob Poley

Dieses Programm sorgt dafür, dass Modelle trainiert werden. Nachdem diese trainiert wurden, werden sie abgespeichert.
'''

import tensorflow as tf
import keras
from tensorflow.keras import layers
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import efficientnet.tfkeras as efn

import sys
sys.path.insert(0, 'C:/Users/robsc/Documents/Studium/Semester 6/Data Science/Portfolio/skripte/trainModel')
import config
sys.path.insert(0, 'C:/Users/robsc/Documents/Studium/Semester 6/Data Science/Portfolio/skripte')
from outputmanager import OutputManager 
 
class Preprocessing:

    def __init__(self) -> None:
        pass

    def dataGenerator(self):
        datagenerator = ImageDataGenerator(
            rescale = 1./255,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True
            )
        return datagenerator

    def imageImporterTest(self):
        test_ds = Preprocessing.dataGenerator().flow_from_directory(
            directory = config.train_path,
            target_size = config.IMG_SIZE,
            batch_size = config.BATCH_SIZE,
            class_mode = 'categorical'
            )
        return test_ds
        

    def imageImporterTrain(self):
        train_ds = self.dataGenerator().flow_from_directory(
            directory = config.train_path,
            target_size = config.IMG_SIZE,
            batch_size = config.BATCH_SIZE,
            class_mode = 'categorical'
            )
        return train_ds

    def imageImporterValidation(self):
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory = config.val_path,
            image_size= config.IMG_SIZE,
            batch_size= config.BATCH_SIZE
            )
        return val_ds 

    def imageImporterTrainpreprocess(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            config.train_path,
            seed=123,
            image_size = config.IMG_SIZE,
            batch_size = config.BATCH_SIZE)
        return train_ds

    def autotune(self):
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = Preprocessing().imageImporterTrainpreprocess().cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = Preprocessing().imageImporterValidation().cache().prefetch(buffer_size=AUTOTUNE)
        return train_ds,val_ds


class Models:

    def __init__(self) -> None:
        pass

    def vgg(self):
        vgg = VGG16(input_shape = config.IMG_SIZE + [3], weights='imagenet', include_top=False)
        for layer in vgg.layers:
            layer.trainable = False
        x = Flatten()(vgg.output)
        prediction = Dense(config.NUM_CLASSES, activation = 'softmax')(x)
        model = Model(inputs = vgg.input, outputs = prediction)
        model.compile(optimizer='adam', loss="binary_crossentropy",metrics=['accuracy'])
        model.summary()
        return model

    def sequential(self):
        data_augmentation = keras.Sequential([
            layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(config.IMG_SIZE + [3])),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
            ])
        model = Sequential([
            data_augmentation,
            layers.experimental.preprocessing.Rescaling(1./255),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(config.NUM_CLASSES)
            ])
        model.compile(optimizer='adam', loss="binary_crossentropy",metrics=['accuracy'])
        model.summary()
        return model

    def denseNet(self):
        Network_Weight="C:/Users/robsc/Documents/Studium/Semester 6/Data Science/Portfolio/skripte/pretrained_models/DenseNet-BC-169-32-no-top.h5"
        from tensorflow.keras.applications.densenet import DenseNet169
        pre_trained_model = DenseNet169(input_shape = (config.IMG_SIZE + [3]), 
                                include_top = False, 
                                weights = None)
        pre_trained_model.load_weights(Network_Weight)
        for layer in pre_trained_model.layers:
            layer.trainable = False  #to make the layers to Freeze Weights
        pre_trained_model.summary()

        x = tf.keras.layers.Flatten()(pre_trained_model.output)
        #Full Connected Layers
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        #Add dropout to avoid Overfit
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        #Add dropout to avoid Overfit
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x= tf.keras.layers.Dense(3 , activation='sigmoid')(x)
        model = Model(pre_trained_model.input, x)
        model.compile(optimizer='adam', loss="binary_crossentropy",metrics=['accuracy'])
        model.summary()
        return model

    
class Trainer:

    def __init__(self):
        self.train_ds = Preprocessing().imageImporterTrain()
        self.val_ds = Preprocessing().imageImporterValidation()
        self.train_ds2, self.val_ds2 = Preprocessing().autotune()
        self.epochs = config.EPOCHS

    def fit(self, model):
        if model == 'Models().NachTensorflow()':
            history = model.fit(
                self.train_ds2,
                validation_data = self.val_ds2,
                epochs = self.epochs)
            return history, model

        else:
            history = model.fit(
                self.train_ds,
                validation_data = self.val_ds,
                epochs =self.epochs)
            return history, model


class ProcessorManually:
    def __init__(self, modellist, modelId):
        self.modellist = modellist
        self.output = OutputManager(modelId)

    def execute(self):
       
        for model in self.modellist:
            hist, mod = Trainer().fit(model)
            self.output.writeConfigModel()
            self.output.saveHistory(hist)
            self.output.saveModel(mod)




if __name__=="__main__":  
    model1 = Models().vgg()
    #model2 = Models().sequential()
    #model3 = Models().denseNet()

    processor = ProcessorManually([model1], 'VGG')
    processor.execute()
    