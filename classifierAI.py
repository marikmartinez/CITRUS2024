import os
import tensorflow as tf
import autokeras as ak
import PIL
import numpy as np
import tifffile as tif
import json
import matplotlib.pyplot as plt

class classifierAI:
    def __init__(self): # constructor for the class
        pass

    def loadImages(self, imagesToLoad):
        pass

    def loadImagesToPredict(self, imagesToAdd):
        pass

    # loads in images, takes in 2 lists, should be the names of the images and the names of the annotation files
    def loadTrainImages(self, imagesToAdd, annotationsToAdd):
        pass

    def train(self):
        pass

    def predict(self):
        pass


class autokerasClassifierAI(classifierAI):
    def __init__(self):
         self.model = ak.ImageClassifier()
         self.allImagesList = []

         self.imageTrainList = []
         self.imageEvalList = []

         self.annotationList = []
         self.predictionList = []

    def getAllImagesList(self):
        return self.allImagesList

    def getImageTrainList(self):
        return self.imageTrainList

    def getImageEvalList(self):
        return self.imageEvalList

    # loads images we would like to evaluate on
    def loadImagesToPredict(self, imagesToAdd):
        self.predictionList = self.loadImages(imagesToAdd)

    def loadSplitImages(self, trainNames, evalNames):
        allImageList = []
        allImageAnnotationList = []

        imageTrainList = []
        imageTrainAnnotationList = []

        imageEvalList = []
        imageEvalAnnotationList = []

        for imagePath in trainNames:
            assert(os.path.exists("old_data" + os.sep + "trainImages" + os.sep + "200m" + os.sep + imageName))
            # loads in the image as a numpy array

            currImage = tif.imread("old_data" + os.sep + "trainImages" + os.sep + "200m" + os.sep + imageName)
            currImage_array = np.array(currImage)

            imageWidth, imageHeight, imageChannels = currImage_array.shape

            # make empty newImage array
            newImage = np.zeros((imageWidth, imageHeight, 3))

            # add RGB bands to images
            newImage[:, :, 0] = currImage_array[:, :, 2]
            newImage[:, :, 1] = currImage_array[:, :, 1]
            newImage[:, :, 2] = currImage_array[:, :, 0]

            imageTrainList.append(newImage)
            allImageAnnotationList.append()
            allImageList.append(newImage)

        # TODO: rename eval stuff to predictionImages stuff
        for imagePath in evalNames:
            assert(os.path.exists("old_data" + os.sep + "trainImages" + os.sep + "200m" + os.sep + imageName))
            # loads in the image as a numpy array

            currImage = tif.imread("old_data" + os.sep + "trainImages" + os.sep + "200m" + os.sep + imageName)
            print('after tif.imread')
            currImage_array = np.array(currImage)

            imageWidth, imageHeight, imageChannels = currImage_array.shape

            # make empty newImage array
            newImage = np.zeros((imageWidth, imageHeight, 3))

            # add RGB bands to images
            newImage[:, :, 0] = currImage_array[:, :, 2]
            newImage[:, :, 1] = currImage_array[:, :, 1]
            newImage[:, :, 2] = currImage_array[:, :, 0]

            imageEvalList.append(newImage)
            allImageList.append(newImage)

        self.allImagesList = allImageList
        self.imageTrainList = imageTrainList
        self.imageEvalList = imageEvalList

    def loadTrainAnnotations(self, imagesToAdd, annotationsToAdd):
        for annotationPath in annotationsToAdd:
            assert(os.path.exists("old_data" + os.sep + "annotations" + os.sep + annotationPath))
            currAnnotation = np.load("old_data" + os.sep + "annotations" + os.sep + annotationPath)


    # train the model on your old_data
    def train(self):
        self.model.fit(self.imageList, self.annotationList)

    def predict(self):
        prediction = self.model.predict(self.predictionList)
        print(prediction)

