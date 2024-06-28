import os
import tensorflow as tf
import autokeras as ak
import PIL
import numpy as np
import json
import matplotlib.pyplot as plt
import utils

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
         self.model = ak.ImageClassifier(max_trials=20)
         self.imagelist = []
         self.annotationList = []

    def imagesList(self):
        return self.imagesList
    
    def getAnnotations(self, msNames):
        for tree in utils.treeTypes:
            for imageName in msNames:
                if imageName[:len(tree)] == tree:
                    self.annotationList.append(tree)

    
    def getRGBImages(self, allImagesData):
        for image in allImagesData:
            imageWidth, imageHeight, imageBands = image.shape
            rgbImage = np.zeros((6, 6, 3))
            rgbImage[:,:,0] = image[:,:,2]
            rgbImage[:,:,1] = image[:,:,1]
            rgbImage[:,:,2] = image[:,:,0]
            self.imagelist.append(rgbImage)
    
    # train the model on your old_data
    def train(self):
        with tf.device('/device:GPU:0'):
            annotationArray = np.array(self.annotationList)
            imageArray = np.array(self.imagelist)
            self.model.fit(imageArray, annotationArray, epochs=5)
            historyDict = self.model.history

        with open("sample.json", "w") as outfile: 
            json.dump(historyDict, outfile)

    def predict(self):
        prediction = self.model.predict(self.predictionList)
        print(prediction)

    def save(self):
        #self.model.save("ak_model", save_format="tf")
        pass


'''
            currImageArray = np.array(currImage)

            imageWidth, imageHeight, imageChannels = currImageArray.shape

            # make empty newImage array
            newImage = np.zeros((imageWidth, imageHeight, 3))

            # add RGB bands to images
            newImage[:, :, 0] = currImageArray[:, :, 1]
            newImage[:, :, 1] = currImageArray[:, :, 2]
            newImage[:, :, 2] = currImageArray[:, :, 3]

            imageTrainList.append(newImage)
            allImageAnnotationList.append()
            allImageList.append(newImage)

        # TODO: rename eval stuff to predictionImages stuff
        for imagePath in evalNames:
            assert(os.path.exists("old_data" + os.sep + "trainImages" + os.sep + "200m" + os.sep + imageName))
            # loads in the image as a numpy array

            currImage = tif.imread("old_data" + os.sep + "trainImages" + os.sep + "200m" + os.sep + imageName)
            print('after tif.imread')
            currImageArray = np.array(currImage)

            imageWidth, imageHeight, imageChannels = currImageArray.shape

            # make empty newImage array
            newImage = np.zeros((imageWidth, imageHeight, 3))

            # add RGB bands to images
            newImage[:, :, 0] = currImageArray[:, :, 2]
            newImage[:, :, 1] = currImageArray[:, :, 1]
            newImage[:, :, 2] = currImageArray[:, :, 0]

            imageEvalList.append(newImage)
            allImageList.append(newImage)

        self.allImagesList = allImageList
        self.imageTrainList = imageTrainList
        self.imageEvalList = imageEvalList
'''



# TODO:
# break out the training, test and validation data into 3 different directories
# data/monosomething/train/treetype
# data/monosomething/val/treetype
# data/monosomething/test/treetype
# 
# make it so we load 600 megabytes of images, figure out exactly how many images roughly that is by dividing by 200kb
# need to figure out where to modify loadspecifictreergb and loadsplitdata its gonna be somewhere in between the two maybe both
#
# actually start loading the training and validation data seperately
