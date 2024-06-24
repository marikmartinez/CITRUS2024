import os
import tensorflow as tf
import autokeras as ak
import PIL
import numpy as np
import tifffile as tiff
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
         self.model = ak.ImageRegressor(output_dim=None, max_trials=3)
         self.imagelist = []
         self.annotationList = []

    def imagesList(self):
        return self.imagesList
    
    def loadSpecificTreeRGB(self, treeName):
        treefolder = utils.upperImagesPath + "imagery-" + treeName + os.sep
        treeImageData = []
        for imageFolder in os.listdir(treefolder):
            imageFolder = treefolder + imageFolder
            for tifImage in os.listdir(imageFolder):
                if tifImage[-4:] == "tiff":
                    currImage = tiff.imread(imageFolder + os.sep + tifImage)
                    currImageArray = np.array(currImage)
                    # make empty newImage array
                    newImage = np.zeros(currImageArray.shape)

                    # add RGB bands to images
                    newImage[:, :, 0] = currImageArray[:, :, 1]
                    newImage[:, :, 1] = currImageArray[:, :, 2]
                    newImage[:, :, 2] = currImageArray[:, :, 3]
                    print(tifImage)
                    currImageArray = np.array(currImage)
                    treeImageData.append(currImageArray)
        return np.array(treeImageData)
    
    def loadSplitImages(self):
        treeTypes = ['Abies_alba',
            'Abies_nordmanniana',
            'Castanea_sativa',
            'Fagus_sylvatica',
            'Larix_decidua',
            'Picea_abies',
            'Pinus_halepensis', 
            'Pinus_nigra', 
            'Pinus_nigra_laricio', 
            'Pinus_pinaster', 
            'Pinus_sylvestris', 
            'Pseudotsuga_menziesii', 
            'Quercus_ilex', 
            'Quercus_petraea', 
            'Quercus_pubescens',
            'Quercus_robur',
            'Quercus_robur',
            'Quercus_rubra',
            'Robinia_pseudoacacia']
        
        tempAnnotation = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        count = 0
        treeTypes = [treeTypes[0]]
        for treeName in treeTypes:
            # loads in the imagesas a numpy array
            currTreeList = self.loadSpecificTreeRGB(treeName)
            
            currAnnotation = tempAnnotation
            currAnnotation[count] = 1
            count += 1

            # add curr annotation to annotationslist currTreeList.size() times
            for n in range(len(currTreeList)):
                self.annotationList.append(currAnnotation)            
            self.imagelist.extend(currTreeList)


    # train the model on your old_data
    def train(self):
        annotationArray = np.array(self.annotationList)
        imageArray = np.array(self.imagelist)
        self.model.fit(imageArray, annotationArray, epochs=1)

    def predict(self):
        prediction = self.model.predict(self.predictionList)
        print(prediction)




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
