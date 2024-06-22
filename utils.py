import random
import os
import numpy as np
import tifffile as tiff



dataPath = "data" + os.sep 
upperImagesPath = dataPath + "monospecific_forest_imagery" + os.sep





# splits the image names into two different files, one for predictions and one for training
# TODO: rename eval stuff to prediction stuff
def splitData(dataPath):
    allDataNames = open(dataPath).read().splitlines()
    random.shuffle(allDataNames)
    splitIndex = int(len(allDataNames)*0.9)
    trainNames = allDataNames[:splitIndex]
    evalNames = allDataNames[splitIndex:]

    with open(dataPath + "trainImages.txt", 'w') as outFile:
        for filename in trainNames:
            outFile.write(filename)
            outFile.write('\n')
    with open(dataPath + "evalImages.txt", 'w') as outFile:
        for filename in evalNames:
            outFile.write(filename)
            outFile.write('\n')


    return (allDataNames, trainNames, evalNames)


def loadAllImages():
    allImagesData = []
    for treefolder in os.listdir(upperImagesPath):
        print(treefolder)
        treefolder = upperImagesPath + treefolder + os.sep
        for imageFolder in os.listdir(treefolder):
            imageFolder = treefolder + imageFolder + os.sep
            for tifImage in os.listdir(imageFolder):
                #print("im name: ", tifImage)
                if tifImage[-4:] == "tiff":
                    currImage = tiff.imread(imageFolder + os.sep + tifImage)
                    currImageArray = np.array(currImage)
                    allImagesData.append(currImageArray)
    return np.array(allImagesData)


def loadSpecificTreeImages(treeName):
    treefolder = upperImagesPath + "imagery-" + treeName + os.sep
    treeImageData = []
    for imageFolder in os.listdir(treefolder):
        imageFolder = treefolder + imageFolder
        for tifImage in os.listdir(imageFolder):
            if tifImage[-4:] == "tiff":
                currImage = tiff.imread(imageFolder + os.sep + tifImage)
                print(tifImage)
                currImageArray = np.array(currImage)
                treeImageData.append(currImageArray)
    return np.array(treeImageData)


def findImageBandAvg(np_allImageData):
    imageWidth, imageHeight, imageBands = np_allImageData[0].shape
    avgBandList = []

    for image in np_allImageData:
        bandList = np.zeros(imageBands)
        for index in range(imageBands):
            bandList[index] = np.mean(image[:,:,index])
        avgBandList.append(bandList)

    return np.array(avgBandList)


