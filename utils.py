import random
import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import json


dataPath = "data" + os.sep
labelPath =  "labels" + os.sep + "TreeSatBA_v9_60m_multi_labels.json"
msLabelPath = dataPath + "allDataNames.txt"
#upperImagesPath = dataPath + "monospecific_forest_imagery" + os.sep
imagesPath = dataPath + "60m"

treeLabels = ["Abies alba", "Acer pseudoplatanus", "Alnus spec", "Betula spec", "Cleared", "Fagus sylvatica", "Fraxinus excelsior", "Larix decidua", "Larix kaempferi", "Picea abies", "Pinus nigra", "Pinus strobus", "Pinus sylvestris", "Populus spec", "Prunus spec", "Pseudotsuga menziesii", "Quercus petraea", "Quercus robur", "Quercus rubra", "Tilia spec"]
treePaths = ["Abies_alba", "Acer_pseudoplatanus", "Alnus_spec", "Betula_spec", "Cleared", "Fagus_sylvatica", "Fraxinus_excelsior", "Larix_decidua", "Larix_kaempferi", "Picea_abies", "Pinus_nigra", "Pinus_strobus", "Pinus_sylvestris", "Populus_spec", "Prunus_spec", "Pseudotsuga_menziesii", "Quercus_petraea", "Quercus_robur", "Quercus_rubra", "Tilia_spec"]

colorList =  [
    '#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628',
    '#984ea3', '#999999', '#e41a1c', '#ffff33', '#00b5ad',
    '#ff00ff', '#808000', '#00ffff', '#800000', '#000080',
    '#00ff00', '#40e0d0', '#ff7f50', '#ffd700', '#6a5acd'
]

color_mapping = dict(zip(treeLabels, colorList))

# splits the image names into two different files, one for predictions and one for training
# TODO: rename eval stuff to prediction stuff
#OLD SPLIT DATA
'''
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
'''


#OLD LOADALLIMAGES
'''
def loadAllImages():
    allImagesData = []
    for treefolder in oslistdir(upperImagesPath):
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
'''

def splitData():
    trainList = []
    evalList = []

    allDataPath = dataPath + "all_image_data.npz"
    loadedData = np.load(allDataPath)
    msNames = loadedData['paths']
    allImagesData = loadedData['data']

    dataDict = dict(zip(msNames, allImagesData))
    print("total data: ", len(msNames))
    for tree in treePaths:
        treePathList = []
        for imageName in msNames:
            if imageName[:len(tree)] == tree:
                treePathList.append(imageName)
        #output names into files
        print(tree, ": ", len(treePathList))
        filePath = dataPath + "monospecific_treetype_lists" + os.sep + tree + ".txt"
        with open(filePath, 'w') as file:
            for path in treePathList:
                file.write(path + '\n')
        
        #split 90:10 for each tree type
        random.shuffle(treePathList)
        splitIndex = int(len(treePathList)*0.9)
        trainList.extend(treePathList[:splitIndex])
        print("train #: ", len(treePathList[:splitIndex]))
        evalList.extend(treePathList[splitIndex:])
        print("eval #: ", len(treePathList[splitIndex:]))
        
    random.shuffle(trainList)
    random.shuffle(evalList)
    
    #loading into npz files
    print("eval data total: ", len(evalList))
    evalData = []
    for path in evalList:
        dataForPath = dataDict[path]
        print(dataForPath)
        evalData.append(dataForPath)
    
    np_evalList = np.array(evalList)
    np_evalData = np.array(evalData)
    evalDataPath = dataPath + "eval_data.npz"
    np.savez(evalDataPath, paths=np_evalList, data=np_evalData)
    
    print("train data total: ", len(trainList))
    trainData = []
    for path in trainList:
        dataForPath = dataDict[path]
        trainData.append(dataForPath)
    
    np_trainList = np.array(trainList)
    np_trainData = np.array(trainData)
    trainDataPath = dataPath + "train_data.npz"
    np.savez(trainDataPath, paths=np_trainList, data=np_trainData)
    

def calculateIndexes(imageData):
    allImageVegetationIndexes = []
    for imageData in dataArray:
        vegetationIndexes = np.zeros(5)
        vegetationIndexes[0] = calculateNdviMean(imageData)
        vegetationIndexes[1] = calculateEviMean(imageData)
        vegetationIndexes[2] = calculateNdwiMean(imageData)
        vegetationIndexes[3] = calculateRendviMean(imageData)
        
        allImageVegetationIndexes.append(vegetationIndexes)
    
    return allImageVegetationIndexes


def loadAllImages(msNames):
    allImagesData = []
    cleanedMsNames = []
    for fileName in msNames:
        currImage = tiff.imread(imagesPath + os.sep + fileName)
        np_currImage = np.array(currImage)
        np_currImage = np_currImage / 10000
        if np_currImage.mean() < 0.003:
            continue
        else:
            allImagesData.append(np_currImage)
            cleanedMsNames.append(fileName)
    np_allImagesData = np.array(allImagesData)
    
    np_cleanedMsNames = np.array(cleanedMsNames)
    #saving into an .npz file
    np_msNames = np.array(msNames)
    allImageDataPath = dataPath + "all_image_data.npz"
    np.savez(allImageDataPath, paths=np_cleanedMsNames, data=np_allImagesData)

    #save file names into a .npy file
    allCleanedMsNamesDataPath = dataPath + "all_cleaned_ms_names.npy"
    np.save(allCleanedMsNamesDataPath, np_cleanedMsNames)

    return np_allImagesData, cleanedMsNames

def calculateNdviMean(imageData, filter):
    ndviArray = []

    for currImage in imageData:
        #changed into np array
        np_imageNdvi = np.array((currImage[:,:,3] - currImage[:,:,2])/(currImage[:,:,3] + currImage[:,:,2]))
        np_imageNdvi = np_imageNdvi[~np.isnan(np_imageNdvi)]
        if len(np_imageNdvi) == 0:
            continue
        #print(np_imageNdvi)
        if filter:
            np_filtered_imageNdvi = np.array(np_imageNdvi[np_imageNdvi >= 0.3])
            if len(np_filtered_imageNdvi) > 0:
                ndviArray.append(np_filtered_imageNdvi.mean())
        else:
            ndviArray.append(np_imageNdvi.mean())

    np_avgImgNdvi = np.array(ndviArray).mean()
    
    return np_avgImgNdvi

def findImageBandAvg(np_allImageData):
    imageWidth, imageHeight, imageBands = np_allImageData[0].shape
    avgBandList = []

    for image in np_allImageData:
        bandList = np.zeros(imageBands)
        for index in range(imageBands):
            bandList[index] = np.mean(image[:,:,index])
        avgBandList.append(bandList)

    return np.array(avgBandList)

def createLineGraph(xvalues, yvalues, title, xlabel, ylabel):
    pass

def createBarGraph():
    pass

# takes 4 bands
def showImage(falseColor, imgData):
    imgWidth, imgHeight, imgBands = imgData.shape
    img = np.zeros(imgWidth, imgHeight, 3)
    if falseColor == True:
        img[:, :, 0] = imgData[:, :, 0]
        img[:, :, 1] = imgData[:, :, 2]
        img[:, :, 2] = imgData[:, :, 3]
    else:
        img[:, :, 0] = imgData[:, :, 1]
        img[:, :, 1] = imgData[:, :, 2]
        img[:, :, 2] = imgData[:, :, 3]
    plt.imshow(img)

def loadLabels(filePath):
    with open(dataPath + filePath, "r") as file:
        jsonData = json.load(file)

    return jsonData

def createMsAnnotations():
    allLabels = loadLabels(labelPath)
    #print(allLabels)
    msLabels = {}

    for fileName, treeSpecies in allLabels.items():
        if treeSpecies[0][1] >= .9:
            msLabels[fileName] = treeSpecies[0][1]
            #puts all file names into a txt file

    json_msLabels = json.dumps(msLabels, indent=4)
    with open(dataPath + "ms_annotations", "w") as outfile:
        outfile.write(json_msLabels)
    msFileNames = list(msLabels.keys())
    msFileNames.sort()
    with open(dataPath + "all_ms_names.txt", "w") as outfile:
        for fileName in msFileNames:
            outfile.write(fileName)
            outfile.write('\n')

def loadJson(filePath):
    with open(dataPath + filePath, "r") as file:
        msJson = json.load(file)

    return msJson

def loadTxt(filePath):
    msTxt = open(dataPath + filePath).read().splitlines()
    
    return msTxt

def makeAvgNdviGraph(treeTypes, msNames, allImagesData, filter):
    treeLabels = ["Abies alba", "Acer pseudoplatanus", "Alnus spec", "Betula spec", "Cleared", "Fagus sylvatica", "Fraxinus excelsior", "Larix decidua", "Larix kaempferi", "Picea abies", "Pinus nigra", "Pinus strobus", "Pinus sylvestris", "Populus spec", "Prunus spec", "Pseudotsuga menziesii", "Quercus petraea", "Quercus robur", "Quercus rubra", "Tilia spec"]

    ndviAveragesPerTreeList = []
    print(allImagesData.shape)
    for tree in treeTypes:
        index = 0
        currTreeImagesList = []
        #puts current tree type data into an array
        for imageName in msNames:
            if imageName[:len(tree)] == tree:
              currTreeImagesList.append(allImagesData[index])
            index += 1
        print(len(currTreeImagesList))
        ndviAveragesPerTreeList.append(calculateNdviMean(currTreeImagesList, filter))

    paired_lists = list(zip(ndviAveragesPerTreeList, treeLabels))

    # Sort the paired lists by the first element (elements of list1)
    sorted_paired_lists = sorted(paired_lists)

    # Unzip the sorted pairs back into two lists
    sorted_list1, sorted_list2 = zip(*sorted_paired_lists)

    # Convert them back to lists (optional, for better usability)
    ndviAveragesPerTreeList = list(sorted_list1)
    sortedTreeLabels = list(sorted_list2)    
    
    # creating the bar plot
    fig = plt.figure(figsize = (10, 15))
    colors = [color_mapping[label] for label in sortedTreeLabels]
    plt.bar(sortedTreeLabels,ndviAveragesPerTreeList, color=colors, width = 0.8)
    plt.xlabel("Species Classes")
    plt.xticks(rotation=45, ha='right')
    if filter:
        plt.ylabel("Mean NDVI")
        plt.title("Mean of Filtered NDVI for Species Classes")
        plt.savefig("graphs/filtered_ndvi_mean.png")
    else:
        plt.ylabel("Mean NDVI")
        plt.title("Mean NDVI for Species Classes")
        plt.savefig("graphs/ndvi_mean.png")

#TODO: make this into EVI, NDWI, RENDVI
def calculateEviMean(imageData):
    eviArray = []
    invalidImgCounter = 0
    for currImage in imageData:
        #changed into np array
        np_imageEvi = np.array(2.5 * ((currImage[:,:,3] - currImage[:,:,2])/(currImage[:,:,3] + 6 * currImage[:,:,2] -7.5 * currImage[:,:,0] + 1)))
        np_imageEvi = np_imageEvi[~np.isnan(np_imageEvi)]
        if len(np_imageEvi) == 0:
            print(currImage)
            invalidImgCounter += 1
            continue
        #print(np_imageEvi)
        eviArray.append(np_imageEvi.mean())

    np_avgImgEvi = np.array(eviArray).mean()
    
    return np_avgImgEvi

def makeAvgEviGraph(treeTypes, msNames, allImagesData):
    treeLabels = ["Abies alba", "Acer pseudoplatanus", "Alnus spec", "Betula spec", "Cleared", "Fagus sylvatica", "Fraxinus excelsior", "Larix decidua", "Larix kaempferi", "Picea abies", "Pinus nigra", "Pinus strobus", "Pinus sylvestris", "Populus spec", "Prunus spec", "Pseudotsuga menziesii", "Quercus petraea", "Quercus robur", "Quercus rubra", "Tilia spec"]

    eviAveragesPerTreeList = []
    print(allImagesData.shape)
    for tree in treeTypes:
        index = 0
        currTreeImagesList = []
        #puts current tree type data into an array
        for imageName in msNames:
            if imageName[:len(tree)] == tree:
              currTreeImagesList.append(allImagesData[index])
            index += 1
        print(len(currTreeImagesList))
        eviAveragesPerTreeList.append(calculateEviMean(currTreeImagesList))

    paired_lists = list(zip(eviAveragesPerTreeList, treeLabels))

    # Sort the paired lists by the first element (elements of list1)
    sorted_paired_lists = sorted(paired_lists)

    # Unzip the sorted pairs back into two lists
    sorted_list1, sorted_list2 = zip(*sorted_paired_lists)

    # Convert them back to lists (optional, for better usability)
    eviAveragesPerTreeList = list(sorted_list1)
    sortedTreeLabels = list(sorted_list2)    

    # creating the bar plot
    fig = plt.figure(figsize = (10, 15))
    colors = [color_mapping[label] for label in sortedTreeLabels]
    plt.bar(sortedTreeLabels,eviAveragesPerTreeList, color=colors, width = 0.8)
    plt.xlabel("Species Classes")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Mean EVI")
    plt.title("Mean EVI for Species Classes")
    plt.savefig("graphs/evi_mean.png")

def calculateNdwiMean(imageData):
    ndwiArray = []
    invalidImgCounter = 0
    for currImage in imageData:
        #imageArray = imageArray[np.nonzero(currImage[:,:,1] + imageArray[:,:,3])]    
        #changed into np array
        np_imageNdwi = np.array((currImage[:,:,3] - currImage[:,:,8])/(currImage[:,:,3] + currImage[:,:,8]))
        
        np_imageNdwi = np_imageNdwi[~np.isnan(np_imageNdwi)]
        if len(np_imageNdwi) == 0:
            print(currImage)
            invalidImgCounter += 1
            print("invalid img counter: " + str(invalidImgCounter))
            continue
        #print(np_imageNdwi)
        ndwiArray.append(np_imageNdwi.mean())

    np_avgImgNdwi = np.array(ndwiArray).mean()
    
    return np_avgImgNdwi

def makeAvgNdwiGraph(treeTypes, msNames, allImagesData):
    treeLabels = ["Abies alba", "Acer pseudoplatanus", "Alnus spec", "Betula spec", "Cleared", "Fagus sylvatica", "Fraxinus excelsior", "Larix decidua", "Larix kaempferi", "Picea abies", "Pinus nigra", "Pinus strobus", "Pinus sylvestris", "Populus spec", "Prunus spec", "Pseudotsuga menziesii", "Quercus petraea", "Quercus robur", "Quercus rubra", "Tilia spec"]

    ndwiAveragesPerTreeList = []
    print(allImagesData.shape)
    for tree in treeTypes:
        index = 0
        currTreeImagesList = []
        #puts current tree type data into an array
        for imageName in msNames:
            if imageName[:len(tree)] == tree:
              currTreeImagesList.append(allImagesData[index])
            index += 1
        print(len(currTreeImagesList))
        ndwiAveragesPerTreeList.append(calculateNdwiMean(currTreeImagesList))

    paired_lists = list(zip(ndwiAveragesPerTreeList, treeLabels))

    # Sort the paired lists by the first element (elements of list1)
    sorted_paired_lists = sorted(paired_lists)

    # Unzip the sorted pairs back into two lists
    sorted_list1, sorted_list2 = zip(*sorted_paired_lists)

    # Convert them back to lists (optional, for better usability)
    ndwiAveragesPerTreeList = list(sorted_list1)
    sortedTreeLabels = list(sorted_list2)    

    # creating the bar plot
    fig = plt.figure(figsize = (10, 15))
    colors = [color_mapping[label] for label in sortedTreeLabels]
    plt.bar(sortedTreeLabels,ndwiAveragesPerTreeList, color=colors, width = 0.8)
    plt.xlabel("Species Classes")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Mean NDWI")
    plt.title("Mean NDWI for Species Classes")
    plt.savefig("graphs/ndwi_mean.png")
    # TODO: turn into 

def calculateRendviMean(imageData):
    rendviArray = []
    invalidImgCounter = 0
    for currImage in imageData:
        #changed into np array
        np_imageRendvi = np.array((currImage[:,:,5] - currImage[:,:,2])/(currImage[:,:,5] + currImage[:,:,2]))
        np_imageRendvi = np_imageRendvi[~np.isnan(np_imageRendvi)]
        if len(np_imageRendvi) == 0:
            print(currImage)
            invalidImgCounter += 1
            print("invalid img counter: " + str(invalidImgCounter))
            continue
        #print(np_imageRendvi)
        rendviArray.append(np_imageRendvi.mean())

    np_avgImgRendvi = np.array(rendviArray).mean()
    
    return np_avgImgRendvi

def makeAvgRendviGraph(treeTypes, msNames, allImagesData):
    rendviAveragesPerTreeList = []
    print(allImagesData.shape)
    for tree in treeTypes:
        index = 0
        currTreeImagesList = []
        #puts current tree type data into an array
        for imageName in msNames:
            if imageName[:len(tree)] == tree:
              currTreeImagesList.append(allImagesData[index])
            index += 1
        print(len(currTreeImagesList))
        rendviAveragesPerTreeList.append(calculateRendviMean(currTreeImagesList))

    paired_lists = list(zip(rendviAveragesPerTreeList, treeLabels))

    # Sort the paired lists by the first element (elements of list1)
    sorted_paired_lists = sorted(paired_lists)

    # Unzip the sorted pairs back into two lists
    sorted_list1, sorted_list2 = zip(*sorted_paired_lists)

    # Convert them back to lists (optional, for better usability)
    rendviAveragesPerTreeList = list(sorted_list1)
    sortedTreeLabels = list(sorted_list2)    

    # creating the bar plot
    fig = plt.figure(figsize = (15, 15))
    colors = [color_mapping[label] for label in sortedTreeLabels]
    plt.bar(sortedTreeLabels,rendviAveragesPerTreeList, color=colors, 
        width = 0.8)
    plt.xlabel("Species Classes")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Mean RENDVI")
    plt.title("Mean RENDVI for Species Classes")
    plt.savefig("graphs/rendvi_mean.png")

#TODO: calculate reflectance
def calculateReflectanceMean(imageData):
    reflectanceList = []
    imageWidth, imageHeight, imageBands = imageData[0].shape

    for image in imageData:
        bandList = np.zeros(imageBands)
        for index in range(imageBands):
            bandList[index] = np.mean(image[:,:,index])
        reflectanceList.append(bandList)


    reflectanceListMeans = np.array(reflectanceList)
    return np.array(reflectanceListMeans)

def makeAvgReflectanceGraph(treeTypes, msNames, allImagesData):
    desiredBandOrder = [10, 0, 1, 2, 4, 5, 6, 3, 7, 11, 8, 9]
    reflectanceAvgPerTreeType = []
    for tree in treeTypes:
        index = 0
        currTreeImagesList = []
        #puts current tree type data into an array
        for imageName in msNames:
            if imageName[:len(tree)] == tree:
                currImage = allImagesData[index]
                sortedImage = currImage[:, :, desiredBandOrder]
                currTreeImagesList.append(sortedImage)
            index += 1
        reflectanceMean = calculateReflectanceMean(currTreeImagesList)
        bandAverages = np.zeros(12)
        for index in range(12):
            bandAverages[index] = reflectanceMean[:,index].mean()
        reflectanceAvgPerTreeType.append(bandAverages)
        #IMAGES PER TREE
        print(tree, " num images = ", len(currTreeImagesList))
    '''
    paired_lists = list(zip(reflectanceAvgPerTreeType, treeLabels))

    # Sort the paired lists by the first element (elements of list1)
    sorted_paired_lists = sorted(paired_lists)

    # Unzip the sorted pairs back into two lists
    sorted_list1, sorted_list2 = zip(*sorted_paired_lists)

    # Convert them back to lists (optional, for better usability)
    reflectanceAvgPerTreeType = list(sorted_list1)
    sortedTreeLabels = list(sorted_list2)    
    '''
    wavelengths = [443, 490, 560, 665, 705, 740, 783, 832, 865, 945, 1613, 2202]
    # creating the bar plot
    fig = plt.figure(figsize = (20, 13))
    #colors = [color_mapping[label] for label in treeLabels]
    index = 0
    for treeType in reflectanceAvgPerTreeType:
        plt.plot(wavelengths, treeType, label = treeLabels[index], color= colorList[index])
        index += 1
    plt.legend()
    #TODO: fix labels, make legend, make band in the back
    plt.xlabel("Wavelength [nm]")
    #plt.xticks(rotation=45, ha='right')
    plt.ylabel("Mean Reflectance [%]")
    plt.title("Mean Reflectance for Species Classes")
    plt.savefig("graphs/reflectance_mean.png")
    




   







