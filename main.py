#import classifierAI
import utils
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
    #akClassifier = classifierAI.autokerasClassifierAI()

    # allNames = open('old_data\\s2names.txt').read().splitlines()
    # trainNames = open('old_data\\trainImages.txt').read().splitlines()
    # evalNames = open('old_data\\evalImages.txt').read().splitlines()

    #allImageNames, trainImageNames, evalImageNames = utils.splitData("old_data\\s2names.txt")
    #allImageNames = open('old_data' + os.sep + 's2names.txt').read().splitlines()
    #print(len(allImageNames), len(trainImageNames), len(evalImageNames))
    # print(trainNames)
    #print(type(trainNames), type(evalNames))

    #akClassifier.loadSplitImages(trainNames, evalNames)

    #print(allImageNames)

    treeTypes = ['Abies_alba',
            'Abies_nordmannaniana',
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

    #load tree data
    treeData = utils.loadSpecificTreeImages(treeTypes[0])
    print('band 1:\n')
    # first array is images, array inside of that is the pixels, the array inside of the pixels is red green and blue and NIR
    print(treeData[0][:,:,0])
    print('band 2\n')
    print(treeData[0][:,:,1])
    print('band 3:\n')
    print(treeData[0][:,:,2])
    print('band 4:\n')
    print(treeData[0][:,:,3])
    
    # LOADS ALL IMAGE DATA
    # np_allImageData = utils.loadAllImages()

    #print(np_allImageData.shape)
    #print(np_allImageData[0][:,:,0])
    #print('max' + str(np_allImageData[:,:,0].max()) + 'min' + str(np_allImageData[:,:,0].min()))
    
    '''
    #print(type(np_allImageData))
    # FIND IMAGE BAND AVG
    np_ImgBandAvg = utils.findImageBandAvg(np_allImageData)
    #print(np_ImgBandAvg)

    plotKeys = {"blue": np.mean(np_ImgBandAvg[:, 0]), "green": np.mean(np_ImgBandAvg[:, 1]),
                "red": np.mean(np_ImgBandAvg[:, 2]), "NIR": np.mean(np_ImgBandAvg[:, 3])}

    x = list(plotKeys.keys())
    y = list(plotKeys.values())

    #bar graph ALL BANDS
    fig = plt.figure(figsize=(10, 5))
    plt.bar(x, y, width=0.4, color='brown')

    plt.title("Average Band Values Per Image")
    plt.xlabel("Multispectral Bands")
    plt.ylabel("Avg Value Per Image")
    plt.savefig('allBandAvg.png')

    #bar graph RGB

    #scaledRGB = utils.scaleRGB(y[:3])
    fig = plt.figure(figsize=(10, 5))
    plt.bar(x[:3], y[:3], color='blue')
    plt.title("Average RBG Band Values Per Image")
    plt.xlabel("RBG Bands")
    plt.ylabel("Avg % reflectance")
    plt.savefig('RBGBandAvg.png')

    #bar graph for trees
    #avg band values for each tree type

    #scatter plot
    #x is band
    #y is values
    '''






