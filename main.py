import utils
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

if len(sys.argv) < 2:
    print("usage: ", sys.argv[0], " ml/graph")
    exit(2)
if sys.argv[1] == "ml":
    machine_learn = True
    makeGraphs = False
elif sys.argv[1] == "graph":
    makeGraphs = True
    machine_learn = False
else:
    print("usage: ", sys.argv[0], " ml/graph")
    exit(3)



if __name__ == '__main__':
    bandNames = ["Blue(0)", "Green(1)", "Red(2)", "Near infared(3)", "Red edge(4)", "Band 6(5)", "Band 7(6)", "Band 8A(7)", "SWIR1(8)", "SWIR2(9)", "Aerosol(10)", "Band 9(11)"]
    
    treeTypes = ["Abies_alba", "Acer_pseudoplatanus", "Alnus_spec", "Betula_spec", "Cleared", "Fagus_sylvatica", "Fraxinus_excelsior", "Larix_decidua", "Larix_kaempferi", "Picea_abies", "Pinus_nigra", "Pinus_strobus", "Pinus_sylvestris", "Populus_spec", "Prunus_spec", "Pseudotsuga_menziesii", "Quercus_petraea", "Quercus_robur", "Quercus_rubra", "Tilia_spec"]
    #SENTINEL
    #band3 = green
    #band4 = red
    #band5=red edge
    #band8 =NIR
    
    #treeTypes = treeTypes[:3]
    #LOAD MONOSPECIFIC FOREST IMAGES (>= 90%)
    #utils.createMsAnnotations() 
    
    #Loads old ms names
    #msNames = utils.loadTxt("all_ms_names.txt")
    #print(msNames)
    
    #print("num of images:" + str(len(msNames)))
    
    #loads cleaned data
    #allImagesData, cleanedMsNames = utils.loadAllImages(msNames)
    
    #splitting data
    utils.splitData()

    #loading cleaned data from npz
    
    dataPath = "data" + os.sep + "all_image_data.npz"
    loaded = np.load(dataPath)
    msNames = loaded['paths']
    allImagesData = loaded['data']
    print("ms names:", len(msNames))

    
    #TODO: save allImagesData as numpy file thing here
    #print(allImagesData)
    #print(allImagesData.shape)
    #print(allImagesData[0].shape)
    if machine_learn:
        import classifierAI
        akClassifier = classifierAI.autokerasClassifierAI()
        akClassifier.getRGBImages(allImagesData)
        akClassifier.getAnnotations(msNames)
        akClassifier.train()
        akClassifier.save()
    elif makeGraphs:
        #make graphs
        
        #utils.makeAvgNdviGraph(treeTypes, msNames, allImagesData, True)
        #utils.makeAvgNdviGraph(treeTypes, msNames, allImagesData, False)
    
        #utils.makeAvgEviGraph(treeTypes, msNames, allImagesData)
        #utils.makeAvgNdwiGraph(treeTypes, msNames, allImagesData)
        #utils.makeAvgRendviGraph(treeTypes, msNames, allImagesData)
        utils.makeAvgReflectanceGraph(treeTypes, msNames, allImagesData)




