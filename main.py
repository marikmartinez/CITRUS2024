import classifierAI
import utils
import matplotlib.pyplot as plt
import numpy as np
import os



machine_learn = True



if __name__ == '__main__':
    if machine_learn:
        akClassifier = classifierAI.autokerasClassifierAI()
        akClassifier.loadSplitImages()
        akClassifier.train()

        #print(allImageNames)
    else:
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
        '''
        for treeName in treeTypes:
            treeData = utils.loadSpecificTreeImages(treeName)
        '''    
        # LOADS ALL IMAGE DATA

        for tree in treeTypes:
            np_allImageData = utils.loadSpecificTreeImages(treeTypes[0])
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

            plt.title("Average Band Values Per image with " + tree)
            plt.xlabel("Multispectral Bands")
            plt.ylabel("Avg Value Per Image")
            plt.savefig('graphs' + os.sep + tree + 'AllBandAvg.png')

            #bar graph RGB

            #scaledRGB = utils.scaleRGB(y[:3])
            fig = plt.figure(figsize=(10, 5))
            plt.bar(x[:3], y[:3], color='blue')
            plt.title("Average RBG Band Values Per Image with " + tree)
            plt.xlabel("RBG Bands")
            plt.ylabel("Avg % reflectance")
            plt.savefig('graphs' + os.sep + tree + 'RBGBandAvg.png')

            #bar graph for trees
            #avg band values for each tree type

            #scatter plot
            #x is band
            #y is values






