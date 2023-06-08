#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
# In[4]:
path_to_assets= os.path.join(os.path.dirname(__file__),'assets')

def home_streamlit():
    
    st.title('PYnt of Blood')
    
    image_path = os.path.join(path_to_assets,r'homepage_image_bloodcell.png')
    image_1 = Image.open(image_path)
    #image = Image.open('Image1.png')
    st.image(image_1, caption='Blood cells')
    st.header('Objectif du projet')
    
    st.markdown("Le diagnostic d’un cancer nécessite plusieurs examens dont l’un des plus importants est l’examen histologique. \
                Un prélèvement ou une biopsie est effectué et une lame est observée par microscopie, ce qui permet de confirmer \
                ou non la présence d’anomalie, identifier les cellules saines ou touchées et poser définitivement le diagnostic.")
    
    st.markdown("Dans le cas de l’hématologie, la numération et formule sanguine doti être réalisé. Cependant, avant de savoir \
                si une cellule est saine ou cancéreuse, il est important de savoir de quel type de cellule il s'agit.")
    
    
    
    st.markdown('**Ce projet vise donc à développer un modèle de computer vision afin d’identifier les 8 différents types de \
                cellules du sang.**')
    
    st.subheader('Les différents types cellulaires :')
    st.markdown('Au total, 8 types de cellule sanguine ont été explorés :')
    
    image_path_2 = os.path.join(path_to_assets,r'homepage_image_hemato.png')
    image_hemato = Image.open(image_path_2)
    #image_hemato = Image.open('Image2.png')
   
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('- Basophiles (BA)')
        st.markdown('- Eosinophiles (EO)')
        st.markdown('- Erythroblastes (ER)')
        st.markdown('- Granulocytes immatures (IG)')
        st.markdown('- Lymphocytes (LYM)')
        st.markdown('- Monocytes (MON)')
        st.markdown('- Neutrophiles (NEU ou SNE)')
        st.markdown('- Plaquettes (PLA)')
    with col2:
        st.image(image_hemato, caption='Hématopoïèse',width=600)
    
    st.write("Les cellules du sang proviennent toute d'un même précurseur (*la cellule souche hématopoïétique*). \
             Au cours de **l'hématopoïèse**, elles vont se différencier.")
    st.write("Chaque cellule va alors présenter des particularités morphologique notamment")
    st.write("- Granulocytes : cytoplasme granuleux")
    st.write("- Neutrophiles : deux fois plus grand que les globules rouges, noyau sciendé en trois parties ;")
    st.write("- Eosinophiles : noyau scindé en deux parties")
    st.write("- Basophiles : trois fois plus grands que les globules rouges, noyau plus foncé")
    st.write("- Lymphocytes : taille similaire aux globules rouges")
    st.write("- Monocytes : noyau en forme de rein, quatre à cinq fois plus grand que les globules rouges.")
    
    
def dataset_streamlit():
    st.header('Datasets')
    st.write('Plusieurs datasets ont été considérés pour ce projet:')
    
    st.subheader("Dataset 'Barcelona Mendeley Data'" )
    st.write("Ce [dataset](https://www.sciencedirect.com/science/article/pii/S2352340920303681) provient d’un article \
             scientifique '*A dataset for microscopic peripheral blood cell images for development of automatic recognition \
            system*'. Les images sont réparties dans 8 dossiers différents, correspondant aux catégories de cellules suivantes: \
            basophiles, éosinophiles, érythroblastes, granulocytes immatures, lymphocytes, monocytes, plaquettes et neutrophiles.\
            Elles sont au format .jpg et de taille 363x360 ou 369x366 en RGB.")
    st.write("Toutes les images sont issues de patients sains (absence de maladies, infections, allergies)")
    
    
    st.subheader("Dataset 'Raabin'")
    st.write("Une [étude](https://www.nature.com/articles/s41598-021-04426-x) réalisée à Téhéran en 2021 se focalisant sur les \
             leucocytes offre l’avantage de mettre à disposition une quantité significative d’images plus diversifiées et déjà \
            labelisées pour 5 classes sur les 8 utilisées à Barcelone.")
    
    st.write("Il a été décidé d’utiliser la totalité des images en tant que test data. Pour pallier à l’absence d’images \
             sur les 3 classes absentes, quelques images (3 à 4) jugées caractéristiques (mais toujours non standardisées) \
                 ont été manuellement trouvées sur des sites de département d’histologie et ajoutées dans les répertoires \
                (Erythroblast (ER), Immature Granulocytes (IG), et Platelet (PLA)) afin d’avoir un résultat pour tous les types de cellules.")
    
    st.subheader('Dataset Acute Promyelocytic Leukemia')
    st.write('Ce dataset contient un fichier au format .csv “master.csv” avec 5 colonnes. La première correspond \
             aux identifiants du patient, la seconde au diagnostic, la troisième à la cohorte, la quatrième à l’âge, \
                 et la dernière au sexe.\
                 La colonne diagnostic contient uniquement des diagnostics de cancer qualifié de APL pour Acute\
                Promyelocytic Leukemia ou AML pour Acute Myelocytic Leukemia. Ces cancers touchent des\
                cellules pro-myéloïdes ou myéloïdes. En plus de ce fichier .csv, il est également fourni une banque\
                de 25915 images réparties en 23 catégories')
                
    st.subheader('Dataset Leukemia Milan')
    st.write('Ce dataset rassemble des images de cancers lymphoblastiques caractérisé la présence trop importante de lymphocytes. Il est attendu que ce dataset ne comprenne que\
                 des images de lymphocytes')
    st.write('Sont disponibles 108 images au format .jpg, associées chacune à un fichier texte au format .xyc\
             reportant les coordonnées des barycentres des blastes. Ces images sont notées XXX_1.jpg ou\
            XXX_0.jpg pour qualifier sur l’image si un patient est malade (1) ou non (0).\
            Sont également disponibles 260 images au format .tif qui représentent en leur centre un lymphocyte\
            cancéreux ou non. Comme pour les images au format .jpg, les patients sains ou malades sont\
            indiqués par respectivement 0 ou 1 à la fin du nom du fichier.\
            Ce dataset pourrait permettre d’entraîner le modèle à la détection des lymphocytes sains ou\
            cancéreux.')

def conclusion_streamlit():
    st.header('Conclusion')
    
    st.subheader('Résumé des performances des différents modèles entraînés')
    
    st.write('Au cours de ce projet, plusieurs modèles de machine learning, deep learning et transfer learning ont été mis en \
             place, et sur différents types de données.')
    

    st.write('Ci-dessous sont répertoriés les différentes performances de chaque modèle selon le type de données utilisées.')
    
    df_machineL = pd.DataFrame({'Name model': ['KNN df70 (normalized)',
                                           'KNN df80 (normalized)','KNN df90 (normalized)', 'RL df70 cv3','RL df80 cv3',
                                           'RL df90 cv3','VotingClassifier',
                                           'RandomForest df70', 'RandomForest df80','RandomForestdf90' ],
                           'Best test accuracy':[0.75402,0.66189,0.29599,0.7,0.7,0.69,
                                                 0.75, 0.76,0.75,0.68],
                           'F1_score interval':['0.47-0.97', '0.20-0.86', '0.01-0.24','0.45-0.97','0.28,0.98',
                                                '0.35,0.98','0.53-0.98','0.55-0.97',
                                               '0.48-0.97','0.23-0.96'],
                           'Best cell predictions': ['Platelet', 'Platelet', 'Platelet','Platelet', 'Platelet', 
                                                'Platelet','Platelet', 'Platelet', 
                                                'Platelet','Platelet'],
                           'Worst cell predictions':['Basophil', 'Basophil', 'All except platelet','Monocyte',
                                                     'Monocyte','Monocyte','Basophil','Monocyte, Basophil',
  
                                                     'Monocyte, Basophil','Basophil',]})
    
    df_deepL = pd.DataFrame({'Name model': ['CNN-1', 'variant CNN-1', 'CNN LeNet', 'CNN FAV'],
                   'number of epoch':[10,20,5,10],
                  'train_accuracy': [0.932,0.99,0.939,0.81], 'val_accuracy':[0.93,0.93,0.71,0.76],
                  'test_accuracy':[0.09, 0.16,0.07,0.07], 'f1_score test':['0-0.14','0-0.14','0-0.38','0-0.18'],
                  'Best cell test prediction':['Eosinophil','Eosinophil','IG','Eosinophil'],
                  'Worst cell test prediction':['All others','Platelet','All others','Basophil, Monocyte, Neutrophil, Platelet']})
    
    df_transferL = pd.DataFrame({'Name model': ['EfficientNet', 'VGG16', 'DenseNet123', 'Inception V3'],
                            'number of epoch':[5,12,5,5],
                             'train_accuracy': [0.925,0.8786,0.9364,0.5], 
                             'val_accuracy':[0.93,0.8765,0.9316,0.51],
                             'test_accuracy':[0.15, 0.17,'-',0.052]})
    
    df_deepL_red = pd.DataFrame({'Name model': ['CNN LeNet', 'CNN FAV'],
                   'number of epoch':[10,10],
                  'train_accuracy': [0.84,0.867], 'val_accuracy':[0.80,0.859],
                  'test_accuracy':[0.43, 0.21], 'f1_score test':['0-0.61','0-0.49'],
                  'Best cell test prediction':['Neutrophil','Monocyte, Eosinophil'],
                  'Worst cell test prediction':['Erythroblast, IG','Basophil, Lymphocyte']})
    
    df_transferL_red = pd.DataFrame({'Name model': ['VGG16','VGG16_red'],
                            'number of epoch':[5,10],
                             'train_accuracy': [0.8786, 0.83], 
                             'val_accuracy':[0.8765,0.80],
                             'test_accuracy':[ 0 ,0.52]})

    
    liste_modele = ['','Machine Learning', 'Deep Learning', 'Transfer Learning']
    option = st.selectbox("Choisir une catégorie d'algorithme", liste_modele)

    
    list_algo_ML = ['KNN df70 (normalized)',
                                           'KNN df80 (normalized)','KNN df90 (normalized)', 'RL df70 cv3','RL df80 cv3',
                                           'RL df90 cv3','VotingClassifier',
                                           'RandomForest df70', 'RandomForest df80','RandomForestdf90' ]
    list_algo_f1_ML=[1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,
             7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10]

    colormap_ML=np.array(['black', 'red', 'darkred', 'green', 'yellowgreen', 'mediumseagreen',
         'orange', 'blue', 'darkblue', 'dodgerblue', 'yellow', 'indigo', 'purple'])
    
    f1_ML=[0.47,0.76,0.74,0.60,0.74,0.55,0.87,0.97,0.20,0.59,0.58,0.54,0.71,0.50,0.55,0.86,0.01,0.00,0.01,0.03,0.05,0.01,0.00,0.24,
   0.57,0.72,0.69,0.57,0.72,0.37,0.79,0.98,0.55,0.72,0.70,0.57,0.75,0.35,0.78,0.98,0.49,0.71,0.69,0.55,0.72,0.35,0.78,0.98,
   0.59,0.79,0.77,0.63,0.75,0.57,0.87,0.99,0.57,0.79,0.75,0.64,0.76,0.55,0.84,0.97,0.51,0.77,0.74,0.64,0.76,0.48,0.84,0.97,
   0.23,0.74,0.65,0.56,0.71,0.31,0.80,0.96]
    
    f1_DLbrutes=[0.02,0.14,0.00,0.02,0.05,0.00,0.01,0.00,0.00,0.00,0.00,0.38,0.00,0.00,0.00,0.00,0.00,0.14,0.00,0.00,0.06,0.00,0.00,0.00,
0.00,0.13,0.01,0.18,0.09,0.00,0.00,0.00]

    list_algo_f1_DLbrutes=[1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4]
    
    
    if option == 'Machine Learning':
        st.write('*Les modèles de machines learning ont été entraînés sur des images réduites exclusivement*')
        
        fig, ax = plt.subplots(figsize=(10,5))
        ax.scatter(range(1,11,1), df_machineL['Best test accuracy'],c=colormap_ML[np.arange(1,11,1)], label='Train accuracy',s=20)
        ax.scatter(x=list_algo_f1_ML, y=f1_ML,c=colormap_ML[list_algo_f1_ML], alpha=0.5,s=15, marker='*', label='f1_score')
        plt.xticks([1,2,3,4,5,6,7,8,9,10], list_algo_ML,rotation=90)
        plt.title('Score and f1_score of each machine learning model')
        plt.xlabel('Models')
        plt.legend()
        st.pyplot(fig)
        
        st.dataframe(df_machineL, width=1000)
    
    if option == 'Deep Learning':
        option_type = st.radio('Type de données :',('Images brutes', 'Images réduites'),horizontal=False )
        if option_type == 'Images brutes' :
            
            fig, ax = plt.subplots(figsize=(7,6))
            ax.scatter(range(1,5,1), df_deepL['train_accuracy'], label='Train accuracy',s=20)
            ax.scatter(range(1,5,1),df_deepL['val_accuracy'],label='Validation accuracy',s=20)
            ax.scatter(range(1,5,1),df_deepL['test_accuracy'],label='Test accuracy',s=20)
            ax.scatter(x=list_algo_f1_DLbrutes, y=f1_DLbrutes,c='green', alpha=0.5,s=15, marker='*', label='f1_score Test')
            plt.xticks([1,2,3,4], df_deepL['Name model'],rotation=90)
            plt.legend(loc='center right', fontsize=8)
            plt.xlabel('Deep Learning model')
            plt.ylabel('Accuracy')
            plt.title('Train, Validation and Test accuracy of each deep learning model')
        
            st.pyplot(fig)
                
            st.dataframe(df_deepL, width=900)
            
        if option_type == 'Images réduites':

            fig, ax = plt.subplots(figsize=(4,4))
            ax.scatter(range(1,3,1), df_deepL_red['train_accuracy'], label='Train accuracy',s=20)
            ax.scatter(range(1,3,1),df_deepL_red['val_accuracy'],label='Validation accuracy',s=20)
            ax.scatter(range(1,3,1),df_deepL_red['test_accuracy'],label='Test accuracy',s=20)
            ax.scatter([1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2], [0.11,0.00,0.42,0.00,0.28,0.46,0.61,0.18,0.00,0.00,0.21,0.00,0.00,0.49,0.31,0.44],
          c='green', alpha=0.5,s=15, marker='*', label='f1_score Test')
            plt.xticks([1,2], df_deepL_red['Name model'])
            plt.legend(loc='center', fontsize=8)
            plt.xlabel('Deep Learning model')
            plt.ylabel('Accuracy')
            plt.title('Train, Validation and Test accuracy of deep learning models')
            st.pyplot(fig)
            

            st.dataframe(df_deepL_red)
        
    if option =='Transfer Learning':
        option_type = st.radio('Type de données :',('Images brutes', 'Images réduites'),horizontal=False )
        if option_type == 'Images brutes':
            fig, ax = plt.subplots(figsize=(5,4))
            ax.scatter(range(1,5,1), df_transferL['train_accuracy'], label='Train accuracy')
            ax.scatter(range(1,5,1),df_transferL['val_accuracy'],label='Validation accuracy',alpha=0.7)
            ax.scatter([1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4], 
                       y=[0.02,0.02,0.00,0.02,0.01,0.01,0.00,0.00,0.00,0.03,0.00,0.04,0.08,0.03,0.00,0.00,0.00,0.08,0.01,
                          0.17,0.10,0.00,0.00,0.00],
                       c='green', alpha=0.5,s=15, marker='*',  label='f1_score')
            ax.scatter([1,2,4], [0.15,0.17, 0.052], label='Test accuracy')
            plt.xticks([1,2,3,4], df_transferL['Name model'],rotation=90)
            plt.legend(fontsize=8)
            plt.xlabel('Transfer Learning model')
            plt.ylabel('Accuracy')
            plt.title('Train, Validation and Test accuracy of each transfer learning model')
        
            st.pyplot(fig)
        
            st.dataframe(df_transferL, width=1000)
            
        if option_type == 'Images réduites' :
            fig, ax = plt.subplots(figsize=(4,3))
            ax.scatter(range(1,3,1), df_transferL_red['train_accuracy'], label='Train accuracy')
            ax.scatter(range(1,3,1),df_transferL_red['val_accuracy'],label='Validation accuracy', alpha=0.6)
            ax.scatter([1,2],[0.17,0.52],label='Test accuracy')
            ax.scatter([1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2], 
                       y=[0.00,0.03,0.00,0.04,0.08,0.03,0.00,0.00,0.02,0.00,0.31,0.00,0.34,0.56,0.79,0.01],
                       c='green', alpha=0.5,s=15, marker='*', label='f1_score')
            plt.xticks([1,2], df_transferL_red['Name model'])
            plt.legend(loc='center', fontsize=8)
            plt.xlabel('Deep Learning model')
            plt.ylabel('Accuracy')
            plt.title('Train, Validation and Test accuracy of each deep learning model')
            
            st.pyplot(fig)
            
            st.dataframe(df_transferL_red, width=1000)
        
        
    st.subheader('Synthèse et perspectives')
    st.write("Les modèles de deep et transfer learning permettent de meilleures prédictions; Cependant, ces prédictions sont conditionnées à la qualité des données. Dans notre cas, le set de données d'entraînement contient des images dans lesquelles les cellules sont centrées, le fond et la luminosité sont uniformes, et toutes les images sont nettes ")
    
    st.write('Plusieurs points peuvent donc être améliorés ou complétés.')
    col1, col2 = st.columns(2)
    with col1:
        st.write("Tout d'abord, il est important d'augmenter la généralisation de nos modèles. Pour ce faire, il est possible \
                 d'augmenter le nombre de nos données en combinant plusieurs datasets de diverses origines. Il est également \
                possible de créer de la diversité en appliquant un ImageDataGenerator de la librairie Keras. Cette classe permet \
                d'appliquer des transformations telles que la rotation, le décalage, le zoom, le retournement horizontal, etc., \
                ce qui aide à améliorer la généralisation du modèle. ImageDataGenerator génère également des lots d'images prêtes \
                à être utilisées pour l'entraînement du modèle. Vous pouvez spécifier la taille du lot, le nombre total d'images \
                à générer et d'autres paramètres de configuration.")
    with col2:
        image_path_3 = os.path.join(path_to_assets,r'homepage_image_generator.png')
        image_generator_ = Image.open(image_path_3)
        #image_generator = Image.open('homepage_image_generator.png')
        st.image(image_generator_)##
        
    st.write('')
    
    col3,col4 = st.columns(2)
    with col3 :
        image_path_4 = os.path.join(path_to_assets,r'homepage_objectdetection.png')
        image_cell_detection = Image.open(image_path_4)
    #    image_cell_detection = Image.open()
        st.image(image_cell_detection, caption='Cell detection with the MobileNet Single Shot Detector Version 2')
    with col4:
        st.write("Il serait également très pertinent de mettre en place un modèle de détection d'objet. ")
        st.write("La détection d'objet présente de nombreux avantage comme l'aide à la prise de décision.\
                     Elle est utilisée dans des domaines tels que l'analyse de données, la reconnaissance de formes, la médecine, \
                    l'agriculture de précision, etc. Ces informations peuvent aider à identifier des modèles, à prévoir des comportements ou \
                    à diagnostiquer des problèmes.")
    st.write("Il existe de nombreux modèles comme le modèle MobileNet Single Shot Detector Version 2 qui combine l'algorithme MobileNet avec la technique de détection en une seule passe (single shot detection).")
    st.write("Un autre modèle populaire est l'algorithme YOLO. Cet algorithme est extrêmement rapide et peut détecter des objets en temps réel.\
                     Tout comme MobileNet Single Shot Detector Version 2, il est capable de détecter plusieurs objets simultanément. \
                    Il attribue des scores de confiance à chaque boîte englobante prédite, ce qui permet de distinguer les objets présents \
                    et d'ignorer les régions vides")
    

    
    
def about_streamlit():
    st.header('A propos')
    
    st.subheader('Contributeurs')
    
    st.text('Jessica Planade')
    st.text('Julia Canac')
    st.text('Richard Bonfils')
    st.text('Frédéric Navez')
    
    st.subheader('Références')
    st.write('Image *Blood cells* : Monocyte, lymphocyte and neutrophil surrounded by red blood cells, dreamstime (free)')
    
    st.write('Image *Hématopoïèse* : Wikipédia')
    
    st.write('Image *Cell Detection by MobileNet Single Shot Detector Version 2*: [repo github](https://github.com/josephofiowa/tensorflow-object-detection/tree/master/data)')
    
    st.write('Dataset Barcelona : *A dataset of microscopic peripheral blood cell images for development of automatic \
             recognition systems*. Andrea Acevedo, Anna Merino, Santiago Alféred, Ángel Molina, Laura Boldú, José Rodellar')
             
    st.write('Datset Raabin : *A large dataset of white blood cells containing cell locations and types, along with segmented \
             nuclei and cytoplasm*.Zahra Mousavi Kouzehkanan, Sepehr Saghari, Sajad Tavakoli, Peyman Rostami, Mohammadjavad \
            Abaszadeh, Farzaneh Mirzadeh, Esmaeil Shahabi Satlsar, Maryam Gheidishahran, Fatemeh Gorgi, Saeed Mohammadi & \
            Reshad Hosseini.')
    st.write('[Architecture du Random Forest](https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/)')
    st.write('[Architecture du VGG16](https://www.mdpi.com/1424-8220/23/2/570)')
    st.write('[Architecture de LeNet](https://d2l.ai/chapter_convolutional-neural-networks/lenet.html)')
    
    
# In[ ]:




