#!/usr/bin/env python
# coding: utf-8

# In[3]:

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

path_to_assets= os.path.join(os.path.dirname(__file__),'assets')
# In[4]:


def home_streamlit():
    
    st.title('PYnt of Blood')

    image_path = os.path.join(path_to_assets,r'homepage_image_bloodcell.png')
    image = Image.open(image_path)
    
    #image = Image.open('Image1.png')
    
    st.image(image, caption='Blood cells')
    
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
    
    image_path = os.path.join(path_to_assets,r'homepage_image_hemato.png')
    image_hemato = Image.open(image_path)
    
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
        st.markdown('- Plaquettes (PLA).')
    with col2:
        st.image(image_hemato, caption='Hématopoïèse',width=600)
    
    st.write("Les cellules du sang proviennent toute d'un même précurseur (*la cellule souche hématopoïétique*). \
             Au cours de l'hématopoïèse, elles vont se différencier.")
    st.write("Chaque cellule va alors présenter des particularités :")
    st.write("- Granulocytes : cytoplasme granuleux")
    st.write("- Neutrophiles : deux fois plus grand que les globules rouges, noyau sciendé en trois parties")
    st.write("- Eosinophiles : noyau scindé en deux parties")
    st.write("- Basophiles : trois fois plus grands que les globules rouges, noyau plus foncé")
    st.write("- Lymphocytes : taille similaire aux globules rouges")
    st.write("- Monocytes : noyau en forme de rein, quatre à cinq fois plus grand que les globules rouges.")
    
    
def dataset_streamlit():
    st.header('Datasets')
    st.write('Plusieurs datasets ont été considérés pour ce projet.')
    
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
            labellisées pour 5 classes sur les 8 utilisées à Barcelone.")
    
    st.write("Il a été décidé d’utiliser la totalité des images en tant que test data. Pour pallier l’absence d’images \
             sur les 3 classes absentes, quelques images (3 à 4) jugées caractéristiques (mais toujours non standardisées) \
                 ont été manuellement trouvées sur des sites de département d’histologie et ajoutées dans les répertoires \
                (Erythroblast (ERY), Immature Granulocytes (IG), et Platelet (PLA)) afin d’avoir un résultat pour tous les types de cellules.")
    
    st.subheader('Dataset Acute Promyelocytic Leukemia')
    st.write('Ce dataset contient un fichier au format .csv “master.csv” avec 5 colonnes. La première correspond \
             aux identifiants du patient, la seconde au diagnostic, la troisième à la cohorte, la quatrième à l’âge, \
                 et la dernière au sexe.\
                 La colonne diagnostic contient uniquement des diagnostics de cancer qualifié de APL pour Acute\
                Promyelocytic Leukemia ou AML pour Acute Myelocytic Leukemia. Ces cancers touchent des\
                cellules pro-myéloïdes ou myéloïdes. En plus de ce fichier .csv, il est également fourni une banque\
                de 25915 images réparties en 23 catégories')
    
    st.subheader('Dataset Leukenia Milan')
    st.write('Ce dataset rassemble des images de cancers lymphoblastiques. Ce type de cancer est diagnostiqué\
            par la présence trop importante de lymphocytes. Il est attendu que ce dataset ne comprenne que\
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
                             'test_accuracy':[0.15, '-','-',0.052]})
    
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

    
    list_algo = ['KNN df70 (normalized)',
                                           'KNN df80 (normalized)','KNN df90 (normalized)', 'RL df70 cv3','RL df80 cv3',
                                           'RL df90 cv3','VotingClassifier',
                                           'RandomForest df70', 'RandomForest df80','RandomForestdf90' ]

    colormap=np.array(['black', 'red', 'darkred', 'green', 'yellowgreen', 'mediumseagreen',
         'orange', 'blue', 'darkblue', 'dodgerblue', 'yellow', 'indigo', 'purple'])
    
    if option == 'Machine Learning':
        st.write('*Les modèles de machine learning ont été entraînés sur des images réduites exclusivement*')
        
        fig, ax = plt.subplots(figsize=(10,5))
        ax.scatter(range(1,11,1), df_machineL['Best test accuracy'],c=colormap[np.arange(0,10,1)])
        ax.boxplot([np.arange(0.47,0.97,0.01), np.arange(0.20,0.86,0.01), np.arange(0.01,0.24,0.01),
             np.arange(0.45,0.97,0.01),np.arange(0.28,0.98,0.01), np.arange(0.35,0.98,0.01),
             np.arange(0.53,0.98,0.01),
             np.arange(0.55,0.97,0.01),np.arange(0.48,0.97,0.01),np.arange(0.23,0.96,0.01)],
           showfliers=False, whis=0)
        plt.xticks([1,2,3,4,5,6,7,8,9,10], list_algo,rotation=90)
        plt.title('Score (scatterplot) and f1_score (boxplot) of each machine learning model')
        st.pyplot(fig)
        
        st.dataframe(df_machineL, width=1000)
    
    if option == 'Deep Learning':
        option_type = st.radio('Type de données :',('Images brutes', 'Images réduites'),horizontal=False,key=31)
        if option_type == 'Images brutes' :
            fig, ax = plt.subplots(figsize=(7,6))
            ax.scatter(range(1,5,1), df_deepL['train_accuracy'], label='Train accuracy')
            ax.scatter(range(1,5,1),df_deepL['val_accuracy'],label='Validation accuracy')
            ax.scatter(range(1,5,1),df_deepL['test_accuracy'],label='Test accuracy')
            plt.xticks([1,2,3,4], df_deepL['Name model'],rotation=90)
            plt.legend(loc='center right', fontsize=8)
            plt.xlabel('Deep Learning model')
            plt.ylabel('Accuracy')
            plt.title('Train, Validation and Test accuracy of each deep learning model')
        
            st.pyplot(fig)
                
            st.dataframe(df_deepL, width=900)
            
        if option_type == 'Images réduites':

            fig, ax = plt.subplots(figsize=(4,3))
            ax.scatter(range(1,3,1), df_deepL_red['train_accuracy'], label='Train accuracy')
            ax.scatter(range(1,3,1),df_deepL_red['val_accuracy'],label='Validation accuracy')
            ax.scatter(range(1,3,1),df_deepL_red['test_accuracy'],label='Test accuracy')
            plt.xticks([1,2], df_deepL_red['Name model'])
            plt.legend(loc='center', fontsize=8)
            plt.xlabel('Deep Learning model')
            plt.ylabel('Accuracy')
            plt.title('Train, Validation and Test accuracy of deep learning models')
            st.pyplot(fig)

            st.dataframe(df_deepL_red)
        
    if option =='Transfer Learning':
        option_type = st.radio('Type de données :',('Images brutes', 'Images réduites'),horizontal=False,key=32 )
        if option_type == 'Images brutes':
            fig, ax = plt.subplots(figsize=(5,4))
            ax.scatter(range(1,5,1), df_transferL['train_accuracy'], label='Train accuracy')
            ax.scatter(range(1,5,1),df_transferL['val_accuracy'],label='Validation accuracy',alpha=0.7)
            ax.scatter([1,4], [0.15, 0.052], label='Test accuracy')
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
            ax.scatter(2,0.52,label='Test accuracy')
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
    with col1 :
        st.write("Tout d'abord, il est important d'augmenter la généralisation de nos modèles. Pour ce faire, il est possible \
                 d'augmenter le nombre de nos données en combinant plusieurs datasets de diverses origines. Il est également \
                possible de créer de la diversité en appliquant un ImageDataGenerator de la librairie Keras. Cette classe permet \
                d'appliquer des transformations telles que la rotation, le décalage, le zoom, le retournement horizontal, etc., \
                ce qui aide à améliorer la généralisation du modèle. ImageDataGenerator génère également des lots d'images prêtes \
                à être utilisées pour l'entraînement du modèle. Vous pouvez spécifier la taille du lot, le nombre total d'images \
                à générer et d'autres paramètres de configuration.")
    with col2:
        image_path = os.path.join(path_to_assets,r'homepage_image_generator.png')
        image_generator = Image.open(image_path)
        #image_generator = Image.open('homepage_image_generator.png')
        st.image(image_generator)
        
    st.write("Il serait également très pertinent de tester la mise en place d'un modèle associant le deep ou le transfer \
             learning avec le machine learning. Cette association peut présenter plusieurs avantages : ")
    st.write("- Capacité à extraire des caractéristiques pertinentes : Les modèles de deep learning et de transfer learning sont \
             généralement très efficaces pour extraire automatiquement des caractéristiques discriminantes à partir des données \
            brutes, notamment des images, du texte ou des signaux.")
    st.write("- Réduction de la dimensionnalité : Les modèles de deep learning peuvent être utilisés pour réduire la \
             dimensionnalité des données en projetant les exemples d'entrée dans un espace de caractéristiques de dimension \
            inférieure. Cela peut être bénéfique lorsque les données présentent un grand nombre de dimensions, ce qui peut \
            rendre la tâche de classification plus difficile pour les modèles de machine learning classiques. La réduction de \
            dimensionnalité permet de réduire la complexité du problème et d'améliorer l'efficacité du modèle de machine \
            learning ultérieur.")
    st.write("Une fois que les caractéristiques pertinentes ont été extraites à l'aide du modèle de deep learning ou de transfer \
             learning, un modèle de machine learning tel que le SVM peut être utilisé pour effectuer la classification ou la \
            régression. Le SVM est un modèle bien établi et robuste. Il peut bénéficier des caractéristiques de haut niveau \
            extraites par le modèle de deep learning et fournir une classification précise. En combinant le pouvoir de \
            l'extraction de caractéristiques des modèles de deep learning avec les capacités de classification des modèles \
            de machine learning traditionnels, on peut obtenir un système plus puissant et performant pour la tâche de \
            classification ou de régression. Cependant, il est important de noter que cette approche peut nécessiter plus de \
            temps de calcul et de ressources en raison de la complexité des modèles de deep learning. ")

    
    
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
    
    st.write('Dataset Barcelona : *A dataset of microscopic peripheral blood cell images for development of automatic \
             recognition systems*. Andrea Acevedo, Anna Merino, Santiago Alféred, Ángel Molina, Laura Boldú, José Rodellar')
             
    st.write('Datset Raabin : *A large dataset of white blood cells containing cell locations and types, along with segmented \
             nuclei and cytoplasm*.Zahra Mousavi Kouzehkanan, Sepehr Saghari, Sajad Tavakoli, Peyman Rostami, Mohammadjavad \
            Abaszadeh, Farzaneh Mirzadeh, Esmaeil Shahabi Satlsar, Maryam Gheidishahran, Fatemeh Gorgi, Saeed Mohammadi & \
            Reshad Hosseini.')
# In[ ]:




