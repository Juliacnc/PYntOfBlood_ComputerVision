# -*- coding: utf-8 -*-
"""
@author: Planade
"""

import streamlit as st
#import pandas as pd
import numpy as np
import joblib
#from sklearn.metrics import confusion_matrix as cm
#from sklearn.metrics import classification_report
#import itertools
#import matplotlib.pyplot as plt
#import seaborn as sns
import os
from PIL import Image


#variables to create path to assets folder from the tabs one
path_to_assets = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')



def predictions():
    
    st.title("Utilisation d'une sélection de modèles sur des images test")
    
    
    st.write('Nous allons présenter ici des résultats sur le dataset Raabin')
 
    st.write("Les modèles utilisés sont ceux présentés dans la partie ML.")  
    
    modele=st.selectbox("Choisir une catégorie de modèle",['','Random Forest','Convolutional Neural Network','Transfer Learning'],key=1)
    
 
# RANDOM FOREST
    
    if modele== 'Random Forest':
        
        st.subheader('Machine Learning : modèle Random forest sur données test réduites') 
        
        st.write("Les images du dataset Raabin sont préparées de la même manière que les images du dataset Barcelona Mendeley Data, afin d'obtenir un fichier réduit.")
        
        st.write("La matrice de confusion et le rapport de classification obtenus sont les suivants")


        image_path = os.path.join(path_to_assets,r'Prediction_RF_class_report_df70raabin.png')
        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(image, width=300) 
        
#        image = Image.open('Prediction_RF_class_report_df70raabin.png')
#        st.image(image, width=300) 
            
        image_path = os.path.join(path_to_assets,r'Prediction_RF_conf_mat_df70raabin.png')
        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(image, width=300) 
        
#        image = Image.open('Prediction_RF_conf_mat_df70raabin.png')
#        st.image(image, width=300)
        
        
        


# CNNs

    if modele== 'Convolutional Neural Network':
        
        st.subheader('Deep Learning : 2 modèles CNN sur données test') 


        red1 = st.checkbox('cocher pour les résultats sur données réduites',key=3)

        if red1:
            
                    
            st.write("Les images du dataset test sont réduites par un resizing au format 224x224 puis par la sélection des 20% de pixels les plus porteurs d'information.")
        
            st.write("Les matrices de confusion et les rapports de classification obtenus sont les suivants")
            
            st.write("Modèle simple architecture LeNet")
            
            image_path = os.path.join(path_to_assets,r'Prediction_LENET_conf_mat_raabin_red.png')
            if os.path.exists(image_path):
                image = Image.open(image_path)
                st.image(image, width=300)  
                
            image_path = os.path.join(path_to_assets,r'Prediction_LENET_class_report_raabin_red.png')
            if os.path.exists(image_path):
                image = Image.open(image_path)
                st.image(image, width=300)
                
            st.write("Modèle à fonctions d'activations variées")
            
            image_path = os.path.join(path_to_assets,r'Prediction_CNN_FAV_conf_mat_raabin_red.png')
            if os.path.exists(image_path):
                image = Image.open(image_path)
                st.image(image, width=300) 
                
            image_path = os.path.join(path_to_assets,r'Prediction_CNN_FAV_class_report_raabin_red.png')
            if os.path.exists(image_path):
                image = Image.open(image_path)
                st.image(image, width=300)


        else:
            
            st.write("Les images du dataset test sont simplement redimensionnées au format 360x360.")
        
            st.write("Les matrices de confusion et les rapports de classification obtenus sont les suivants")
            
            st.write("Modèle simple architecture LeNet")

            image_path = os.path.join(path_to_assets,r'Prediction_LENET_mat_report_raabin.png')
            if os.path.exists(image_path):
                image = Image.open(image_path)
                st.image(image, width=600)    

            st.write("Modèle à fonctions d'activations variées")
            
            image_path = os.path.join(path_to_assets,r'Prediction_CNN_FAV_mat_report_raabin.png')
            if os.path.exists(image_path):
                image = Image.open(image_path)
                st.image(image, width=600) 
     


#VGG16

    if modele== 'Transfer Learning':
        
        st.subheader('Deep Learning : modèle CNN boosté par TL, sur données test') 
        
      
        red2 = st.checkbox('cocher pour les résultats sur données réduites',key=4)

        if red2:
            
                        
            st.write("Les images du dataset test sont réduites par un resizing au format 224x224 puis par la sélection des 20% de pixels les plus porteurs d'information.")
            
            st.write("Les matrices de confusion et les rapports de classification obtenus sont les suivants")
            
            st.write("Modèle simple architecture LeNet")
            
            image_path = os.path.join(path_to_assets,r'Prediction_VGG16_conf_mat_raabin_red.png')
            if os.path.exists(image_path):
                image = Image.open(image_path)
                st.image(image, width=300)  
                
            image_path = os.path.join(path_to_assets,r'Prediction_VGG16_class_report_raabin_red.png')
            if os.path.exists(image_path):
                image = Image.open(image_path)
                st.image(image, width=300)
        


        else:
    
            st.write("Les images du dataset test sont simplement redimensionnées au format 360x360.")
            
            st.write("La matrice de confusion et le rapport de classification obtenus sont les suivants")
            
            image_path = os.path.join(path_to_assets,r'Prediction_VGG16_mat_report_raabin.png')
            if os.path.exists(image_path):
                image = Image.open(image_path)
                st.image(image, width=600) 
                
#            image = Image.open('Prediction_VGG16_mat_report_raabin.png')
#            st.image(image, width=600)  



 
        
# PREDICTIONS
    
    st.subheader('Regarder le résultat de prédiction pour une image individuelle (non réduite)')
    
    st.write("Menu de sélection de différents types d'images à ajouter")
    
    image_path = os.path.join(path_to_assets,r'MO_14626.jpg')
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, width=300)
 
#    image = Image.open('MO_14626.jpg')
#    st.image(image, width=300) 
    
    #Charger un .csv constituté d'images exemple ?
    X=image.resize((360,360))
    y='monocyte'
         
    st.write("L'image appartient à la classe ",y)
        
    modele=st.selectbox("Choisir une catégorie de modèle",['Random Forest','Convolutional Neural Network','Transfer Learning'],key=2)
    
    
    if modele== 'Random Forest':
        
        from sklearn.ensemble import RandomForestClassifier
##        from sklearn.preprocessing import MinMaxScaler as MMSc
        
        mod = joblib.load('Random_Forest_model_df70.pkl')
        
        img = X.convert("L") # convert to Gray mode
        img = img.resize((224,224)) # resize the image to (224, 224)
        X = np.asarray(img) # convert PIL Image object to numpy array
        
   #On n'a pas sauvegarder le scaler ni SelectPercentile, donc on ne peut pas réappliquer. 
   #Les données réduites sont sauvegardées sans les noms, donc on ne peut pas aller chercher la bonne image directement préparée
        
#        y_predict = mod.predict(X)
        
#        class_names=['Basophil','erythroblast','eosinophil','immunoglobulin','lymphocyte','monocyte','neutrophil','platelet']
#        st.write("L'images est classée comme ",class_names(y_predict))


    if modele== 'Convolutional Neural Network':


        
        #mod = joblib.load('CNN_FAV.h5',compile=False)
        mod = joblib.load('CNN_FAV_brutes.h5')
        mod.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    
        y_predict = mod.predict(X)
        y_predict = y_predict.argmax(axis=1)
    
        class_names=['Basophil','eosinophil','erythroblast','immunoglobulin','lymphocyte','monocyte','platelet','neutrophil']
    
        st.write("L'images est classée comme ",class_names(y_predict))

    if modele== 'Transfer Learning':


        
        #mod = joblib.load('modele_choisi',compile=False)
        mod = joblib.load('VGG16_c4.h5')
        mod.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        
        y_predict = mod.predict(X)
        y_predict = y_predict.argmax(axis=1)
    
        class_names=['Basophil','eosinophil','erythroblast','immunoglobulin','lymphocyte','monocyte','platelet','neutrophil']
    
        st.write("L'images est classée comme ",class_names(y_predict))
   
    
    
    #class_report, conf_matrix=predire(mod,X,y)
    
    
    #Affichage
    
    #st.print(class_report)
    
    #fig=plt.figure(figsize=(10,10))
    #plt.imshow(conf_matrix, interpolation='nearest',cmap='Blues')
    #plt.title("Confusion matrix test_data",fontsize=20)
    #plt.colorbar()
    #tick_marks = np.arange(len(class_names))
    #plt.xticks(tick_marks, class_names, rotation =90,fontsize=15)
    #plt.yticks(tick_marks, class_names,fontsize=15)
    #for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
    #    plt.text(j, i, conf_matrix[i, j], horizontalalignment = "center", color = "white" if conf_matrix[i, j] > ( conf_matrix.max() / 2) else "black",fontsize=15)
    #plt.ylabel('True labels',fontsize=20)
    #plt.xlabel('Predicts labels',fontsize=20)
    
    #st.plt(fig)
    
    

    
    
    
    
    
    
    
    
    
    