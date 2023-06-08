# -*- coding: utf-8 -*-
"""
@author: Planade
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report
import itertools
import matplotlib.pyplot as plt
import seaborn as sns


def predire(mod,X,y):
    y_predict = mod.predict(X)
    if modele== 'Convolutional Neural Network' | 'Transfer Learning':
        y_predict = y_predict.argmax(axis=1)
    class_report = classification_report(y, y_predict)
    conf_matrix = cm(y, y_predict)

    # renvoie classification report et confusion matrix
    return class_report, conf_matrix



def predictions():
    
    st.title("Utilisation d'une sélection de modèles sur des images test")

 
   
    
    modele=st.selectbox("Choisir une catégorie de modèle",['Random Forest','Convolutional Neural Network','Transfer Learning'])
    
    
    if modele== 'Random Forest':
        
        st.subheader('Machine Learning : modèle Random forest sur données test réduites') 
        
        st.write("Les images du dataset test sont préparée de la même manière que les images du dataset Barcelona Mendeley Data, afin d'obtenir un fichier réduit.")
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import MinMaxScaler as MMSc
        
        mod = joblib.load('RFpca70.pkl')
        
        data=pd.read_csv('imagesTest_pca70.csv')
        data = data.drop(['Unnamed: 0','index'], axis=1)
        
        X = data.drop('code', axis=1)
        y=data['code']
        y=y.replace(['BA', 'ER', 'EO', 'IG', 'LYM','MON', 'SNE', 'PLA'], [1,2,3,4,5,6,7,8])
        X=MMSc.fit_transform(X)
        
        class_name=['Basophil','erythrophil','eosinophil','immunoglobulin','lymphocyte','monocyte','neutrophil','platelet']


    if modele== 'Convolutional Neural Network':
        
        st.subheader('Deep Learning : modèle CNN sur données test') 
        
        st.write("Les images du dataset test sont simplement redimensionnées au format 360x360.")
        

        from tensorflow.keras.preprocessing import image_dataset_from_directory
        import pathlib
        
        #mod = joblib.load('CNN_FAV.h5',compile=False)
        mod = joblib.load('CNN_FAV.h5')
        mod.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        
        data=pathlib.Path('TestA')
        X = image_dataset_from_directory(data,subset=None,seed=123,image_size=(360,360))
        y = np.concatenate([y for x,y in X])
    
        class_name=['Basophil','eosinophil','erythrophil','immunoglobulin','lymphocyte','monocyte','neutrophil','platelet']
    
        st.write("Le modèle utilisé ici est le réseau de neurones présenté dans la partie ML. Il produit, sur les données d'entraînement, ....................;.")
    


    if modele== 'Transfer Learning':
        
        st.subheader('Deep Learning : modèle CNN boosté par TL, sur données test') 
        
        st.write("Les images du dataset test sont simplement redimensionnées au format 360x360.")

        from tensorflow.keras.preprocessing import image_dataset_from_directory
        import pathlib
        
        #mod = joblib.load('modele_choisi',compile=False)
        mod = joblib.load('modele_choisi')
        mod.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        
        data=pathlib.Path('TestA')
        X = image_dataset_from_directory(data,subset=None,seed=123,image_size=(360,360))
        y = np.concatenate([y for x,y in X])
    
        class_name=['Basophil','eosinophil','erythrophil','immunoglobulin','lymphocyte','monocyte','neutrophil','platelet']
         
    
        
    #prédiction     
    
    predire(mod,X,y)
    
    
    #Affichage
    
    st.print(class_report)
    
    fig=plt.figure(figsize=(10,10))
    plt.imshow(conf_matrix, interpolation='nearest',cmap='Blues')
    plt.title("Confusion matrix test_data",fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation =90,fontsize=15)
    plt.yticks(tick_marks, class_names,fontsize=15)
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, conf_matrix[i, j], horizontalalignment = "center", color = "white" if conf_matrix[i, j] > ( conf_matrix.max() / 2) else "black",fontsize=15)
    plt.ylabel('True labels',fontsize=20)
    plt.xlabel('Predicts labels',fontsize=20)
    
    st.plt(fig)
    
    

    
    
    
    
    
    
    
    
    
    