# -*- coding: utf-8 -*-
"""
@author: Planade
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image


# variables to create path to assets folder from the tabs one
#path_to_assets = os.path.join(os.path.dirname(
#    os.path.dirname(__file__)), 'Streamlit','assets')
path_to_assets = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')


def predictions():

    st.title("Utilisation d'une sélection de modèles sur des images test")

    st.write('Nous allons présenter ici des résultats sur le dataset Raabin')

    st.write("Les modèles utilisés sont ceux présentés dans la partie ML.")

    modele = st.radio("Choisir une catégorie de modèle",[ 'Random Forest', 'Convolutional Neural Networks', 'Transfer Learning'], key=41)
    #modele = st.radio("Choisir une catégorie de modèle"(Italics),[ 'Random Forest', 'Convolutional Neural Networks', 'Transfer Learning'], key=41)


# RANDOM FOREST

    if modele == 'Random Forest':

        st.subheader(
            'Machine Learning : modèle Random forest sur données test réduites')

        st.write("Les images du dataset Raabin sont préparées de la même manière que les images du dataset Barcelona Mendeley Data : ")
        st.info("réduction par resize à 224x224 pixels puis sélection des 20% de pixels les plus informatifs par SelectPercentile puis PCA conservant 70% de l'information.")

        st.write(
            "La matrice de confusion et le rapport de classification obtenus sont les suivants :")

        image_path = os.path.join(
            path_to_assets, r'Prediction_RF_mat_report_df70raabin.png')
        if os.path.exists(image_path):
            im = Image.open(image_path)
            st.image(im, width=600)
            
#        image_path = os.path.join(
#            path_to_assets, r'Prediction_RF_class_report_df70raabin.png')
#        if os.path.exists(image_path):
#            im = Image.open(image_path)
#            st.image(im, width=300)

#        image_path = os.path.join(
#            path_to_assets, r'Prediction_RF_conf_mat_df70raabin.png')
#        if os.path.exists(image_path):
#            im = Image.open(image_path)
#            st.image(im, width=300)


# CNNs

    if modele == 'Convolutional Neural Networks':

        st.subheader('Deep Learning : 2 modèles CNN sur données test')

        red1 = st.checkbox('cocher pour afficher les résultats sur données réduites', key=43)
        #red1 = st.checkbox('cocher pour les résultats sur données réduites',red:['cocher pour les résultats sur données réduites'], key=43)

        if red1:

            st.info("Les images du dataset test sont réduites par un resizing au format 224x224 puis par la sélection des 20% de pixels les plus porteurs d'information.")

            st.write(
                "Les matrices de confusion et les rapports de classification obtenus sont les suivants :")

            st.caption("Modèle simple architecture LeNet")

            image_path = os.path.join(
                path_to_assets, r'Prediction_LENET_mat_report_raabin_red.png')
            if os.path.exists(image_path):
                im = Image.open(image_path)
                st.image(im, width=600)

            st.caption("Modèle à fonctions d'activations variées")

            image_path = os.path.join(
                path_to_assets, r'Prediction_CNN_FAV_mat_report_raabin_red.png')
            if os.path.exists(image_path):
                im = Image.open(image_path)
                st.image(im, width=600)

        else:

            st.info(
                "Les images du dataset test sont simplement redimensionnées au format 360x360.")

            st.write(
                "Les matrices de confusion et les rapports de classification obtenus sont les suivants")

            st.caption("Modèle simple architecture LeNet")

            image_path = os.path.join(
                path_to_assets, r'Prediction_LENET_mat_report_raabin.png')
            if os.path.exists(image_path):
                im = Image.open(image_path)
                st.image(im, width=600)

            st.caption("Modèle à fonctions d'activations variées")

            image_path = os.path.join(
                path_to_assets, r'Prediction_CNN_FAV_mat_report_raabin.png')
            if os.path.exists(image_path):
                im = Image.open(image_path)
                st.image(im, width=600)


# VGG16

    if modele == 'Transfer Learning':

        st.subheader(
            'Deep Learning : modèle CNN boosté par TL, sur données test')

        red2 = st.checkbox(
            'cocher pour les résultats sur données réduites', key=44)

        if red2:

            st.info("Les images du dataset test sont réduites par un resizing au format 224x224 puis par la sélection des 20% de pixels les plus porteurs d'information.")

            st.write(
                "Les matrices de confusion et les rapports de classification obtenus sont les suivants")

            st.write("Modèle simple architecture LeNet")

            image_path = os.path.join(
                path_to_assets, r'Prediction_VGG16_conf_mat_raabin_red.png')
            if os.path.exists(image_path):
                im = Image.open(image_path)
                st.image(im, width=300)

            image_path = os.path.join(
                path_to_assets, r'Prediction_VGG16_class_report_raabin_red.png')
            if os.path.exists(image_path):
                im = Image.open(image_path)
                st.image(im, width=300)

        else:

            st.info(
                "Les images du dataset test sont simplement redimensionnées au format 360x360.")

            st.write(
                "La matrice de confusion et le rapport de classification obtenus sont les suivants")

            image_path = os.path.join(
                path_to_assets, r'Prediction_VGG16_mat_report_raabin.png')
            if os.path.exists(image_path):
                im = Image.open(image_path)
                st.image(im, width=600)


# PREDICTIONS

    st.subheader(
        'Regarder le résultat de prédiction pour une image individuelle (non réduite)')

    cat = st.selectbox("Choisir un type cellulaire", [
                       'basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'platelet', 'neutrophil'], key=49)

    image_path_1 = os.path.join(path_to_assets, 'PBC_dataset_normal_DIB_examples', cat)

    if cat == 'basophil':
        image_path = os.path.join(image_path_1, 'BA_47.jpg')
    if cat == 'eosinophil':
        image_path = os.path.join(image_path_1, 'EO_29763.jpg')        
    if cat == 'erythroblast':
        image_path = os.path.join(image_path_1, 'ERB_29121.jpg')        
    if cat == 'ig':
        image_path = os.path.join(image_path_1, 'IG_60755.jpg')        
    if cat == 'lymphocyte':
        image_path = os.path.join(image_path_1, 'LY_16036.jpg')        
    if cat == 'monocyte':
        image_path = os.path.join(image_path_1, 'MO_963471.jpg')        
    if cat == 'neutrophil':
        image_path = os.path.join(image_path_1, 'SNE_970608.jpg')        
    if cat == 'platelet':
        image_path = os.path.join(image_path_1, 'PLATELET_999931.jpg')        
       
    if os.path.exists(image_path):
        im = Image.open(image_path)
        st.image(im, width=300)

    X = im.resize((360, 360))
    test_image = image.img_to_array(X)
    test_image_norm = test_image / 255.0
    test_image_norm = np.expand_dims(test_image_norm, axis=0)

    modele = st.radio("Choisir une catégorie de modèle", [
                          '', 'Convolutional Neural Network', 'Transfer Learning'], key=42)

    if modele == 'Random Forest':

        st.write("On n'a pas sauvegardé les scaler : MinMax, SelectPercentile, et PCA, entraînés sur les données d'entraînement !")

        from sklearn.ensemble import RandomForestClassifier
##        from sklearn.preprocessing import MinMaxScaler as MMSc

        mod = joblib.load('Random_Forest_model_df70.pkl')

        img = X.convert("L")  # convert to Gray mode
        img = img.resize((224, 224))  # resize the image to (224, 224)
        X = np.asarray(img)  # convert PIL Image object to numpy array

        # On n'a pas sauvegarder le scaler ni SelectPercentile, donc on ne peut pas réappliquer.
        # Les données réduites sont sauvegardées sans les noms, donc on ne peut pas aller chercher la bonne image directement préparée

#        y_predict = mod.predict(X)

#        class_names=['basophil','erythroblast','eosinophil','imature granulocyte','lymphocyte','monocyte','neutrophil','platelet']
#        st.write("L'images est classée comme ",class_names(y_predict))

    if modele == 'Convolutional Neural Network':

        #        X = np.asarray(X)
        #mod = joblib.load('CNN_FAV.h5',compile=False)
        model_path = os.path.join(path_to_assets, r'CNN_lenet_brutes.h5')
#        mod = tf.keras.models.load_model('CNN_lenet_brutes.h5',compile=False)
        mod = tf.keras.models.load_model(model_path,compile=False)
        mod.compile(
            optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        y_proba = mod.predict(test_image_norm)
        y_predict = y_proba.argmax(axis=-1)

        class_names = ['basophil', 'eosinophil', 'erythroblast',
                       'imature granulocyte', 'lymphocyte', 'monocyte', 'platelet', 'neutrophil']
#        class_name=image_data.class_names
        df_proba = pd.DataFrame({"Catégories de cellules": class_names,
                                "Probabilité d'appartenance": y_proba[0, :]})
        df_proba =df_proba.set_index("Catégories de cellules")
        
        st.write("Avec le modèle simple d'architecture LeNet, cette image est classée en tant que ", sorted(
            class_names)[y_predict[0]])

        st.dataframe(df_proba.T, width=1300)
        
        model_path_2 = os.path.join(path_to_assets, r'CNN_FAV_brutes.h5')
        mod2 = tf.keras.models.load_model(model_path_2,compile=False)
#        mod2 = tf.keras.models.load_model('CNN_FAV_brutes.h5',compile=False)
        mod2.compile(
            optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        y_proba2 = mod2.predict(test_image_norm)
        y_predict2 = y_proba2.argmax(axis=-1)

        df_proba2 = pd.DataFrame({"Catégories de cellules": class_names,
                                "Probabilité d'appartenance": y_proba2[0, :]})
        df_proba2 =df_proba2.set_index("Catégories de cellules")
        
        st.write("Avec le modèle CCN_FAV, cette image est classée en tant que ", sorted(
            class_names)[y_predict2[0]])

        st.dataframe(df_proba2.T, width=1300)
#        st.write("L'images est classée comme ",class_names(y_predict[0]))

    if modele == 'Transfer Learning':
        model_path_3 = os.path.join(path_to_assets, r'vgg16_c4.h5')
        mod = tf.keras.models.load_model(model_path_3,compile=False)
#        mod = tf.keras.models.load_model('VGG16_c4.h5',compile=False)
        mod.compile(
            optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        y_proba = mod.predict(test_image_norm)
        y_predict = y_proba.argmax(axis=-1)
        
        class_names = ['basophil', 'eosinophil', 'erythroblast',
                       'imature granulocyte', 'lymphocyte', 'monocyte', 'platelet', 'neutrophil']
#        class_name=image_data.class_names
        df_proba = pd.DataFrame({"Catégories de cellules": class_names,
                                "Probabilité d'appartenance": y_proba[0, 1:]})
        df_proba =df_proba.set_index("Catégories de cellules")
        
        st.write("Avec le modèle VGG16 réentraîné sur 4 couches, cette image est classée en tant que ", sorted(
            class_names)[y_predict[0]-1])
#        st.write(y_predict)
        st.dataframe(df_proba.T, width=1300)



    # Affichage

    # st.print(class_report)

    # fig=plt.figure(figsize=(10,10))
    #plt.imshow(conf_matrix, interpolation='nearest',cmap='Blues')
    #plt.title("Confusion matrix test_data",fontsize=20)
    # plt.colorbar()
    #tick_marks = np.arange(len(class_names))
    #plt.xticks(tick_marks, class_names, rotation =90,fontsize=15)
    #plt.yticks(tick_marks, class_names,fontsize=15)
    # for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
    #    plt.text(j, i, conf_matrix[i, j], horizontalalignment = "center", color = "white" if conf_matrix[i, j] > ( conf_matrix.max() / 2) else "black",fontsize=15)
    #plt.ylabel('True labels',fontsize=20)
    #plt.xlabel('Predicts labels',fontsize=20)

    # st.plt(fig)
