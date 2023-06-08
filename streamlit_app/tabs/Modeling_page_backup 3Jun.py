#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st    
from PIL import Image
import os

path_to_assets = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')

def preprocessing():
    
    st.title("Préprocessing") 
    
    st.markdown("Cela fait référence à l'étape de transformation des données brutes en un format plus approprié pour l'entraînement d'un modèle. \
                Il s'agit d'une étape essentielle du pipeline de traitement des données dans de nombreux projets d'apprentissage automatique.")
    
    st.header("Machine Learning") 
    
    st.markdown("Le nombre conséquent d'images de la base de données utilisées lors de la modélisation pose problème pendant le chargement, en raison \
                des dimensions de 360 par 363 pixels sur 3 canaux RGB.")
    st.markdown("La PCA est réalisée sur les images moyennes obtenues lors de l’étape de visualisation des données.")
    st.markdown("Elle permet de faire une réduction intelligente des données, afin d'améliorer le temps de calcul pour la partie \
                Machine Learning.")

    st.header("Deep Learning") 
    
    st.markdown(" ")


def machine_learning_modelling():
    
    st.title("Modélisation")

    st.header("Machine Learning")    
    
    st.markdown("Le Machine Learning est un sous-domaine de l'intelligence artificielle qui concerne le développement de techniques permettant \
                aux ordinateurs d'apprendre et de s'améliorer à partir de données, sans être explicetement programmés.")
    st.markdown("Il existe différents types d'algorithmes de machine learning, notamment :")
    st.markdown(" • L'apprentissage supervisé : il consiste à fournir des données d'entraînement avec des étiquettes ou des résultats attendus, \
                afin que le modèle puisse apprendre à prédire ces résultats ;")
    st.markdown(" • L'apprentissage non supervisé : il ne nécessite pas de données étiquetées. Le modèle est chargé de découvrir des schémas ou \
                des structures cachées dans les données ;")
    st.markdown(" • L'apprentissage par renforcement : le modèle apprend par essais et erreurs.")
    st.markdown("Avant de débuter l’entraînement des modèles testés, une étape de normalisation des données a été effectuée par l’utilisation de la \
                fonction MinMax de la librairie scikit learn.")

    
    liste_model1 = ["", "Random Forest"]
    option1 = st.selectbox("Choisir une catégorie d'algorithme", liste_model1)
        
    if option1 == "Random Forest":
        st.write("La méthode Random Forest fait partie des apprentissages supervisés en Machine Learning.")
        st.write("Le Random Forest est construit à partir d'un ensemble d'arbres de décision. Chaque arbre est construit en utilisant un sous-ensemble \
                 aléatoire de données d'entraînement et un sous-ensemble aléatoire des caractéristiques disponibles. Ce sous-échantillonnage aléatoire \
                 permet de créer une diversité entre les arbres individuels, ce qui contribue à l'efficacité de l'algorithme.")
        st.write("Lorsqu'il est utilisé pour la classification, chaque arbre de décision d'un random forest prédit la classe d'une observation en fonction \
                 de ses caractéristiques. La prédiction finale du random forest est déterminée par un vote majoritaire parmi les arbres.")     
        
        image_path = os.path.join(path_to_assets,r"modelling_architecture_RF.png")
        image = Image.open(image_path)
        st.image(image, caption="Architecture du Random Forest")

        st.write("Les forêts aléatoires sont des modèles d'ensemble, qui semblent donc particulièrement appropriés à notre jeu de données.")
        st.write("La matrice de confusion a été générée à partir de df70.")
         
        image_path = os.path.join(path_to_assets,r"modelling_RF_matrix.png")
        image = Image.open(image_path)
        st.image(image, caption="Matrice de confusion")
         
        st.write("La performance de précision sur les données test ainsi que le score moyen affichent un résultat de 0.77.")


def deep_learning_modelling():

    st.header("Deep Learning")    

    st.markdown("Le Deep Learning est un domaine du Machine Learning. Basé sur des réseaux de neurones artificiels, il gère de grandes quantités de \
                données en ajoutant des couches au réseau. De nombreuses \
                couches de neurones sont interconnectés. Chaque couche traite les données reçues de la couche précédente et extrait progressivement des \
                caractéristiques plus complexes et abstraites.")
    
    st.subheader("• Convolutional Neural Network (CNN)") 
    st.markdown("Parmi les réseaux de neurones profonds, on trouve les réseaux de neurones convolutifs (CNN). Leurs applications sont axées notamment \
                sur la reconnaissance d’image et vidéo, les systèmes de recommendation, et le traitement du langage naturel. Les CNN sont composés de couches \
                convolutionnelles et de couches de pooling et peuvent être ajustées en utilisant des paramètres.")
   
    
    liste_model2 = ["", "LeNet", " CNN FAV"]
    option2 = st.selectbox("Choisir une catégorie d'algorithme", liste_model2)
        
    if option2 == "LeNet":
        st.write("L'architecture LeNet CNN est composée de 7 couches. La composition des couches se compose de 3 couches convolutives, \
                 2 couches de sous-échantillonnage et 2 couches entièrement connectées.")
    
        image_path = os.path.join(path_to_assets,r"modelling_architecture_lenet.png")
        image = Image.open(image_path)
        st.image(image, caption="Architecture de LeNet")
        
        st.write("L'algorithme est composé de 5 couches : \
                 • Une couche de convolution avec la fonction d’activation 'relu'; \
                 • Une couche de maxpooling ; \
                 • Un dropout ; \
                 • Un flatten ; \
                 • Une couche dense avec la fonction d'activation 'softmax'.")
                 
        st.write("Le modèle est compilé avec la fonction de perte 'sparse categorical crossentropy', l’optimizer adam et l’accuracy comme métrique. \
                 L’entrainement du modèle a été effectué sur 5 epochs.")
        st.write("La matrice de confusion et le classification report sont présentées ci-dessous.")
            
        image_path = os.path.join(path_to_assets,r"modelling_lenet_matrix.png")
        image = Image.open(image_path)
        st.image(image, caption="Matrice de confusion")
    
        st.write("Le modèle a réussi à généraliser sur ces données avec une accuracy de 0,43. ")

    if option2 == "CNN FAV":
        st.write("Les couches convolutives consistent à appliquer un filtre de convolution à l’image pour détecter des caractérisques de l’image. \
                 Une image passe à travers une succession de filtres créant de nouvelles images appelées cartes de convolutions.")
        st.write("Les filtres à travers une couche d’activation non linéaire appelée rectified Linear unit (relu) qui consiste à remplacer les nombres \
                 négatifs des images filtrées par des zéros.")
        st.write("La couche de pooling consiste à réduire progressivement la taille de l’image en ne gardant que les informations les plus importantes. \
                 Le pixel ayant la valeur maximale est appelé Max Pooling. Avec la couche pooling, la quantité de paramètres et de calcul dans le réseau \
                 sont reduits, et cela va permettre de controler le sur-apprentissage.")
        
        st.write("La matrice de confusion et le classification report sont présentées ci-dessous.")
        
        image_path = os.path.join(path_to_assets,r"modelling_CNN_FAV_matrix.png")
        image = Image.open(image_path)
        st.image(image, caption="Matrice de confusion")
        
        st.write(" ")

    st.subheader("• Transfer Learning")
    st.markdown("Le Transfer Learning est une approche de l'apprentissage automatique qui consiste à utiliser les connaissances acquises lors de la \
                résolution d'une tâche pour améliorer les performances sur une autre tâche apparentée. Au lieu d'entraîner un modèle à partir de zéro \
                sur une grande quantité de données, le transfert d'apprentissage permet de tirer parti des connaissances déjà acquises par un modèle \
                préalablement entraîné sur une tâche similaire.")
    
    
    liste_model3 = ["", "VGG16"]
    option3 = st.selectbox("Choisir une catégorie d'algorithme", liste_model3)

    if option3 == "VGG16": 
        st.write("Le modèle VGG16 que nous allons utiliser est un modèle pré-entrainé sur des millions d'images de la base ImageNet. Les images pour \
                 le jeu d'entraînement grâce à une librairie Keras.")
        st.write("Le modèle va donc apprendre sur le jeu d'entraînement puis valider les prédictions sur le jeu de validation. A noter que ces deux \
                 jeux de données sont labellisés : c'est-à -dire que l'on connaît les classes auxquelles appartiennent les images.")
                 
        st.write("Le modèle est composé de 6 couches : \
                 • Trois couches Dense avec la fonction d’activation 'relu'; \
                 • Une couche de GlobalAveragePooling ; \
                 • Deux dropout.")
        
        image_path = os.path.join(path_to_assets,r"modelling_architecture_VGG16.png")
        image = Image.open(image_path)
        st.image(image, caption="Architecture du VGG16")
        
        st.write("Le modèle a bien été entraîné avec une précision de 88.68%. Il n’a pas pu être entraîné en totalité en raison d’une interruption \
                 brutale de la machine.")
 
        st.write("La matrice de confusion et le classification report sont présentés ci-dessous.")       
 
        image_path = os.path.join(path_to_assets,r"modelling_VGG16_matrix.png")
        image = Image.open(image_path)
        st.image(image, caption="Matrice de confusion")
        
        st.write("")
        
        
