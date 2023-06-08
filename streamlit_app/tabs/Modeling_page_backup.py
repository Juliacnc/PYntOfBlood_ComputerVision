#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st    
from PIL import Image
   
def machine_learning_modelling():
    
    st.title("Modélisation")

    st.subheader('Machine Learning')    
    
    st.markdown("Le Machine Learning est un sous-domaine de l'intelligence artificielle qui concerne le développement de techniques permettant aux ordinateurs d'apprendre et de s'améliorer à partir de données, sans être explicetement programmés.")
    st.markdown("Avant de débuter l’entraînement des modèles testés, une étape de normalisation des données a été effectuée par l’utilisation de la fonction MinMax de la librairie scikit learn.")

    if st.checkbox("KNN"):
        st.write("La méthode KNN est implémentée sur les trois set de données (pca70, pca80, pca90) avec un nombre de voisins variant de 2 à 99, et deux métriques de calcul des distances entre points : « minkowsky », qui utilise le mode de calcul euclidien par défaut (une comparaison avec le mode « euclidean » montre que l'on obtient bien le même résultat), et « manhattan ».")
               
        image_1 = Image.open("Image3.gif")
        st.image(image_1, caption="KNN")
        
        st.write("Un overfitting dans le cas d'un entraînement où le poids des voisins est pondéré par leur distance au point recherché.")
        
        image_2 = Image.open("Image4.png")
        st.image(image_2, caption="Courbes de score")
        
        st.write("La matrice de confusion a été générée à partir de df90.")
        
        image_3 = Image.open("Image5.png")
        st.image(image_3, caption="Matrice de confusion")
        
        st.write("L’algorithme fonctionne sur les données du df90. On n'observe pas d’effondrement de précision avec les données normalisées.")

def deep_learning_modelling():

    st.subheader("Deep Learning")    

    st.markdown("Le Deep Learning est un domaine du Machine Learning. Basé sur des réseaux de neurones artificielles, il gère de grandes quantités de données en ajoutant des couches au réseau. Les réseaux de neurones artificiels utilisés dans le deep learning sont composés de nombreuses couches de neurones interconnectés. Chaque couche traite les données reçues de la couche précédente et extrait progressivement des caractéristiques plus complexes et abstraites.")
    
    st.caption("Convolutional Neural Network") 
    st.markdown("Parmi les réseaux de neurones profonds, on trouve les réseaux de neurones convolutifs appelés CNN. Leurs applications sont axées notamment sur la reconnaissance d’image et vidéo, les systèmes de recommandation, et le traitement du langage naturel. Les CNN sont composés de couches convolutionnelles et de couches de pooling peuvent être ajustées en utilisant des paramètres.")
   
    if st.checkbox("CNN"):
        st.write("Le CNN est un type de réseau neuronal profond spécialement conçu pour le traitement des données en grille, telles que des images ou des vidéos. Il est principalement utilisé dans le domaine de la vision par ordinateur pour des tâches telles que la classification d'images, la détection d'objets et la segmentation sémantique. Les CNN ont révolutionné le domaine en permettant des avancées significatives dans la capacité des machines à comprendre et à analyser visuellement le monde qui les entoure.")
        st.write("Les couches convolutives consistent à appliquer un filtre de convolution à l’image pour détecter des caractérisques de l’image. Une image passe à travers une succession de filtres créant de nouvelles images appelées cartes de convolutions.")
        st.write("Les filtres à travers une couche d’activation non linéaire appelée rectified Linear unit (relu) qui consiste à remplacer les nombres négatifs des images filtrées par des zéros.")
        st.write("La couche de pooling consiste à réduire progressivement la taille de l’image en ne gardant que les informations les plus importantes. Le pixel ayant la valeur maximale est appelé Max Pooling. Avec la couche pooling, la quantité de paramètres et de calcul dans le réseau sont reduits, et cela va permettre de controler le sur-apprentissage.")
    
        st.write(" ")
    
        image_4 = Image.open("Image6.jpg")
        st.image(image_4, caption="Architecture du CNN")
        
        image_5 = Image.open("Image7.png")
        st.image(image_5, caption="Courbes de score")  
        
        st.write("On obtient une accuracy de 0,99 et une val_accuracy de 0,92. Les résultats de validation sont corrects, mais des valeurs aussi élevées pourraient révéler un overfitting assez important.")
            
        image_6 = Image.open("Image8.png")
        st.image(image_6, caption="Matrice de confusion")
    
        st.write("On obtient une accuracy générale de 0,16, qui, tout en restant insuffisante, est légèrement supérieure à la distribution au hasard (0,12). Ici, ce sont les IG qui obtiennent le meilleur f1-score à 0,38 avec une précision de 0,24 et un rappel de 0,97. Le modèle ne détecte pas correctement les IG mais lorsqu’il les détecte il les classe correctement.")


    st.caption('Transfer Learning')
    st.markdown("Le Transfer Learning est une approche de l'apprentissage automatique qui consiste à utiliser les connaissances acquises lors de la résolution d'une tâche pour améliorer les performances sur une autre tâche apparentée. Au lieu d'entraîner un modèle à partir de zéro sur une grande quantité de données, le transfert d'apprentissage permet de tirer parti des connaissances déjà acquises par un modèle préalablement entraîné sur une tâche similaire.")
    
    if st.checkbox("EfficientNet"):
        st.write("A ce modèle, ont été ajoutées : une couche de GlobalAveragePooling2D, deux paires de couches Dense et Dropout à 0,2, une couche Flatten et enfin une couche Dense. Les fonctions d’activation sont des ‘relu’ à l’exception de la fonction de la dernière qui est un ‘softmax’. Le modèle est compilé avec la loss ‘sparse categorical cross entropy’, l’optimiseur ‘adam’ et la métrique ‘accuracy’. Le modèle est entraîné sur 5 epochs.")
        
        image_7 = Image.open("Image9.png")
        st.image(image_7, caption="Architecture de l'EfficientNet")
        
        st.write("On obtient une accuracy de 0,91 et une val_accuracy de 0,93. Cependant, lorsque l’on évalue le modèle sur les données test, on descend à une accuracy de 0,0006 ce qui est très en dessous des résultats obtenus avec le CNN.")
 
        image_8 = Image.open("Image10.png")
        st.image(image_8, caption="Courbe de score")
    
        
