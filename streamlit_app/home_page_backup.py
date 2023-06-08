#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import os

# In[4]:


def home_streamlit():
    #lien vers assets où sont hébergées les photos
    path_to_assets= os.path.join(os.path.dirname(__file__),'assets')
    st.title('PYnt of Blood')

    st.text('Jessica Planade, Julia Canac, Richard Bonfils, Frédéric Navez')
    
    st.caption('_Formation bootcamp-MARS23_')
    
    from PIL import Image
    image_path = os.path.join(path_to_assets,r'Image1.png')
    image = Image.open(image_path)
    st.image(image, caption='Blood cells')
    
    st.header('Objectif du projet')
    
    st.markdown("Le diagnostic d’un cancer nécessite plusieurs examens dont l’un des plus importants est l’examen histologique.\
    Un prélèvement sanguin ou une biopsie est effectué et une lame est observée par microscopie, ce qui permet de confirmer ou non la présence d’anomalie, identifier les cellules saines ou touchées et poser définitivement le diagnostic.")
    
    st.markdown("Dans le cas de l’hématologie, la numération et formule sanguine (NFS) doit être réalisée. Cependant,\
    avant de savoir si une cellule est saine ou cancéreuse, il est important de savoir de quel type de cellule il s'agit.")
    
    st.markdown('**Ce projet vise donc à développer un modèle de computer vision permettant d’identifier 8 différents types de cellules du sang.**')
    
    st.subheader('Les différents types cellulaires :')
    st.markdown('Au total, 8 types de cellule sanguine ont été explorés:')
    
    image_path = os.path.join(path_to_assets,r'Image2.png')
    image_hemato = Image.open(image_path)
    
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
    
    st.markdown("Les cellules du sang proviennent toute d'un même précurseur (*la cellule souche hématopoïétique*) et se diversifient au cours de **l'hématopoïèse**.")
    st.markdown("Chaque cellule va alors présenter des particularités morphologiques, notamment:")
    st.markdown("- Granulocytes : cytoplasme granuleux")
    st.markdown("- Neutrophiles : deux fois plus grand que les globules rouges, noyau scindé en trois parties")
    st.markdown("- Eosinophiles : noyau scindé en deux parties ;")
    st.markdown("- Basophiles : trois fois plus grands que les globules rouges, noyau plus foncé")
    st.markdown("- Lymphocytes : taille similaire aux globules rouges")
    st.markdown("- Monocytes : noyau en forme de rein, quatre à cinq fois plus grand que les globules rouges.")
    
    
def dataset_streamlit():
    st.header('Dataset')
    st.markdown('Deux dataset vont être utilisés durant cette présentation.')
    
    st.subheader("Dataset 'Barcelona Mendeley Data'" )
    st.markdown("Ce [dataset](https://www.sciencedirect.com/science/article/pii/S2352340920303681) provient d’un article scientifique '*A dataset for microscopic peripheral blood cell images for development of automatic recognition system*'. Les images sont réparties dans 8 dossiers différents, correspondant aux catégories de cellules suivantes: basophiles, éosinophiles, érythroblastes, granulocytes immatures, lymphocytes, monocytes, plaquettes et neutrophiles. Elles sont au format .jpg et de taille 363x360 en RGB.")
    st.markdown("Toutes les images sont issues de patients sains (absence de maladies, infections, allergies)")
    
    
    st.subheader("Dataset Test 'Raabin Data'")
    st.markdown("Une étude réalisée à Téhéran en 2021 se focalisant sur les leucocytes offre l’avantage de mettre à disposition une quantité significative d’images plus diversifiées et déjà labelisées pour 5 classes sur les 8 utilisées dans l'étude de Barcelone.")
    st.markdown("Il a été décidé d’utiliser la totalité des images en tant que test data. Pour palier à l’absence d’images sur les 3 classes absentes, quelques images (3 à 4) jugées caractéristiques (mais toujours non standardisées) provenant de sites web de département d’histologie ont été ajoutées dans les répertoires (Erythroblast), Immature Granulocytes (IG), et Platelet) afin d’avoir un résultat même restreint pour tous les types de cellules.")

def conclusion_streamlit():
    st.header('Conclusion')
    df_machineL = pd.DataFrame({'Name model': ['KNN df70 (unnormalized)', 'KNN df80 (unnormalized)',
                                          'KNN df90 (unnormalized)','KNN df70 (normalized)',
                                           'KNN df80 (normalized)','KNN df90 (normalized)'],
                           'Best test accuracy':[0.7859,0.76046,0.76046,0.75402,0.66189,0.29599],
                           'F1_score interval':['-','-','-', '0.47-0.97', '0.20-0.86', '0.01-0.24'],
                           'Best predictions': ['-','-','-', 'Platelet', 'Platelet', 'Platelet'],
                           'Worst predictions':['-','-','-', 'Basophil', 'Basophil', 'All except platelet']})
    liste_modele = ['Machine Learning', 'Deep Learning', 'Transfert Learning']
    option = st.selectbox("Choisir une catégorie d'algorithme", liste_modele)
    
    if option == 'Machine Learning':
        st.dataframe(df_machineL, width=1000)
    
# In[ ]:




