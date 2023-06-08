"""
@author: Richard B
"""
import os
from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random

#variables to create path to assets folder from the tabs one
path_to_assets = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')
#variables to create path to data folder from the tabs one
path_to_data = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')

def EDA_page():
    st.header ('Exploratory Data Analysis (EDA)')

    st.info ('3 bases de données étaient référencées dans la description du projet')
    # texte
    text=''
    st.markdown("Une analyse du contenu de chacun de ces 3 datasets a été effectuée en phase préparatoire")
    st.markdown("Leur contenu ont été consolidés dans des dictionnaires et dataframes successifs")
    st.markdown("Quelques données générales sur le contenu total des 3 datasets et leurs **43221** fichiers disponibles")

    with st.expander("Détail des données consolidées", expanded=False):
        req_info = st.radio("Merci de sélectionner la catégorie qui vous intéresse", ["Type de documents", "Catégories de cellules représentées","Dataframe"],index=0, key=311)
        if req_info == "Type de documents":
            image_path = os.path.join(path_to_assets,r'File_types_per_dataset_NormScale.png')
            if os.path.exists(image_path):
                image = Image.open(image_path)
                st.image(image, width=600)
            else:
                st.error(f"Image file not found: {image_path}")
            image_path_2 = os.path.join(path_to_assets,r'File_types_per_dataset_LogScale.png')
            if os.path.exists(image_path_2):
                image = Image.open(image_path_2)
                st.image(image, width=600)
            else:
                st.error(f"Image file not found: {image_path}")
        if req_info == "Catégories de cellules représentées":
            image_path = os.path.join(path_to_assets,r'Nb_classtype_by_dataset.png')
            if os.path.exists(image_path):
                image = Image.open(image_path)
                st.image(image, width=600)
                st.text("Catégories représentées")
            else:
                st.error(f"Image file not found: {image_path}")
        if req_info == "Dataframe":
            csv_lines_2 = st.slider("Spécifiez le nombre de lignes désirées", min_value=1, max_value=50, value=5,key=32)
            head_tail = st.radio("Ordre des lignes",["Head","Tail","Shuffle"],index=0)
            dataframe_path = os.path.join(path_to_data, 'file_info_concat_with_categories.csv')
            df_three_datasets = pd.read_csv(dataframe_path)
            if head_tail== "Head":
                st.dataframe(df_three_datasets.head(csv_lines_2))
            if head_tail== "Tail":
                st.dataframe(df_three_datasets.tail(csv_lines_2))
            if head_tail== "Shuffle":
                random_lines = df_three_datasets.sample(n=csv_lines_2)
                st.dataframe(random_lines)
#        else: 
#           st.text("No info requested")

    st.info("Pour simplifier l'apprentissage, la priorité a été donnée à l'analyse et l'utilisation du **dataset de Barcelone**, qui contient des images de patients sains.")
    st.markdown("Après un nettoyage rapide, consistant essentiellement en")
    st.markdown("- la vérification de l'absence de doublons")
    st.markdown("- la détection et la suppression d'un fichier corrompu")
    st.markdown("... nous avons procédé à différentes transformation des images à fin d'analyse")
    st.info("Un dataframe de 17092 lignes a été consolidé après parcours des différents répertoires du dataset.")


    with st.expander("Exemples de cellules de l'étude Barcelone", expanded=False):
        dataframe_path = os.path.join(path_to_data, 'Barcelone_local.csv')
    # Read the DataFrame
        df_barcelona = pd.read_csv(dataframe_path)
    # Get unique cell types
        unique_cell_types = df_barcelona['cell_type'].unique()
    # Get user inputs
        selected_cell_types = st.multiselect("Select Cell Types", unique_cell_types, default=['eosinophil'])
    # Filter the DataFrame based on selected cell types
        filtered_df = df_barcelona[df_barcelona['cell_type'].isin(selected_cell_types)]
    # Randomly select nb_cells images
        nb_cells = st.slider("Select Number of Cells", 2, 10)
        random_images = filtered_df.groupby('cell_type').apply(lambda x: x.sample(nb_cells)).reset_index(drop=True)
    # Calculate the number of rows and columns for the grid
        num_images = len(random_images)
        num_cols = min(num_images, 4)
#        num_rows = (num_images - 1) // num_cols + 1
        num_rows = (num_images) // num_cols + 1
    # Iterate over the unique cell types
        for cell_type in selected_cell_types:
    # Filter the DataFrame for the current cell type
            filtered_df = df_barcelona[df_barcelona['cell_type'] == cell_type]
    # Randomly select nb_cells images for the current cell type
            random_images = filtered_df.sample(nb_cells)
    # Calculate the number of rows and columns for the grid
            num_images = len(random_images)
            num_cols = min(num_images, 4)
            if num_images == 0:
                continue  # Skip iteration if no images for the current cell type
            num_rows = (num_images - 1) // num_cols + 1
    # Display the cell type as a header
            st.subheader(cell_type)
    # Create the grid figure
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))
            axs = axs.flatten()
    # Iterate over the subplots and images
            for i, (ax, _, row) in enumerate(zip(axs, range(num_images), random_images.iterrows())):
        # Get the full image path
                image_path = row[1]['chemin'] + row[1]['image']
        #image_path = row['chemin'] + row['image']
        # Read the image
                image = plt.imread(image_path)
        # Display the image on the corresponding subplot
                ax.imshow(image)
                ax.axis('off')
    # Remove empty subplots
            for j in range(len(random_images), len(axs)):
                axs[j].remove()
    # Show the grid
            plt.tight_layout()
            st.pyplot(fig)


    with st.expander("Description du dataframe", expanded=False):
        csv_lines = st.slider("Spécifiez le nombre de lignes désirées", min_value=1, max_value=50, value=5,key=33)
        head_tail = st.radio("Ordre des lignes",["Head","Tail","Shuffle"],index=0, key=34)
        dataframe_path = os.path.join(path_to_data, 'df_barc.csv')
        df_barcelona = pd.read_csv(dataframe_path)
        if head_tail== "Head":
            st.dataframe(df_barcelona.head(csv_lines))
        if head_tail== "Tail":
            st.dataframe(df_barcelona.tail(csv_lines))
        if head_tail== "Shuffle":
            random_lines = df_barcelona.sample(n=csv_lines)
            st.dataframe(random_lines)

    with st.expander("Contenu du DataSet, Nombre d'images par catégorie", expanded=False):
        image_path = os.path.join(path_to_assets,r'Barc_nb_per_cat.png')
        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(image, width=600)
        else:
            st.text("image not found")

    st.markdown("L'aspect de chaque type de cellules a été analysé, en terme de")
    st.markdown("- valeurs moyennes de pixels, qui donne une image moyenne")
    st.markdown("- valeurs moyennes des valeurs des 3 couleurs RGB")
    st.markdown("Les images suivantes affichent ces informations pour chacun des types de cellule, ainsi qu'une image prise au hasard")
    with st.expander("Images par catégorie Barcelone", expanded=False):
        categories_options = st.multiselect('Catégories de celllules sélectionnées',['Basophil (BA)','Eosinophil (EO)','Erythroblast (ER)','Immature Granulocyte (IG)','Lymphocyte (LM)','Monocyte (MON)','Plaquette (PLA)','Neutrophile (NEU)'], key=35)
        st.write('Images par catégorie', categories_options)
        #['Basophil (BA)','Eosinophil (EO)'\,'Erythroblast (ER)','Immature Granulocyte (IG)','Lymphocyte (LM)','Monocyte (MON)','Plaquette (PLA)','Neutrophile (NEU)'])
        if 'Erythroblast (ER)' in categories_options:
            image_path = os.path.join(path_to_assets,r'Erythroblast_Barcelona.png')
            if os.path.exists(image_path):
                image = Image.open(image_path)
                st.image(image, width=600)
        if 'Basophile (BA)' in categories_options:
            image_path_2 = os.path.join(path_to_assets,r'Basophil_Barcelona.png')
            if os.path.exists(image_path_2):
                image = Image.open(image_path_2)
                st.image(image, width=600)
        if 'Eosinophil (EO)' in categories_options:
            image_path_3 = os.path.join(path_to_assets,r'Eosinophil_Barcelona.png')
            if os.path.exists(image_path_3):
                image = Image.open(image_path_3)
                st.image(image, width=600)
        if 'Immature Granulocyte (IG)' in categories_options:
            image_path_4 = os.path.join(path_to_assets,r'Ig_Barcelona.png')
            if os.path.exists(image_path_4):
                image = Image.open(image_path_4)
                st.image(image, width=600)
        if 'Lymphocyte (LM)' in categories_options:
            image_path_5 = os.path.join(path_to_assets,r'Lymphocyte_Barcelona.png')
            if os.path.exists(image_path_5):
                image = Image.open(image_path_5)
                st.image(image, width=600)        
        if 'Neutrophil (NEU)' in categories_options:
            image_path_6 = os.path.join(path_to_assets,r'Neutrophil_Barcelona.png')
            if os.path.exists(image_path_6):
                image = Image.open(image_path_6)
                st.image(image, width=600)        
        if 'Monocyte (MON)' in categories_options:
            image_path_7 = os.path.join(path_to_assets,r'Monocyte_Barcelona.png')
            if os.path.exists(image_path_7):
                image = Image.open(image_path_7)
                st.image(image, width=600)        
        if 'Plaquette (PLA)' in categories_options:
            image_path_8 = os.path.join(path_to_assets,r'Platelet_Barcelona.png')
            if os.path.exists(image_path_8):
                image = Image.open(image_path_8)
                st.image(image, width=600)        
        st.markdown("Pour les catégories sélectionnées, respectivement: une image prise au hasard, son spectre RGB,le spectre RGB de l'image moyenne de cette catégorie, et l'image moyenne elle-même.")

    st.markdown('Similairement, pour chaque type de cellules, nous avons déterminé')
    st.markdown('- une image exemple prise au hasard')
    st.markdown('- une image moyenne')
    st.markdown('- une image réduite au nb de pixels, mean cell detection')
    
    with st.expander("Images par catégorie Barcelone, détection", expanded=False):
        image_path_9 = os.path.join(path_to_assets,r'EDA_Barcelona_Samples_Mean_MeanCellDetection.png')
        if os.path.exists(image_path_9):
            image = Image.open(image_path_9)
            st.image(image, width=600)
        else:
            st.text("image not found")

    st.markdown("Un dictionnaire contenant une image moyenne, par catégorie a ensuite été généré.\
        Pour se faire il a fallu redimensionner certaines images qui étaient de taille différente :\
            la méthode resize de cv2 a été utilisée")
    #with st.expander("Description du dataframe d'images moyennes", expanded=False):
    #    dataframe_path = os.path.join(path_to_data, 'df_barc_images_moy_seuil.csv')
    #    df_barcelona_2 = pd.read_csv(dataframe_path)
    #    st.dataframe(df_barcelona_2.head(csv_lines))
    #    st.text("Images moyennes")

    st.markdown("Une première étape de seuillage sur des images type de chaque catégorie a permis de mesurer\
        la taille moyenne de la surface projetée des cellules\
        , en pixel, pour chaque type cellulaire.")
    with st.expander("Description des images seuillées", expanded=False):
        dataframe_path = os.path.join(path_to_data, 'df_barc_image_seuil_par_cat.csv')
        df_barcelona_3 = pd.read_csv(dataframe_path)
        st.dataframe(df_barcelona_3.head(csv_lines))
        st.text("Images seuil par catégorie")

    st.info("Une analyse en composantes principales a été réalisée sur l’ensemble des images du dataset\
        afin de générer un fichier de travail plus léger contenant des features importantes pour les étapes d'entraînement\
        (réduction des données)")
    with st.expander("Analyse en composantes principales", expanded=False):
        image_path = os.path.join(path_to_assets,r'Barcelona_PrincipalComponentAnalysis_MeanImages.jpg')
        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(image, width=600)
        st.markdown("On constate qu’on a bien une discrimination entre les différentes catégories,\
            permettant d'utiliser cette stratégie pour réduire la taille des données\
            tout en conservant leurs caractéristiques singulières")
        dataframe_path_4 = os.path.join(path_to_data, 'images_pca_90.csv')
        df_barcelona_4 = pd.read_csv(dataframe_path_4)
        st.dataframe(df_barcelona_4.head(csv_lines))
        st.text("PCA90_Barcelona")
        dataframe_path_5 = os.path.join(path_to_data, 'images_pca_80.csv')
        df_barcelona_5 = pd.read_csv(dataframe_path_5)
        st.dataframe(df_barcelona_5.head(csv_lines))
        st.text("PCA80_Barcelona")
        dataframe_path_6 = os.path.join(path_to_data, 'images_pca_70.csv')
        df_barcelona_6 = pd.read_csv(dataframe_path_6)
        st.dataframe(df_barcelona_6.head(csv_lines))
        st.text("PCA70_Barcelona")

    with st.expander("Taux de réductions", expanded=False):
        dataframe_path_7 = os.path.join(path_to_data, 'EDA_Features_Summary.csv')
        encodings = ['utf-8', 'latin1', 'utf-16']
        df_EDA_features = None
        for encoding in encodings:
            try:
                df_EDA_features = pd.read_csv(dataframe_path_7, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        if df_EDA_features is not None:
            st.dataframe(df_EDA_features.head(5))
            st.text("Différents niveaux de réduction")
        else:
            st.error("Failed to read the CSV file. Please check the encoding or file format.")
            df_EDA_features = pd.read_csv(dataframe_path_7)
            st.dataframe(df_EDA_features.head(5))
            st.text("Différents niveaux de réduction")


    st.info("Une analyse et un traitement similaires ont été appliqués au dataset de Raabin, utilisé comme échantillon test de nos modèles entrainés")
    st.markdown('[al., Z. M. (2022)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8782871/)')
    st.markdown('*A large dataset of white blood cells containing cell locations and types, along with segmented nuclei and cytoplasm. Scientific Reports (Sci Rep) nature research*')
    with st.expander("Contenu du DataSet, Nombre d'images par catégorie", expanded=False):
        image_path = os.path.join(path_to_assets,r'Raabin_nb_per_cat_logscale.png')
        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(image, width=600)
        image_path = os.path.join(path_to_assets,r'Raabin_nb_per_cat.png')
        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(image, width=600)

    with st.expander("Images par catégorie dataset (test) Raabin", expanded=False):
        categories_options = st.multiselect('Catégories de celllules sélectionnées',['Basophil (BA)','Eosinophil (EO)','Erythroblast (ER)','Immature Granulocyte (IG)','Lymphocyte (LM)','Monocyte (MON)','Plaquette (PLA)','Neutrophile (NEU)'],key=36)
        st.write('Images par catégorie', categories_options)
        #['Basophil (BA)','Eosinophil (EO)'\,'Erythroblast (ER)','Immature Granulocyte (IG)','Lymphocyte (LM)','Monocyte (MON)','Plaquette (PLA)','Neutrophile (NEU)'])
        if 'Erythroblast (ER)' in categories_options:
            image_path = os.path.join(path_to_assets,r'Erythroblast_Raabin.png')
            if os.path.exists(image_path):
                image = Image.open(image_path)
                st.image(image, width=600)
        if 'Basophile (BA)' in categories_options:
            image_path_2 = os.path.join(path_to_assets,r'Basophil_Raabin.png')
            if os.path.exists(image_path_2):
                image = Image.open(image_path_2)
                st.image(image, width=600)
        if 'Eosinophil (EO)' in categories_options:
            image_path_3 = os.path.join(path_to_assets,r'Eosinophil_Raabin.png')
            if os.path.exists(image_path_3):
                image = Image.open(image_path_3)
                st.image(image, width=600)
        if 'Immature Granulocyte (IG)' in categories_options:
            image_path_4 = os.path.join(path_to_assets,r'Ig_Raabin.png')
            if os.path.exists(image_path_4):
                image = Image.open(image_path_4)
                st.image(image, width=600)
        if 'Lymphocyte (LM)' in categories_options:
            image_path_5 = os.path.join(path_to_assets,r'Lymphocyte_Raabin.png')
            if os.path.exists(image_path_5):
                image = Image.open(image_path_5)
                st.image(image, width=600)        
        if 'Neutrophil (NEU)' in categories_options:
            image_path_6 = os.path.join(path_to_assets,r'Neutrophil_Raabin.png')
            if os.path.exists(image_path_6):
                image = Image.open(image_path_6)
                st.image(image, width=600)        
        if 'Monocyte (MON)' in categories_options:
            image_path_7 = os.path.join(path_to_assets,r'Monocyte_Raabin.png')
            if os.path.exists(image_path_7):
                image = Image.open(image_path_7)
                st.image(image, width=600)        
        if 'Plaquette (PLA)' in categories_options:
            image_path_8 = os.path.join(path_to_assets,r'Platelet_Raabin.png')
            if os.path.exists(image_path_8):
                image = Image.open(image_path_8)
                st.image(image, width=600)        
        st.markdown("Pour les catégories sélectionnées, respectivement: une image prise au hasard, son spectre RGB,le spectre RGB de l'image moyenne de cette catégorie, et l'image moyenne elle-même.")
        st.markdown("L'aspect particulèrement flou des catégories 'platelet', 'Erythroblast', 'Immature granulocyte'est lié au faible nombre d'images présentes. L’effet de moyenne est insuffisant pour faire ressortir l’image type des cellules, tout comme des pics bien dessinés.")
        
    st.markdown("Pour finir, une image correspondant à la 'réduction intelligente', selective percentile 20 a été réalisée pour chacun des deux datasets.")
    with st.expander("images selected percentile", expanded=False):
        dataset_options2 = st.multiselect('Sélection du ou des Dataset(s)',['Barcelone','Raabin'],key=37)
        if 'Barcelone' in dataset_options2:
            image_path = os.path.join(path_to_assets,r'Image_réduite_Barcelone.png')
            if os.path.exists(image_path):
                image = Image.open(image_path)
                st.image(image, width=600)
                st.text('images réduites Barcelone')
        if 'Raabin' in dataset_options2:
            image_path_2 = os.path.join(path_to_assets,r'Image_réduite_Raabin.png')
            if os.path.exists(image_path_2):
                image = Image.open(image_path_2)
                st.image(image, width=600)
                st.text('images réduites Raabin')
EDA_page()
