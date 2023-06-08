#!/usr/bin/env python
# coding: utf-8

# In[1]:
import streamlit as st

# Import de la page home

from home_page import home_streamlit, dataset_streamlit, conclusion_streamlit, about_streamlit

# Import de la page modeling

from tabs.Modeling_page import machine_learning_modelling, deep_learning_modelling

# import de la page EDA

from tabs.EDA_page import EDA_page

# import de la page prediction

#from tabs.prediction_page import predictions,predire A REINTRODUIRE APRES correction prediction
from tabs.prediction_page_v2_en_cours import predictions

# In[2]:

def main():
    
    liste_menu = ['Introduction', 'Dataset', 'EDA', 'Modelisation', 'Prediction', 'Conclusion']    
    menu = st.sidebar.radio('Sommaire', liste_menu)
    
    if menu == liste_menu[0]:
        home_streamlit()
    if menu == liste_menu[1]:
        dataset_streamlit()
    if menu == liste_menu[2]:
        EDA_page()
    if menu == liste_menu[3]:
        machine_learning_modelling()
        deep_learning_modelling()
    if menu==liste_menu[4]:
        predictions()
    if menu == liste_menu[5]:
        conclusion_streamlit()
    if menu == liste_menu[6]:
        about_streamlit()

if __name__ == '__main__':
    main()
    
# In[ ]:




