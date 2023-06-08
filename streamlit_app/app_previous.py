#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from home_page import home_streamlit, dataset_streamlit, conclusion_streamlit
 #import de la page home

# In[2]:


def main():
    
    liste_menu =['Introduction', 'Dataset', 'EDA', 'Modelisation', 'Prediction', 'Conclusion']
    
    menu = st.sidebar.radio('Sommaire', liste_menu)
    
    if menu ==liste_menu[0]:
        home_streamlit()
    if menu == liste_menu[1]:
        dataset_streamlit()
    if menu == liste_menu[5]:
        conclusion_streamlit()
    

if __name__ == '__main__':
    main()
# In[ ]:




