# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 13:18:30 2021

@author: franc
"""

import NLPClass
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter, OrderedDict
# from tqdm.notebook import tqdm

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import TensorDataset, DataLoader

import torchtext
from torchtext.data import get_tokenizer

# from sklearn.utils import shuffle
# from sklearn.metrics import classification_report
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer

from googletrans import Translator
# pip install googletrans==4.0.0rc1

import pickle
# pip install pickle-mixin

from nltk.corpus import wordnet as wn
###########################----------------------###########################
# Testeo los métodos


# Levanto la base de fluidez
cwd = 'D://Franco//Doctorado//Laboratorio//NLP' # path Franco escritorio
# cwd = 'C://Franco//NLP' # path Franco Udesa

pickle_traduccion = '//Scripts//traducciones.pkl'
df_pacientes = pd.ExcelFile(cwd+r'\Bases\Transcripciones fluidez.xlsx')
df_pacientes = pd.read_excel(df_pacientes, 'Hoja 1')
df_pacientes = df_pacientes.drop(df_pacientes[df_pacientes["Estado"] != "Completo"].index)

nlp_class = NLPClass.NLPClass()
try:
    df_translate = pd.read_pickle(cwd+pickle_traduccion)
except (OSError, IOError):
    df_translate = pd.DataFrame(columns=['spanish','english'])
    
df_translate = nlp_class.translations_dictionary(df_translate, path = cwd+pickle_traduccion)

lista_columnas_concatenar = ['fluency_animals_0_15_correctas_individuales',
                             'fluency_animals_15_30_correctas_individuales',
                             'fluency_animals_30_45_correctas_individuales',
                             'fluency_animals_45_60_correctas_individuales']

# Reemplazo los Nans por espacios vacíos y concateno horizontalmente las columnas anteriores.
df_pacientes[lista_columnas_concatenar] = df_pacientes[lista_columnas_concatenar].fillna('')
resultado = nlp_class.join_horizontally_strings(df_pacientes, lista_columnas_concatenar," ")

# Tokenizo las filas resultantes
lista_tokenizada = nlp_class.tokenize_list(resultado)
# Obtengo los tokens únicos y la cantidad de veces que aparece cada uno.
unique_words, count_words = nlp_class.count_words(lista_tokenizada,0.8) 
# Obtengo un dataframe donde la primer columna tiene las palabras originales y la segunda columna las palabras traducidas. Esta función se asegura que 
# cada traducción tenga un synset en WordNet. En caso que no exista, devuelve "no_translation"
df_translate = nlp_class.translate_checking_wordnet_and_hypernym(unique_words, df_translate,hypernym_check = "animal.n.01",len_src ="spanish", len_dest="english")



number_nodes = []
for i,row in df_translate.iterrows():
    number_nodes.append(nlp_class.hypernym_min_nodes_distance_from_synset_to_hypernym(row['english'], hypernym_check = "animal.n.01"))
    
df_translate['nodes'] = number_nodes
       

df_translate.to_pickle(cwd+pickle_traduccion)






# %% Download and load fasttext model

import fasttext.util

######### First time only ###########
# fasttext.util.download_model('es', if_exists='ignore')  # English
ft = fasttext.load_model('cc.es.300.bin')
# ft.save_model('cc.es.300.bin') # First time only

#%% Convierto a word embedding cada palabra dicha por cada paciente.

words_vector = list()
for i_lista,word_list in enumerate(lista_tokenizada):
    words_vector.append([])
    for i_word,word in enumerate(word_list):
        words_vector[i_lista].append(ft.get_word_vector(word))

# %% Calculo la distancia semántica entre cada palabra contigua dicha por cada paciente.
words_distances = list()
for i_lista, vector_list in enumerate(words_vector):
    words_distances.append([])
    for i_vector,vector in enumerate(vector_list):
        if(i_vector!=0):
            resultado = nlp_class.words_distance(vector_list[i_vector-1],vector_list[i_vector])
            words_distances[i_lista].append(resultado)
        
# %% Calculo la Ongoing Semantic Variability de cada paciente.

ongoing_semantic = list()
for words in words_distances:
    ongoing_semantic.append(nlp_class.ongoing_semantic_variability(words))