# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 10:59:05 2021

@author: franc
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter, OrderedDict
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import torchtext
from torchtext.data import get_tokenizer

from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

class NLPClass:
    def __init__(self):
        self.numero = 1
        
    
    def tokenize_list(self, text_dataframe, tokenizer_type = "basic_english"):
      '''
      Recibe una lista de strings y devuelve 

      Parameters
      ----------
      text_dataframe : list
          Una lista de strings donde cada elemento tiene palabras separadas 
          por espacios.
          
      tokenizer_type : el tipo de tokenizador a utilizar, si no se especifica,
      usa basic_english. Otros tipos podrían ser: spacy, moses, toktok,revtok,
      subword.

      Returns
      -------
      tokens : list
          Una lista donde cada elemento es una lista que contiene un token.

      '''
      tokenizer = get_tokenizer(tokenizer_type)
      tokens = [tokenizer(s) for s in text_dataframe]
      return tokens
  
    def count_words(self, tokens, percentaje = 0):
        '''
        Devuelve una lista de palabras únicas extraídas del parámetro tokens y
        la cantidad de veces que se repite cada una ordenadas descendientemente.

        Parameters
        ----------
        tokens : list of token list
            Una lista donde cada elemento es una lista de tokens.
        
        percentaje: int
            Entero entre 0 y 1. 
            Si es 0, no imprime nada. 
            Si es mayor a 0, imprime por consola qué cantidad de palabras más
            comunes (en porcentaje) superan el percentaje pasado por parámetro.
            
        Returns
        -------
        La lista única de palabras y la cantidad de veces que aparece cada una.

        '''
        words = Counter()
        for s in tokens:
            for w in s:
                words[w] += 1
                
        sorted_words = OrderedDict(words.most_common())
        # sorted_words = list(words.keys())
        # sorted_words.sort(key=lambda w: words[w], reverse=True)
        
        if (percentaje>0):
            count_occurences = sum(words.values())

            accumulated = 0
            counter = 0
            
            while accumulated < count_occurences * percentaje:
              accumulated += sorted_words.values()[counter]
              counter += 1
            
            print(f"The {counter * 100 / len(words)}% most common words "
                  f"account for the {accumulated * 100 / count_occurences}% of the occurrences")
                    
                
        return list(sorted_words.keys()),list(sorted_words.values())
        
    def join_horizontally_strings(self, df, separator = " ", *args):
        '''
        Toma cada fila del dataframe df, y une el contenido de cada columna
        pasada en *args separada por el parámetro separator

        Parameters
        ----------
        df : pandas.dataframe
            El dataframe que contiene las columnas a ser unidas.
        *args : string
            Los nombres de las columnas que quieren ser unidos.

        Returns
        -------
        lista : list of strings
            Una lista donde en cada fila tiene la unión entre las distintas
            columnas pasadas en *args separadas por separator.

        '''
        lista=[]
        for i, row in df.iterrows():
            lista.append("")
            for column in args:
                lista[i] = lista[i] + row[column] + separator
            lista[i] = lista[i].rstrip()
                
        return lista
    
    def plot_word_distribution(self,count_words,min_repetition=1):
        
        lista = [w for w in count_words if w>min_repetition]
        plt.bar(range(len(lista)), lista)
        plt.show()

# Testeo los métodos
cwd = 'D://Franco//Doctorado//Laboratorio//NLP'
df_pacientes = pd.ExcelFile(cwd+r'\Bases\Transcripciones fluidez.xlsx')
df_pacientes = pd.read_excel(df_pacientes, 'Hoja 1')
nlp_class = NLPClass()
df_pacientes['fluency_animals_0_15_correctas_individuales'].fillna('', inplace=True)
df_pacientes['fluency_animals_15_30_correctas_individuales'].fillna('', inplace=True)
df_pacientes['fluency_animals_30_45_correctas_individuales'].fillna('', inplace=True)
df_pacientes['fluency_animals_45_60_correctas_individuales'].fillna('', inplace=True)

resultado = nlp_class.join_horizontally_strings(df_pacientes, " ", 'fluency_animals_0_15_correctas_individuales',
                            'fluency_animals_15_30_correctas_individuales',
                            'fluency_animals_30_45_correctas_individuales',
                            'fluency_animals_45_60_correctas_individuales')

lista_tokenizada = nlp_class.tokenize_list(resultado)
unique_words, count_words = nlp_class.count_words(lista_tokenizada,True)
nlp_class.percentaje_common_words(unique_words, count_words,0.8)
nlp_class.plot_word_distribution(count_words,3)



