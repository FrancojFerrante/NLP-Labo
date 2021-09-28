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

from googletrans import Translator
# pip install googletrans==4.0.0rc1

import pickle
# pip install pickle-mixin

from nltk.corpus import wordnet as wn

class NLPClass:
    def __init__(self):
        self.numero = 1
        
    def translate_dictionary(self, df_translate=None, path=""):
        
        if df_translate is None:
            df_translate = pd.DataFrame(columns=['spanish','english'])
        
        df_translate = df_translate.append(pd.DataFrame({'spanish': ["yaguareté"], 'english': ["jaguar"]}), ignore_index = True)
        df_translate = df_translate.append(pd.DataFrame({'spanish': ["llama"], 'english': ["llama"]}), ignore_index = True)
        df_translate = df_translate.append(pd.DataFrame({'spanish': ["picaflor"], 'english': ["hummingbird"]}), ignore_index = True)
        df_translate = df_translate.append(pd.DataFrame({'spanish': ["chita"], 'english': ["cheetah"]}), ignore_index = True)
        df_translate = df_translate.append(pd.DataFrame({'spanish': ["torcaza"], 'english': ["dove"]}), ignore_index = True)
        df_translate = df_translate.append(pd.DataFrame({'spanish': ["yacaré"], 'english': ["alligator"]}), ignore_index = True)
        df_translate = df_translate.append(pd.DataFrame({'spanish': ["corvina"], 'english': ["croaker"]}), ignore_index = True)
        df_translate = df_translate.append(pd.DataFrame({'spanish': ["vizcacha"], 'english': ["viscacha"]}), ignore_index = True)
        df_translate = df_translate.append(pd.DataFrame({'spanish': ["orca"], 'english': ["killer_whale"]}), ignore_index = True)
        df_translate = df_translate.append(pd.DataFrame({'spanish': ["barata"], 'english': ["german_cockroach"]}), ignore_index = True)
        df_translate = df_translate.append(pd.DataFrame({'spanish': ["coipo"], 'english': ["coypu"]}), ignore_index = True)
        df_translate = df_translate.append(pd.DataFrame({'spanish': ["cuncuna"], 'english': ["caterpillar"]}), ignore_index = True)
        df_translate = df_translate.append(pd.DataFrame({'spanish': ["carpincho"], 'english': ["capybara"]}), ignore_index = True)
        df_translate = df_translate.append(pd.DataFrame({'spanish': ["jote"], 'english': ["buzzard"]}), ignore_index = True)
        df_translate = df_translate.append(pd.DataFrame({'spanish': ["zorzal"], 'english': ["fieldfare"]}), ignore_index = True)
        df_translate = df_translate.append(pd.DataFrame({'spanish': ["guanaco"], 'english': ["guanaco"]}), ignore_index = True)
        df_translate = df_translate.append(pd.DataFrame({'spanish': ["pejerrey"], 'english': ["silverside"]}), ignore_index = True)
        df_translate = df_translate.append(pd.DataFrame({'spanish': ["mandril"], 'english': ["mandrill"]}), ignore_index = True)
        df_translate = df_translate.append(pd.DataFrame({'spanish': ["peludo"], 'english': ["armadillo"]}), ignore_index = True)
        df_translate = df_translate.append(pd.DataFrame({'spanish': ["chingue"], 'english': ["skunk"]}), ignore_index = True)
        df_translate = df_translate.append(pd.DataFrame({'spanish': ["guaren"], 'english': ["brown_rat"]}), ignore_index = True)


        


        if (path != ""):
            df_translate.to_pickle(path)

        return df_translate

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
              accumulated += list(sorted_words.values())[counter]
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
    
    def translate(self, text, lan_src = 'es', lan_dest = 'en'):
        
        translator = Translator()
        text_translate = []
        for element in text:
            text_translate.append(translator.translate(element.replace("-"," "), src=lan_src, dest=lan_dest))
        return text_translate
    
    def get_pickle_and_translate(self,texts,path = ""):
        if path != "":
            try:
                df_translate = pd.read_pickle(path)
            except (OSError, IOError) as e:
                df_translate = pd.DataFrame(columns=['spanish','english'])
        else:
            df_translate = pd.DataFrame(columns=['spanish','english'])
        
        hyper = lambda s: s.hypernyms()
        for text in texts:
            
            if text not in df_translate['spanish'].to_list():
                try:
                    es_animal = False
                    iter_traducciones = -1
                    traduccion = nlp_class.translate([text])
                    while (not es_animal):
                        palabras = []
                        while (len(palabras)==0):
                            iter_traducciones+=1
                            palabra_traducida = traduccion[0].extra_data["parsed"][1][0][0][5][0][4][iter_traducciones][0].lower()
                            palabras = wn.synsets(palabra_traducida)
                            palabras = [x for x in palabras if (".n.") in x.name().lower()]
                        iterador = -1
                        for i in range(0,len(palabras)):
                            # list(palabras[iterador].closure(hyper, depth=1)) == palabras[iterador].hypernyms()
                            hiperonimos = list(palabras[i].closure(hyper))
                            for hiperonimo in hiperonimos:    
                                if ("animal.n.01" == hiperonimo.name()):
                                    es_animal = True
                                    break
                            if es_animal:
                                break
    
                except:
                    df2 = pd.DataFrame({'spanish': [text],'english': "no_hay_traduccion"})
                    df_translate = df_translate.append(df2, ignore_index = True)

                else:
                    df2 = pd.DataFrame({'spanish': [text],'english': palabra_traducida})
                    df_translate = df_translate.append(df2, ignore_index = True)

        # print(text + str(list(hiperonimos)))
                
    
                
                
                
                
                
                # hyper = lambda s: s.hypernyms()
                # for i,row in df_translate.iterrows():
                #     palabras = wn.synsets(row['english'])
                #     palabras = [x for x in palabras if ".n." in x.name()]
                #     if (len(palabras)==0):
                #         translator = Translator()
                #         respuesta = translator.translate(row['spanish'], src="es", dest="en")
                #         palabras = wn.synsets(respuesta.extra_data["parsed"][1][0][0][5][0][4][1][0])
                #         palabras = [x for x in palabras if ".n." in x.name()]
                #     es_animal = False
                #     iterador = -1
                #     while not es_animal:
                #         # list(palabras[iterador].closure(hyper, depth=1)) == palabras[iterador].hypernyms()
                #         iterador +=1
                #         hiperonimos = list(palabras[iterador].closure(hyper))
                #         for hiperonimo in hiperonimos:    
                #             if ("animal.n.01" == hiperonimo.name()):
                #                 es_animal = True
                #     print(row['english'] + str(list(palabras[iterador].closure(hyper))))

                
                
                
                
                
                
                
                
                
                
                
                
        return df_translate



# Testeo los métodos
# cwd = 'D://Franco//Doctorado//Laboratorio//NLP' # path Franco escritorio
cwd = 'C://Franco//NLP' # path Franco Udesa
pickle_traduccion = '//Scripts//traduccionesasasa.pkl'
df_pacientes = pd.ExcelFile(cwd+r'\Bases\Transcripciones fluidez.xlsx')


df_pacientes = pd.read_excel(df_pacientes, 'Hoja 1')
nlp_class = NLPClass()

resul = nlp_class.translate_dictionary(path=cwd+pickle_traduccion)

df_pacientes['fluency_animals_0_15_correctas_individuales'].fillna('', inplace=True)
df_pacientes['fluency_animals_15_30_correctas_individuales'].fillna('', inplace=True)
df_pacientes['fluency_animals_30_45_correctas_individuales'].fillna('', inplace=True)
df_pacientes['fluency_animals_45_60_correctas_individuales'].fillna('', inplace=True)

resultado = nlp_class.join_horizontally_strings(df_pacientes, " ", 'fluency_animals_0_15_correctas_individuales',
                            'fluency_animals_15_30_correctas_individuales',
                            'fluency_animals_30_45_correctas_individuales',
                            'fluency_animals_45_60_correctas_individuales')

lista_tokenizada = nlp_class.tokenize_list(resultado)
unique_words, count_words = nlp_class.count_words(lista_tokenizada,0.8)
df_translate = nlp_class.get_pickle_and_translate(unique_words,cwd+pickle_traduccion)
# df_translate = nlp_class.get_pickle_and_translate(unique_words[34:40])


# hyper = lambda s: s.hypernyms()
# for i,row in df_translate.iterrows():
#     palabras = wn.synsets(row['english'])
#     palabras = [x for x in palabras if ".n." in x.name()]
#     if (len(palabras)==0):
#         translator = Translator()
#         respuesta = translator.translate(row['spanish'], src="es", dest="en")
#         palabras = wn.synsets(respuesta.extra_data["parsed"][1][0][0][5][0][4][1][0])
#         palabras = [x for x in palabras if ".n." in x.name()]
#     es_animal = False
#     iterador = -1
#     while not es_animal:
#         # list(palabras[iterador].closure(hyper, depth=1)) == palabras[iterador].hypernyms()
#         iterador +=1
#         hiperonimos = list(palabras[iterador].closure(hyper))
#         for hiperonimo in hiperonimos:    
#             if ("animal.n.01" == hiperonimo.name()):
#                 es_animal = True
#     print(row['english'] + str(list(palabras[iterador].closure(hyper))))



df_translate.to_pickle(cwd+pickle_traduccion)


