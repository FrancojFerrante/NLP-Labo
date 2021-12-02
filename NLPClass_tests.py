# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 13:18:30 2021

@author: franc
"""

import pandas as pd
import NLPClass

###########################----------------------###########################
# Testeo los métodos


# Levanto la base de fluidez
cwd = 'D://Franco//Doctorado//Laboratorio//NLP' # path Franco escritorio
cwd = 'D://Franco//Doctorado//Laboratorio//BasesFluidezFonoYSeman//basesdedatos'

pickle_traduccion = '//Scripts//traducciones.pkl'
df_pacientes = pd.ExcelFile(cwd+r'\transcripciones_fluidez.xlsx')
df_pacientes = pd.read_excel(df_pacientes, 'Hoja 1')
df_pacientes = df_pacientes.drop(df_pacientes[df_pacientes["Estado"] != "Completo"].index)

nlp_class = NLPClass.NLPClass()
try:
    df_translate = pd.read_pickle(cwd+pickle_traduccion)
except (OSError, IOError):
    df_translate = pd.DataFrame(columns=['spanish','english'])

# df_translate = nlp_class.translations_dictionary(df_translate, path = cwd+pickle_traduccion)

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

# %%

lista_tokenizada_nodos = []
contador =-1
for elemento in lista_tokenizada:
    contador +=1
    lista_tokenizada_nodos.append([])
    for palabra in elemento:
        lista_tokenizada_nodos[contador].append(df_translate[df_translate['spanish'] == palabra]['nodes'].values[0])
        
cantidad_palabras = 0
cantidad_palabras_no_encontradas = 0
for elemento in lista_tokenizada_nodos:
    for nodo in elemento:
        cantidad_palabras+=1
        if nodo==-1:
            cantidad_palabras_no_encontradas+=1
# %%

# number_nodes = []
# for i,row in df_translate.iterrows():
#     number_nodes.append(nlp_class.hypernym_min_nodes_distance_from_synset_to_hypernym(row['spanish'], hypernym_check = "animal.n.01"))

# df_translate['nodes_spanish'] = number_nodes


# %% Download and load fasttext model


# ongoing_semantic_list = nlp_class.ongoing_semantic_variability_complete(lista_tokenizada)


# %%

import NLPClass
nlp_class = NLPClass.NLPClass()
dict_translation = nlp_class.add_to_pickle_translation_file("D://Franco//Doctorado//Laboratorio//BasesFluidezFonoYSeman//Scripts",["palabra","vaso"],lan_src = "spanish",lan_dest = "english")