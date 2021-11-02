# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 15:43:43 2021

@author: franc
"""

import nltk
import os
import pandas as pd
import NLPClass

# Levanto la base de fluidez
cwd = 'D://Franco//Doctorado//Laboratorio//BasesFluidezFonoYSeman//TranscripcionesChi' # path Franco escritorio
# cwd = 'C://Franco//NLP' # path Franco Udesa

df_pacientes = pd.ExcelFile(cwd+r'\transcripciones_fluidez-flor_alifano_fas.xlsx')
df_pacientes = pd.read_excel(df_pacientes, 'Hoja 1')

lista_columnas_concatenar = ['fluency_a_0_15_todo',
                             'fluency_a_15_30_todo',
                             'fluency_a_30_45_todo',
                             'fluency_a_45_60_todo']

nlp_class = NLPClass.NLPClass()

# Reemplazo los Nans por espacios vacíos y concateno horizontalmente las columnas anteriores.
df_pacientes[lista_columnas_concatenar] = df_pacientes[lista_columnas_concatenar].fillna('')
resultado = nlp_class.join_horizontally_strings(df_pacientes, lista_columnas_concatenar," ")

# Tokenizo las filas resultantes
lista_tokenizada = nlp_class.tokenize_list(resultado)

# Obtengo los tokens únicos y la cantidad de veces que aparece cada uno.
unique_words, count_words = nlp_class.count_words(lista_tokenizada,0.8) 
