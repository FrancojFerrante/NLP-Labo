# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 10:59:05 2021

@author: franc
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import Counter, OrderedDict

import math

import torchtext
from torchtext.data import get_tokenizer

from googletrans import Translator
# from deep_translator import GoogleTranslator
# pip install googletrans==4.0.0rc1

import pickle
# pip install pickle-mixin

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

# python -m spacy download es_core_news_sm
import spacy

import fasttext.util

import contractions

import re      # libreria de expresiones regulares
import string   # libreria de cadena de caracteres

import itertools

import sys
sys.path.append("/tmp/TEST")

from treetagger import TreeTagger

import pathlib

from scipy.spatial import distance

class NLPClass:
    def __init__(self):
        self.numero = 1
        nltk.download('wordnet')
    def translations_dictionary(self, df_translate=None, path=""):
        '''
        It appends to a dictionary different animals names in spanish and
        english languages. It adds them so that english animals names appear
        in WordNet synset.

        Parameters
        ----------
        df_translate : pandas.dataframe, optional.
                If it's not None, the rows are appended. Otherwise it's
                initialized and then the rows are appended.
                The default is None.
        path : string, optional
            The path where to save the pickle file with the dictionary. Unless
            path is empty.
            The default is "".

        Returns
        -------
        df_translate : pandas.dataframe.
            Pandas.dataframe with the new rows appended.

        '''

        df_auxiliar = pd.DataFrame(columns=['spanish','english'])
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["yaguareté"], 'english': ["jaguar"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["llama"], 'english': ["llama"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["picaflor"], 'english': ["hummingbird"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["chita"], 'english': ["cheetah"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["torcaza"], 'english': ["dove"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["yacaré"], 'english': ["alligator"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["corvina"], 'english': ["croaker"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["vizcacha"], 'english': ["viscacha"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["orca"], 'english': ["killer_whale"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["barata"], 'english': ["german_cockroach"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["coipo"], 'english': ["coypu"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["cuncuna"], 'english': ["caterpillar"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["carpincho"], 'english': ["capybara"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["jote"], 'english': ["buzzard"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["zorzal"], 'english': ["fieldfare"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["guanaco"], 'english': ["guanaco"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["pejerrey"], 'english': ["silverside"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["mandril"], 'english': ["mandrill"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["peludo"], 'english': ["armadillo"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["chingue"], 'english': ["skunk"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["guaren"], 'english': ["brown_rat"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["cata"], 'english': ["budgerigar"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["bonito"], 'english': ["atlantic_bonito"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["cachalote"], 'english': ["sperm_whale"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["morena"], 'english': ["moray_eels"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["jaiba"], 'english': ["callinectes_sapidus"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["cervatillo"], 'english': ["fawn"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["mulita"], 'english': ["nine-banded_armadillo"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["carpintero"], 'english': ["woodpecker"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["centolla"], 'english': ["maja_squinado"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["palometa"], 'english': ["pomfret"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["suricata"], 'english': ["meerkat"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["vampiro"], 'english': ["vampire_bats"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["laucha"], 'english': ["mouse"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["guanaco"], 'english': ["guanaco"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["vicuña"], 'english': ["vicuna"]}), ignore_index = True)
        df_auxiliar = df_auxiliar.append(pd.DataFrame({'spanish': ["carancho"], 'english': ["caracara"]}), ignore_index = True)

        if df_translate is None:
            df_translate = df_auxiliar.copy(deep=True)
        else:
            for i,row in df_auxiliar.iterrows():
                if row['spanish'] not in df_translate['spanish'].values:
                    df_translate = df_translate.append(row)

            

            # df_translate = pd.concat([df_translate, df_auxiliar.ix[df_auxiliar._merge=='left_only', ['spanish']]])
          
        if (path != ""):
            df_translate.to_pickle(path)

        return df_translate

    def tokenize_list(self, text_dataframe, tokenizer_type = "basic_english"):
      '''
      It receives a list of strings and returns a list of string list where
      each string is a token obtained from apply the tokenizer_type.

      Parameters
      ----------
      text_dataframe : string list
          A string list where each element has words separated by spaces.
    
      tokenizer_type : 
          The kind of tokenizer to be applied. Basic_english applied by 
          default. Other tokenizers could be: spacy, moses, toktok, revtok, 
          subword.

      Returns
      -------
      tokens : list of string list
          A list where each element is a list that contains tokens.
      '''
      
      tokenizer = get_tokenizer(tokenizer_type)
      tokens = [tokenizer(x) if str(x)!="nan" else x for x in text_dataframe]
      return tokens
  
    def count_words(self, tokens, percentaje = 0):
        '''
        It returns a word unique list extracted from tokens parameter and the
        number of times that each word appear in descendingly ordered.

        Parameters
        ----------
        tokens : list of token list
            A list where each element is a token list.
        
        percentaje: int
            Int between 0 and 1.
            If it is 0, it doesn't print anything. 
            If it is greater than 0, it prints by console how many of 
            the most common words (in percentage) exceed the percentage passed 
            by parameter.
            
        Returns
        -------
        The word unique list and the number of times that each one appear.

        '''
        
        words = Counter()
        for s in tokens:
            if str(s) != "nan":
                for w in s:
                    words[w] += 1
                
        sorted_words = OrderedDict(words.most_common())
        
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
        
    def join_horizontally_strings(self, df, column_list, separator = " "):
        '''
        It takes each df dataframe row and joins the content of each column 
        passed in *args separated by the separator parameter.

        Parameters
        ----------
        df : pandas.dataframe
            Dataframe that contains the columns to join.
        *args : string
            The columns names to be joined.

        Returns
        -------
        lista : list of strings
            A list where each row has the union between the different columns 
            passed in * args separated by separator.

        '''
        
        lista=[]
        contador = 0
        for i, row in df.iterrows():
            lista.append("")
            for column in column_list:
                lista[contador] = lista[contador] + row[column] + separator
            lista[contador] = lista[contador].rstrip()
            contador+=1
        return lista
    
    def add_to_pickle_translation_file(self,path,words,lan_src = "spanish",lan_dest = "english"):
        
        df_translation = self.read_pickle_translation_file(path)
        for i,word in enumerate(words):
            df_check = df_translation[(df_translation.word == word) & (df_translation.lan_src == lan_src) & (df_translation.lan_dest == lan_dest)]
            if len(df_check.index) == 0:
                print("Traduciendo: " + str(i) + "/" + str(len(words)))
                new_row = [word,self.translate([word],lan_src,lan_dest)[0].extra_data["parsed"],lan_src,lan_dest]
                df_length = len(df_translation)
                df_translation.loc[df_length] = new_row
        df_translation.to_pickle(path+"//translations.pkl")

    
    def read_pickle_translation_file(self,path):
        try:
            df_translation = pd.read_pickle(path+"//translations.pkl")
        except (OSError, IOError):
            print("translation.pkl no encontrado")
            df_translation = pd.DataFrame(columns=['word','translation','lan_src','lan_dest'])
        return df_translation
        
    
    def translate(self, text, lan_src = 'spanish', lan_dest = 'english'):
        '''
        It translates text from one language to another using googletrans.

        Parameters
        ----------
        text : string list
            Strings to be translated.
        lan_src : string, optional.
            The language source. 
            The default is 'es'.
        lan_dest : string, optional
            The language destiny. 
            The default is 'en'.

        Returns
        -------
        text_translate : translated_object list
            A list where each element is a translation from each text list 
            element.

        '''
        translator = Translator()
        translated_objects = []
        for element in text:
            translated_objects.append(translator.translate(element, src=lan_src, dest=lan_dest))
        return translated_objects
    def translate_checking_wordnet_and_hypernym(self, texts, df_translate = None, hypernym_check = '', len_src = 'spanish', len_dest = 'english'):
        '''
        It receives a word list in len_src language and returns a dataframe
        with the original word list and its len_dest translation. If the
        original word doesn't have a translation that exists on WordNet synset
        or the hypernym_check on the hypernym tree, it returns
        "no_translation".
        Parameters
        ----------
        texts : string list
            list with words to translate.
        df_translate : pandas.dataframe, optional
            A dataframe with two columns: len_src with words in len_src language
            and len_dest with words in len_dest language. If it's not None,
            the rows are appended. Otherwise it's initialized and then the
            rows are appended.
            The default is None
        hypernym_check : string, optional
            The synset to be checked if exists on translated hypernym tree.
            The default is "".
        len_src : string, optional
            Language source.
            The default is "spanish".
        len_dest : string, optional
            Language destiny.
            he default is "english".

        Returns
        -------
        df_translate : pandas.dataframe
            Dataframe with two columns: len_src with words in len_src language
            and len_dest with words in len_dest language.
        '''
        
        # If df_translate not exist, initialize it
        if df_translate is None:
            df_translate = pd.DataFrame(columns=[len_src,len_dest])
                           
        for text in texts:
            if text not in df_translate[len_src].to_list():
                try:
                    has_hyper = False
                    iter_translates = -1
                    translation_object = self.translate([text]) # Get the translation_object with all posible translations
                    while (not has_hyper):
                        translated_synsets = []
                        while (len(translated_synsets)==0):
                            iter_translates+=1
                            translated_word = translation_object[0].extra_data["parsed"][1][0][0][5][0][4][iter_translates][0].lower() # Extract a posible translation
                            translated_synsets = wn.synsets(translated_word.replace(" ","_"),pos=wn.NOUN)
                            translated_synsets = [x for x in translated_synsets if ".n." in x.name().lower()] # keep nouns only
                        if hypernym_check != '':
                            synset_with_hypernym, _ = self.get_synset_that_has_hypernym(translated_synsets, hypernym_check = hypernym_check) # check if hypernym_check is part of translated_synsets hypernym tree
                            if len(synset_with_hypernym)>0:
                                has_hyper = True
                        else:
                            has_hyper = True
                except:
                    df2 = pd.DataFrame({len_src: [text],len_dest: "no_translation"})
                    df_translate = df_translate.append(df2, ignore_index = True)

                else:
                    # Add translation to dictonary
                    df2 = pd.DataFrame({len_src: [text],len_dest: [translated_word]})
                    df_translate = df_translate.append(df2, ignore_index = True)
        return df_translate

    def get_hypernyms_to(self, synset, hypernym_destiny = "animal.n.01"):
        """
        Check if the synset's hypernyms tree contains the hypernym_destiny

        Parameters
        ----------
        synset : WordNet.synset
            Synset to be checked.
        hypernym_destiny : string, optional
            Synset key to be searched in the synset's hypernyms tree.
            The default is "animal.n.01".

        Returns
        -------
        hypernyms_to_destiny : synset list
            A synset list from the original synset to the hypernym_destiny.

        """
        
        # [f(x) if condition else g(x) for x in sequence]
        # And, for list comprehensions with if conditions only,
        
        # [f(x) for x in sequence if condition]
        

        hypernym_destiny = wn.synset(hypernym_destiny)
        total_hypernyms = synset.hypernym_paths()
        hypernyms_to_destiny = None
        hypernyms_to_destiny = [list(itertools.takewhile(lambda ele: ele != hypernym_destiny, x[::-1])) + [hypernym_destiny]  for x in total_hypernyms if (hypernym_destiny in x)]
        if (hypernyms_to_destiny==0):
            hypernyms_to_destiny = None
        return hypernyms_to_destiny

    def get_synset_that_has_hypernym(self, synsets, hypernym_check = "animal.n.01"):
        """
        It receives a list of synsets and return all the synsets that has the
        hypernym_check in its hypernyms tree and the synsets's hypernyms

        Parameters
        ----------
        synsets : synset list
            A list with synsets.
        hypernym_check : string, optional
            The synset to be searched in the hypernyms tree.
            The default is "animal.n.01".

        Returns
        -------
        synsets_with_hyper : synset
            All the synset whose hypernyms tree contains the hypernym_check.
        synsets_hypernyms : synset list
            The synsets_with_hyper's hypernyms tree.

        """
        synsets_with_hyper = []
        synsets_hypernyms = []
        for i in range(0,len(synsets)):
            hypernyms = self.get_hypernyms_to(synsets[i], hypernym_destiny = hypernym_check)
            if (hypernyms is not None):
                synsets_with_hyper.append(synsets[i])
                synsets_hypernyms.append(hypernyms)
        return synsets_with_hyper, synsets_hypernyms

    def hypernym_min_nodes_distance_from_synset_to_hypernym(self, word, hypernym_check = "animal.n.01"):
        """
        It calculates the number of nodes from the word synset to the hypernym_check
        in the hypernyms tree. Get the shortest path

        Parameters
        ----------
        word : string
            Word from starting node.
        hypernym_check : string, optional
            Synset key to be searched in the hypernyms tree. 
            The default is "animal.n.01".

        Returns
        -------
        int
            Number of nodes from word to hypernym_check.

        """
        
        translated_synsets = wn.synsets(word.replace(" ","_"),pos=wn.NOUN)
        synsets_with_hypernym, synsets_hypernyms = self.get_synset_that_has_hypernym(translated_synsets, hypernym_check = hypernym_check)
        if len(synsets_with_hypernym)>0:
            aux=[]
            for n in synsets_hypernyms:
                for m in n:
                    aux.append(len(m))
            if aux!=[]:
                return (np.min(aux))
        else:
            return -1
        
    def min_nodes_distance_all_words(self,words,hypernym_check= "synset.n.01"):
        minima = []
        for word in words:
            word_min = self.hypernym_min_nodes_distance_from_synset_to_hypernym(word,hypernym_check)
            if word_min != -1:
                minima.append(word_min)
        if minima!=[]:
            return np.min(minima)
        else:
            return -1
    
    def semantic_words_distance(self, u,v):
        """
        Given two vectors, it calculates the semantic word distance as the 
        cosine of the angle between the two vectors as (Sanz et al. 2021)

        Parameters
        ----------
        u : float list
            A word embedding.
        v : float list
            A word embedding.

        Returns
        -------
        float
            The semantic word distance between u and v.

        """
        return 1-(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)))
    
    def ongoing_semantic_variability(self, vector_distances):
        """
        Given a vector, it calculates the ongoing semantic variability as defined at
        (Sanz et al. 2021)

        Parameters
        ----------
        vector_distances : float list
            A list where each float element represents the semantic distance 
            between two words.

        Returns
        -------
        average : float list
            A list where each float element represents the ongoing semantic
            variability.

        """
        if (len(vector_distances) < 2):
            return np.nan
        summation = sum((vector_distances-np.mean(vector_distances))*(vector_distances-np.mean(vector_distances)))
        average = summation/(len(vector_distances))
        return average
    
    def cross_feature_variability(self,vector_words,concept):
        
        # Convierto a word embedding cada palabra dicha por cada paciente.
        words_vector = self.get_word_fast_text_vector(vector_words)

                
        # Convierto a word embedding el concepto.
        concept_vector = self.get_word_fast_text_vector(concept)[0][0]

        
        # Calculo la distancia semántica entre cada palabra y el concepto
        words_distances = self.cross_semantic_distance(words_vector,concept_vector)
                
        # Calculo la cross_feature_variability de cada paciente.
        
        ongoing_semantic_list = list()
        for words in words_distances:
            ongoing_semantic_list.append(self.ongoing_semantic_variability(words))

        return ongoing_semantic_list
    
    def cross_semantic_distance(self,words_vector,concept_vector):
        # Calculo la distancia semántica entre cada palabra y el concepto
        words_distances = list()
        for i_lista, vector_list in enumerate(words_vector):
            words_distances.append([])
            for i_vector,vector in enumerate(vector_list):
                resultado = self.semantic_words_distance(vector_list[i_vector],concept_vector)
                words_distances[i_lista].append(resultado)
        return words_distances
                
                
    def download_and_save_fast_text_model(self,save = True):
        fasttext.util.download_model('es', if_exists='ignore')  # English
        ft = fasttext.load_model('cc.es.300.bin')
        if save:
            ft.save_model('cc.es.300.bin') # First time only
        self.fasttext = ft
        return ft
    
    def load_fast_text_model(self,path_fast_text):
        self.fasttext = fasttext.load_model(path_fast_text+'cc.es.300.bin')
        return self.fasttext
    
    def get_word_fast_text_vector(self,vector_words):
        
        words_vector = list()
        for i_lista,word_list in enumerate(vector_words):
            words_vector.append([])
            for i_word,word in enumerate(word_list):
                words_vector[i_lista].append(self.fasttext.get_word_vector(word))
                
        return words_vector
    
    def ongoing_semantic_distance(self,words_vector):
                                  
        words_distances = list()
        for i_lista, vector_list in enumerate(words_vector):
            words_distances.append([])
            for i_vector,vector in enumerate(vector_list):
                if(i_vector!=0):
                    resultado = self.semantic_words_distance(vector_list[i_vector-1],vector_list[i_vector])
                    words_distances[i_lista].append(resultado)
        return words_distances
    
    def ongoing_semantic_variability_complete(self, vector_words):
        """
        Given a list of strings list, it calculates the ongoing semantic 
        variability for each element as defined at (Sanz et al. 2021)

        Parameters
        ----------
        vector_words : list of strings list
            A list where each element has a list of words.

        Returns
        -------
        ongoing_semantic_list : float list
            A list where each float element represents the ongoing semantic
            variability.

        """

        # Obtengo el vector de fasttext para cada palabra de cada paciente
        words_vector = self.get_word_fast_text_vector(vector_words)
        
        
        # Calculo la distancia semántica entre cada palabra (vector) contigua dicha por cada paciente.
        words_distances = self.ongoing_semantic_distance(words_vector)

                
        # Calculo la Ongoing Semantic Variability de cada paciente.
            
        ongoing_semantic_list = list()
        for words in words_distances:
            ongoing_semantic_list.append(self.ongoing_semantic_variability(words))

        return ongoing_semantic_list
        

    def expand_contractions_dataframe(self, token_list):
      
        # Defino una funcion anonima que al pasarle un argumento devuelve el resultado de aplicarle la funcion anterior a este mismo argumento
        round0 = lambda x: contractions.fix(x)
        
        # Dataframe que resulta de aplicarle a las columnas la funcion de limpieza
        token_expanded = token_list.apply(round0)
        
        return token_expanded
    
    def expand_contractions(self, token):
      
        # Expando el token
        token_expanded = contractions.fix(token)
              
        return token_expanded
    
    # Defino una funcion que recibe un texto y devuelve el mismo texto sin signos,
    def clean_text_paula(self, text):
        
        # pasa las mayusculas del texto a minusculas
        text = text.lower()                                              
        # reemplaza texto entre corchetes por espacio en blanco.. ¿ y \% no se..
        text = " ".join(re.findall('(?<!\S)[a-z_]+(?=[,.!?:;]?(?!\S))', text))
                     
        # reemplaza texto entre corchetes por espacio en blanco.. ¿ y \% no se..
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text) 

        # # reemplaza signos de puntuacion por espacio en blanco.. %s -> \S+ es cualquier caracter que no sea un espacio en blanco
        # text = re.sub('[%s]' % re.escape(string.punctuation), '', text) 
        # # remueve palabras que contienen numeros.
        # text = re.sub('\w*\d\w*', '', text)       
        # # Sacamos comillas, los puntos suspensivos, <<, >>
        # text = re.sub('[‘’“”…«»]', '', text)
        # text = re.sub('\n', ' ', text)                  
        return text
        
    # Defino una funcion que recibe un texto y devuelve el mismo texto sin signos,
    def clean_text(self, text):
        
        # pasa las mayusculas del texto a minusculas
        text = text.lower()                                              
        # reemplaza texto entre corchetes por espacio en blanco.. ¿ y \% no se..
        text = re.sub('\[.*?¿\]\%', '', text)                           
        # reemplaza signos de puntuacion por espacio en blanco.. %s -> \S+ es cualquier caracter que no sea un espacio en blanco
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text) 
        # remueve palabras que contienen numeros.
        text = re.sub('\w*\d\w*', '', text)       
        # Sacamos comillas, los puntos suspensivos, <<, >>
        text = re.sub('[‘’“”…«»]', '', text)
        text = re.sub('\n', ' ', text)                  
        return text
        
    def lemmatizer(self, words, language = "english"):
            
        words_clean = []
        if (language == "english"):
            nltk.download('wordnet')
            lemmatizer = WordNetLemmatizer() # funcion para lematizar
            for w in words:
                words_clean.append(lemmatizer.lemmatize(w))
        elif (language == "spanish"):
                
            nlp = spacy.load("es_core_news_sm")
            for token in words:
                words_clean.append(nlp(token)[0].lemma_)
        return words_clean
    
    def remove_stopwords(self,words,language = "english"):

        nltk.download('stopwords') # hay que descargar este modulo en particular
        sw = nltk.corpus.stopwords.words(language) # lista de stopwords

        words_clean = []
        for w in words:
          if not w in sw: # si no es stopword, agregamos la version lematizada
            words_clean.append(w)
        return words_clean
    
    def normalize_noun_count(self,df_tagged,language="english"):
        
        noun_counter=0
        if (language=="english"):
            for i,row in df_tagged.iterrows():
                if((row[1]=="NN") | (row[1]=="NNS")):
                    noun_counter+=1
        elif language == "spanish":
            for i,row in df_tagged.iterrows():
                if(row[1]=="NC"):
                    noun_counter+=1
        return noun_counter/(len(df_tagged.index))
    
    def normalize_verb_count(self,df_tagged,language="english"):
        verb_counter=0
        for i,row in df_tagged.iterrows():
            if(str(row[1]).startswith("V")):
                verb_counter+=1
        return verb_counter/(len(df_tagged.index))
    
    def normalize_adj_count(self,df_tagged,language="english"):
        
        adj_counter=0
        if (language=="english"):
            for i,row in df_tagged.iterrows():
                if(row[1]=="JJ"):
                    adj_counter+=1
        elif language == "spanish":
            for i,row in df_tagged.iterrows():
                if(row[1]=="ADJ"):
                    adj_counter+=1
        return adj_counter/(len(df_tagged.index))
    
    def normalize_adv_count(self,df_tagged,language="english"):
        
        adv_counter=0
        if (language=="english"):
            for i,row in df_tagged.iterrows():
                if(row[1]=="RB"):
                    adv_counter+=1
        elif language == "spanish":
            for i,row in df_tagged.iterrows():
                if(row[1]=="ADV"):
                    adv_counter+=1
        return adv_counter/(len(df_tagged.index))
    
    
    def normalize_noun_count_content(self,df_tagged,language="english"):
        
        noun_counter=0
        content_word_counter=0

        if (language=="english"):
            for i,row in df_tagged.iterrows():
                if((str(row[1]).startswith("V")) | (row[1]=="NN") | (row[1]=="NNS") |
                   (str(row[1]).startswith("JJ")) | (str(row[1]).startswith("RB"))):
                    content_word_counter+=1
                    if((row[1]=="NN") | (row[1]=="NNS")):
                        noun_counter+=1
                
        elif language == "spanish":
            for i,row in df_tagged.iterrows():
                if((str(row[1]).startswith("V")) | (row[1]=="NC") |
                   (row[1]=="ADV") | (row[1]=="ADJ")):
                    content_word_counter+=1
                    if (row[1]=="NC"):
                        noun_counter+=1
        return noun_counter/content_word_counter
    
    def normalize_verb_count_content(self,df_tagged,language="english"):
        verb_counter=0
        content_word_counter=0

        if (language=="english"):
            for i,row in df_tagged.iterrows():
                if((str(row[1]).startswith("V")) | (row[1]=="NN") | (row[1]=="NNS") |
                   (str(row[1]).startswith("JJ")) | (str(row[1]).startswith("RB"))):
                    content_word_counter+=1
                    if(str(row[1]).startswith("V")):
                        verb_counter+=1
                
        elif language == "spanish":
            for i,row in df_tagged.iterrows():
                if((str(row[1]).startswith("V")) | (row[1]=="NC") |
                   (row[1]=="ADV") | (row[1]=="ADJ")):
                    content_word_counter+=1
                    if (str(row[1]).startswith("V")):
                        verb_counter+=1
        return verb_counter/content_word_counter
    
    def normalize_adj_count_content(self,df_tagged,language="english"):
        adj_counter=0
        content_word_counter=0

        if (language=="english"):
            for i,row in df_tagged.iterrows():
                if((str(row[1]).startswith("V")) | (row[1]=="NN") | (row[1]=="NNS") |
                   (str(row[1]).startswith("JJ")) | (str(row[1]).startswith("RB"))):
                    content_word_counter+=1
                    if(str(row[1]).startswith("JJ")):
                        adj_counter+=1
                
        elif language == "spanish":
            for i,row in df_tagged.iterrows():
                if((str(row[1]).startswith("V")) | (row[1]=="NC") |
                   (row[1]=="ADV") | (row[1]=="ADJ")):
                    content_word_counter+=1
                    if (str(row[1]).startswith("ADJ")):
                        adj_counter+=1
        return adj_counter/content_word_counter
        
    def normalize_adv_count_content(self,df_tagged,language="english"):
        adv_counter=0
        content_word_counter=0

        if (language=="english"):
            for i,row in df_tagged.iterrows():
                if((str(row[1]).startswith("V")) | (row[1]=="NN") | (row[1]=="NNS") |
                   (str(row[1]).startswith("JJ")) | (str(row[1]).startswith("RB"))):
                    content_word_counter+=1
                    if(str(row[1]).startswith("RB")):
                        adv_counter+=1
                
        elif language == "spanish":
            for i,row in df_tagged.iterrows():
                if((str(row[1]).startswith("V")) | (row[1]=="NC") |
                   (row[1]=="ADV") | (row[1]=="ADJ")):
                    content_word_counter+=1
                    if (str(row[1]).startswith("ADV")):
                        adv_counter+=1
        return adv_counter/content_word_counter
    
    def TreeTagger_text_to_list(self,text,language = "spanish"):
        
        tt = TreeTagger(path_to_treetagger='C:\TreeTagger',language=language);
        resultado = tt.tag(text);
        return resultado

    def Filter_tagged_words_list(self,tagged_list,word_type='NN'):
        
        tag_filtered = []
        for tag in tagged_list:
            if (tag[1]==word_type):
                tag_filtered.append(tag)
        return tag_filtered
    
    def TreeTagger(self, transcripts_dict,path_to_save):
        """
        Parameters:
        ----------
        IMPORTANT: TreeTagger module must be located in C:\\TreeTagger.
        
        transcripts_dict: must be structured as follows: 
                          {'path': [list of paths to the .csv files containing the transcripts],
                           'language': [list of transcript languages for each path in the list, 
                            given in the same order]}.

        path_to_save: path to save results.
        """
        
        transcripts = pd.DataFrame(transcripts_dict)
        TreeTagger_path = Path('C:\\TreeTagger')
    
        os.makedirs(path_to_save,exist_ok=True)

        os.chdir(TreeTagger_path)
        os.system('set PATH=' + str(Path(TreeTagger_path,'bin'))+';%PATH%')
        for i,r in transcripts.iterrows():

            parent = r['path'].parent    
            os.makedirs(Path(parent,'Transcripts'),exist_ok=True)
            print(r['path'])
            transcript = pd.read_csv(r['path'],sep=',',encoding='utf_8',header=0)
            lang = r['language']

            for j,r2 in transcript.iterrows():
                
                filepath = Path(parent,'Transcripts',str(r2['ID']) + '_' + r['path'].name).with_suffix('.txt')
                file = open(filepath,'wb')
                file.write(str(r2['Transcript']).encode('utf8'))
                file.close()
                os.system('tag-' + str(lang)  + ' ' + str(filepath) + ' ' + str(Path(parent,'Transcripts','tagged_' + filepath.name)))            
               
    def FilterTaggedWords(self,path,word_type='NN'):
        """
        Parameters:
        ----------

        path: path to .txt files where the tagged transcripts are saved.
        word_type: code of the word type to be kept, as defined by TreeTagger (default=NN)

        """    
        os.makedirs(Path(path,'Filtered'),exist_ok=True)
        
        files = Path(path).glob('tagged_*Chile*.txt')

        for file in files:
            data = pd.read_table(file,sep='\t',names = ['word','category','lemma'], header=None)
            
            for w_type in word_type:
                
                filtered_data = data.loc[data.iloc[:,1] == w_type,:].reset_index(drop=True)
            if len(file.name.split('_')) == 5:
                (_,name1,name2,group,_) = file.name.split('_')
                name = name1 + '_' + name2
            else:
                (_,name,group,_) = file.name.split('_')
            filename_to_save = Path(path,'Filtered', name + '_' + group).with_suffix('.csv')
            filtered_data.to_csv(filename_to_save,encoding='utf8',index=False)
            
    def clusterization(self,vector_words):
        if len(vector_words)>1:
            acum = 0
            for i in range(0,len(vector_words)-1):
                for j in range(i+1,len(vector_words)):
                    acum+=(np.abs(np.dot(vector_words[i],vector_words[j])/(np.linalg.norm(vector_words[i])*np.linalg.norm(vector_words[j]))))
            threshold = ((math.factorial(len(vector_words)-2)*2)/(math.factorial(len(vector_words))))*acum
            
            clusters = []
            actual_cluster = [vector_words[0]]
            for i in range(1,len(vector_words)):
                array_2d = np.stack(actual_cluster,axis=0)
                mu = np.mean(array_2d,axis=0)
                if np.abs(((distance.cosine(mu, vector_words[i])-1)*-1)) > threshold:
                    actual_cluster.append(vector_words[i])
                else:
                    clusters.append(actual_cluster)
                    actual_cluster = [vector_words[i]]
            clusters.append(actual_cluster)
            return clusters
            
        else:
            return np.nan


            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
                                
        