# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 10:59:05 2021

@author: franc
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, OrderedDict

import math

from torchtext.data import get_tokenizer

from googletrans import Translator
# from deep_translator import GoogleTranslator
# pip install googletrans==4.0.0rc1

# pip install pickle-mixin

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

# python -m spacy download es_core_news_sm
import spacy

import fasttext.util

import contractions

import re      # library of regular expressions
import string   # string library

import itertools

import sys
sys.path.append("/tmp/TEST")

from treetagger import TreeTagger


from scipy.spatial import distance
from scipy.stats import kurtosis
from scipy.stats import skew

from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import shutil

import requests
from bs4 import BeautifulSoup

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import fasttext
import fasttext.util


class NLPClass:
    def __init__(self):
        self.numero = 1
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()

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
      tokens = [tokenizer(x) if ((str(x)!="nan"))else x for x in text_dataframe]
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
        '''
        It check if the word is in path+"//translations.pkl", if it is not, it adds it to the file.

        Parameters
        ----------
        path : string
            The path where the translation file is.
        words : list of strings
            List of words to obtain the translation.
        lan_src : string, optional
            DESCRIPTION. language in which each word of words is.
            The default is "spanish".
        lan_dest : string, optional
            language in which each word will be translated.
            The default is "english".

        Returns
        -------
        None.

        '''
        df_translation = self.read_pickle_translation_file(path)
        
        traducidas = 0

        for i,word in enumerate(words):

            df_check = df_translation[(df_translation.word == word) & (df_translation.lan_src == lan_src) & (df_translation.lan_dest == lan_dest)]

            if len(df_check.index) == 0:
                
                traducidas +=1

                print("Traduciendo " + word +": " + str(i) + "/" + str(len(words)))

                new_row = [word,self.translate([word],lan_src,lan_dest)[0].extra_data["parsed"],lan_src,lan_dest]

                df_length = len(df_translation)

                df_translation.loc[df_length] = new_row
                
            if traducidas%50 == 0:
                df_translation.to_pickle(path)
                
        df_translation.to_pickle(path)
    
    def read_pickle_translation_file(self,path):
        '''
        Read pickle file with the all translations DataFrame.

        Parameters
        ----------
        path : string
            Path where the picke file is.

        Returns
        -------
        df_translation : pandas.DataFrame
            df with all the translations with the following structure:
                word|translation|lan_src|lan_dest|

        '''
        try:
            df_translation = pd.read_pickle(path)
        except (OSError, IOError):
            print("translation.pkl no encontrado")
            df_translation = pd.DataFrame(columns=['word','translation','lan_src','lan_dest'])
        return df_translation
    
   
    def read_pickle_synonyms_file(self,path):
        '''
        Read pickle file with the all synonyms DataFrame.

        Parameters
        ----------
        path : string
            Path where the picke file is.

        Returns
        -------
        df_synonyms : pandas.DataFrame
            df with all the synonyms obtained with the following structure:
                word|synonym_0|synonym_1|synonym_2|...

        '''
        try:
            df_synonyms = pd.read_pickle(path+"//synonyms.pkl")
        except (OSError, IOError):
            print("synonyms.pkl no encontrado")
            return pd.DataFrame(columns=["word"])

        return df_synonyms
    
    
    def read_pickle_synonyms_file_fasttext(self,path):
        '''
        Read pickle file with the all synonyms DataFrame.

        Parameters
        ----------
        path : string
            Path where the picke file is.

        Returns
        -------
        df_synonyms : pandas.DataFrame
            df with all the synonyms obtained with the following structure:
                word|synonym_0|synonym_1|synonym_2|...

        '''
        try:
            df_synonyms = pd.read_pickle(path+"//synonyms_fasttext.pkl")
        except (OSError, IOError):
            print("synonyms.pkl no encontrado")
            return pd.DataFrame(columns=["word"])

        return df_synonyms
        
    
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
        """
        Given a list of words and a hypernym, finds the minimum number of nodes between each word's
        synset and the given hypernym, using the hypernym_min_nodes_distance_from_synset_to_hypernym method.
        If at least one word's minimum distance is found, returns the minimum value.
        If none of the words have a minimum distance (i.e., they do not have the given hypernym), returns -1.    
        
        Args:
        - words (list): A list of words for which to find the minimum distance to a specific hypernym.
        - hypernym_check (str): The hypernym to which the minimum distance should be found. Default is "synset.n.01".
    
        Returns:
        - int: The minimum number of nodes between each word in the input list and the specified hypernym. If no valid 
               minimum distance is found, returns -1.
        """
        
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
        """
         Calculates the cross-feature variability for a given set of words and a concept.
         This metric measures how much the semantic distances between the words and the concept vary across participants.
        
         Args:
         - vector_words (list): a list of strings, where each string contains a participant's transcribed words
         - concept (string): the concept to which the distances will be calculated
        
         Returns:
         - ongoing_semantic_list (list): a list of cross-feature variabilities for each participant's words in vector_words
         """
         
        # I convert to word embedding every word spoken by each patient.
        words_vector = self.get_word_fast_text_vector(vector_words)

                
        # I convert the concept to word_embedding.
        concept_vector = self.get_word_fast_text_vector(concept)[0][0]

        
        # I calculate the semantic distance between each word and the concept.
        words_distances = self.cross_semantic_distance(words_vector,concept_vector)
                
        # I calculate the cross_feature_variability of each patient.
        
        ongoing_semantic_list = list()
        for words in words_distances:
            ongoing_semantic_list.append(self.ongoing_semantic_variability(words))

        return ongoing_semantic_list
    
    
    
    def cross_semantic_distance(self,words_vector,concept_vector):
        """
        Calculates the semantic distance between each word in a list of word vectors and a given concept vector.
        
        Args:
            words_vector (list): A list of word vectors, where each element of the list is a list of vectors for a given patient.
            concept_vector (list): A word vector representing a concept.
            
        Returns:
            list: A list of lists containing the semantic distance between each word vector and the concept vector.
        """
        words_distances = list()
        for i_lista, vector_list in enumerate(words_vector):
            words_distances.append([])
            for i_vector,vector in enumerate(vector_list):
                resultado = self.semantic_words_distance(vector_list[i_vector],concept_vector)
                words_distances[i_lista].append(resultado)
        return words_distances
                
                
    def download_and_save_fast_text_model(self,save = True):
        """Downloads and saves a pre-trained FastText model in Spanish.
        Args:
            save (bool, optional): Whether to save the model or not. Defaults to True.
    
        Returns:
            fasttext.FastText._FastText: The downloaded and loaded FastText model.
        """
        fasttext.util.download_model('es', if_exists='ignore')  # English
        ft = fasttext.load_model('cc.es.300.bin')
        if save:
            ft.save_model('cc.es.300.bin') # First time only
        self.fasttext = ft
        return ft
    
    def load_fast_text_model(self,path_fast_text):
        """
        Loads a pre-trained FastText model from disk and sets it as the model to be used by the instance of the class.
        
        Parameters:
            path_fast_text (str): The path to the directory where the FastText model binary file is saved.
            
        Returns:
            The FastText model object that was loaded from disk.
        """
        self.fasttext = fasttext.load_model(path_fast_text)
        return self.fasttext
    
    def reduce_fasttext_model(self,path_fast_text,size):
        """
        Loads a pre-trained FastText model from disk and sets it as the model to be used by the instance of the class.
        
        Parameters:
            path_fast_text (str): The path to the directory where the FastText model binary file is saved.
            
        Returns:
            The FastText model object that was loaded from disk.
        """
        
        fasttext.util.reduce_model(self.fasttext, size)
        self.fasttext_reduced = self.fasttext
        self.fasttext = fasttext.load_model(path_fast_text)

        return self.fasttext_reduced
    
    def get_word_fast_text_vector_reduced(self,vector_words):
        """
        Returns the word embedding for the given list of words.
    
        Args:
            vector_words (list): A list of words. Each word can be a single string or a list of strings.
            
        Returns:
            list: A list of lists of word embeddings. The outer list has the same length as `vector_words`, and
                each inner list contains the word embeddings for the corresponding element in `vector_words`.
                
        Raises:
            ValueError: If the fastText model has not been loaded.

        """
        words_vector = list()
        for i_lista,word_list in enumerate(vector_words):
            words_vector.append([])
            for i_word,word in enumerate(word_list):
                words_vector[i_lista].append(self.fasttext_reduced.get_word_vector(word))
                
        return words_vector
    
    def ongoing_semantic_variability_complete_fasttext_size(self, vector_words):
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

        
        # I obtain the fasttext vector for each word for each patient.
        words_vector = self.get_word_fast_text_vector_reduced(vector_words)
        
        
        # I calculate the semantic distance between each contiguous word (vector) spoken by each patient.
        words_distances = self.ongoing_semantic_distance(words_vector)

                
        # I calculate the Ongoing Semantic Variability of each patient.
            
        ongoing_semantic_list = list()
        for words in words_distances:
            ongoing_semantic_list.append(self.ongoing_semantic_variability(words))

        return ongoing_semantic_list
    
    def delete_all_zeros_vectores(self,vector_words):
        vector_words_filtrado = []
        for words in vector_words:
            lista_filtrada = [sublista for sublista in words if any(elemento != 0 for elemento in sublista)]
            vector_words_filtrado.append(lista_filtrada)
        return vector_words_filtrado
    
    def get_word_fast_text_vector(self,vector_words):
        """
        Returns the word embedding for the given list of words.
    
        Args:
            vector_words (list): A list of words. Each word can be a single string or a list of strings.
            
        Returns:
            list: A list of lists of word embeddings. The outer list has the same length as `vector_words`, and
                each inner list contains the word embeddings for the corresponding element in `vector_words`.
                
        Raises:
            ValueError: If the fastText model has not been loaded.

        """
        words_vector = list()
        for i_lista,word_list in enumerate(vector_words):
            words_vector.append([])
            for i_word,word in enumerate(word_list):
                words_vector[i_lista].append(self.fasttext.get_word_vector(word))
                
        return words_vector
    
    def ongoing_semantic_distance(self,words_vector):
        """
        Computes the ongoing semantic distance between pairs of words within a list of word vectors.

        Args:
            words_vector (list): A list of word vectors.

        Returns:
            list: A list of lists, where each sublist represents the ongoing semantic distance between each pair of words
                  in the corresponding input list of word vectors. If a word vector contains NaN values, it will be
                  excluded from the computation.

        Example:
            >>> words_vector = [[np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6]), np.nan],
                                [np.array([0.7, 0.8, 0.9]), np.array([0.2, 0.3, 0.4]), np.array([0.1, 0.2, 0.3])]]
            >>> ongoing_semantic_distance(words_vector)
            [[0.2922831028530194], [0.44797496897048146, 0.41269131875038147]]
        """
        words_distances = list()
        for i_lista, vector_list in enumerate(words_vector):
            words_distances.append([])
            vector_list = [x for x in vector_list if str(x) != 'nan']
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

        print("OSV: Calculating osv")
        print("OSV: Getting osv vectors")
        # I obtain the fasttext vector for each word for each patient.
        words_vector = self.get_word_fast_text_vector(vector_words)
        
        words_vector_filtered = self.delete_all_zeros_vectores(words_vector)
        
        print("OSV: Calculating distances")
        # I calculate the semantic distance between each contiguous word (vector) spoken by each patient.
        words_distances = self.ongoing_semantic_distance(words_vector_filtered)

                
        # I calculate the Ongoing Semantic Variability of each patient.
        print("OSV: Calculating SV")
        ongoing_semantic_list = list()
        for words in words_distances:
            ongoing_semantic_list.append(self.ongoing_semantic_variability(words))

        return ongoing_semantic_list
        

    def expand_contractions_dataframe(self, token_list):
        """This function expands contractions in a given dataframe column.

        Args:
            token_list (pandas.Series): A series with contractions that need to be expanded.
    
        Returns:
            pandas.Series: A series with the expanded contractions.
    
        Example:
            >>> df = pd.DataFrame({'text': ["I don't know what I'd do without you.", "He's always on time."]})
            >>> token_list = df['text']
            >>> expanded_tokens = expand_contractions_dataframe(token_list)
            >>> print(expanded_tokens)
            0    I do not know what I would do without you.
            1                      He is always on time.
            Name: text, dtype: object
        """
        # I define an anonymous function that when passed an argument returns the result of applying the previous function to the same argument.
        round0 = lambda x: contractions.fix(x)
        
        # Dataframe resulting from applying the cleanup function to the columns
        token_expanded = token_list.apply(round0)
        
        return token_expanded
    
    def expand_contractions(self, token):
        """This function expands a contraction in a given token.
        Args:
            token (str): A string containing a contraction that needs to be expanded.
        
        Returns:
            str: A string with the expanded contraction.
        
        Example:
            >>> token = "I can't believe it's raining again."
            >>> expanded_token = expand_contractions(token)
            >>> print(expanded_token)
            I cannot believe it is raining again.
        """
      
        # Expand the token
        token_expanded = contractions.fix(token)
              
        return token_expanded
    
    def clean_text_paula(self, text):
        """This function cleans a given text by removing punctuation marks.

        Args:
            text (str): A string containing the text to be cleaned.
        
        Returns:
            str: A string with no punctuation marks.
        
        Example:
            >>> text = "This is a sentence! It has punctuation, and some numbers 12345."
            >>> cleaned_text = clean_text_paula(text)
            >>> print(cleaned_text)
            this is a sentence it has punctuation and some numbers
        """
        
        # capitalize the text to lowercase letters
        text = text.lower()                                              
        # replaces bracketed text with white space... ¿ and \% I do not know...
        
        text = " ".join(re.findall('(?<!\S)[a-z_]+(?=[,.!?:;]?(?!\S))', text))
                     
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text) 

        # # replaces punctuation marks with whitespace... %s -> \S+ is any character that is not a blank space.
        # text = re.sub('[%s]' % re.escape(string.punctuation), '', text) 
        # # removes words containing numbers.
        # text = re.sub('\w*\d\w*', '', text)       
        # # We remove quotation marks, ellipses, <<, >>.
        # text = re.sub('[‘’“”…«»]', '', text)
        # text = re.sub('\n', ' ', text)                  
        return text
        
    def clean_text_spanish(self, text, char_replace = ''):
        """
        This function takes a text and removes all punctuation marks, words containing numbers, and other characters
        that are not alphanumeric or whitespace. It also converts the text to lowercase and replaces certain
        characters with a user-specified replacement character.
    
        Args:
            text (str): The input text to be cleaned.
            char_replace (str, optional): The character to replace all removed characters with. Defaults to an
                empty string.
    
        Returns:
            str: The cleaned text with no punctuation marks, words containing numbers, or other unwanted characters.
        """        
        # capitalize the text to lowercase letters
        text = text.lower()                                              
        # replaces bracketed text with white space... ¿ and \% I do not know...
        text = re.sub('\[.*?¿\]\%', char_replace, text)                           
        # # replaces punctuation marks with whitespace... %s -> \S+ is any character that is not a blank space.
        text = re.sub('[%s]' % re.escape(string.punctuation), char_replace, text) 
        # # removes words containing numbers.
        text = re.sub('\w*\d\w*', char_replace, text)       
        # # We remove quotation marks, ellipses, <<, >>.
        text = re.sub('[‘’“”…«»]', char_replace, text)
        # I keep only alphanumeric characters
        text = re.sub('[^a-zA-Z0-9 \náéíóúÁÉÍÓÚñÑü\.]', char_replace, text)
        text = re.sub('\n', char_replace, text)   
        text = re.sub('\s+', ' ', text) 
                       
        return text
        
    def lemmatizer(self, words, language = "english"):
        """
        This function takes a list of words and returns a new list of words where each word has been lemmatized. The
        function supports two languages: English and Spanish.
    
        Args:
            words (list): A list of words to be lemmatized.
            language (str, optional): The language of the input text. Defaults to "english".
    
        Returns:
            list: A new list of lemmatized words.
        """
        words_clean = []
        if (language == "english"):
            nltk.download('wordnet')
            lemmatizer = WordNetLemmatizer() # lemmatizing function
            for w in words:
                words_clean.append(lemmatizer.lemmatize(w))
        elif (language == "spanish"):
                
            nlp = spacy.load("es_core_news_sm")
            for token in words:
                words_clean.append(nlp(token)[0].lemma_)
        return words_clean
    
    def remove_stopwords(self,words,language = "english"):
        """
        This function takes a list of words and removes any stop words from it, where stop words are defined as words
        that are considered to be uninformative and are usually removed from text before processing. The function
        supports two languages: English and Spanish.
    
        Args:
            words (list): A list of words to be filtered.
            language (str, optional): The language of the input text. Defaults to "english".
    
        Returns:
            list: A new list of words with all stop words removed.
        """
        nltk.download('stopwords') # this particular module must be downloaded
        sw = nltk.corpus.stopwords.words(language) # stopwords list

        words_clean = []
        for w in words:
          if not w in sw: # if it is not stopword, we add the lemmatized version
            words_clean.append(w)
        return words_clean
    
    def normalize_noun_count(self,df_tagged,language="english"):
        """
        This function takes a pandas dataframe that contains tagged text, and returns the proportion of words in the text
        that are nouns. The function supports two languages: English and Spanish.
    
        Args:
            df_tagged (pandas.DataFrame): A dataframe containing tagged text.
            language (str, optional): The language of the input text. Defaults to "english".
    
        Returns:
            float: The proportion of words in the text that are nouns.
        """
        noun_counter=0
        if (language=="english"):
            for i,row in df_tagged.iterrows():
                if((row[1]=="noun") | (row[1]=="noun")):
                    noun_counter+=1
        elif language == "spanish":
            for i,row in df_tagged.iterrows():
                if(row[1]=="NC"):
                    noun_counter+=1
        return noun_counter/(len(df_tagged.index))
    
    def normalize_verb_count(self,df_tagged,language="english"):
        """
        This function takes a pandas dataframe that contains tagged text, and returns the proportion of words in the text
        that are verbs. The function supports two languages: English and Spanish.
        
        Args:
            df_tagged (pandas.DataFrame): A dataframe containing tagged text.
            language (str, optional): The language of the input text. Defaults to "english".
        
        Returns:
            float: The proportion of words in the text that are nouns.
        """
        verb_counter=0
        for i,row in df_tagged.iterrows():
            if(row[1] == "verb"):
                verb_counter+=1
        return verb_counter/(len(df_tagged.index))
    
    def normalize_adj_count(self,df_tagged,language="english"):
        """
        This function takes a pandas dataframe that contains tagged text, and returns the proportion of words in the text
        that are adjectives. The function supports two languages: English and Spanish.
        
        Args:
            df_tagged (pandas.DataFrame): A dataframe containing tagged text.
            language (str, optional): The language of the input text. Defaults to "english".
        
        Returns:
            float: The proportion of words in the text that are nouns.
        """
        adj_counter=0
        if (language=="english"):
            for i,row in df_tagged.iterrows():
                if(row[1]=="adjective"):
                    adj_counter+=1
        elif language == "spanish":
            for i,row in df_tagged.iterrows():
                if(row[1]=="ADJ"):
                    adj_counter+=1
        return adj_counter/(len(df_tagged.index))
    
    def normalize_adv_count(self,df_tagged,language="english"):
        """
        This function takes a pandas dataframe that contains tagged text, and returns the proportion of words in the text
        that are adverbs. The function supports two languages: English and Spanish.
    
        Args:
            df_tagged (pandas.DataFrame): A dataframe containing tagged text.
            language (str, optional): The language of the input text. Defaults to "english".
    
        Returns:
            float: The proportion of words in the text that are nouns.
        """        
        adv_counter=0
        if (language=="english"):
            for i,row in df_tagged.iterrows():
                if(row[1]=="adverb"):
                    adv_counter+=1
        elif language == "spanish":
            for i,row in df_tagged.iterrows():
                if(row[1]=="ADV"):
                    adv_counter+=1
        return adv_counter/(len(df_tagged.index))
    
    
    def normalize_noun_count_content(self,df_tagged,language="english"):
        """
        This function takes a pandas dataframe that contains tagged text, and returns the proportion of content words in the text
        that are nouns. The function supports two languages: English and Spanish.
        
        Args:
            df_tagged (pandas.DataFrame): A dataframe containing tagged text.
            language (str, optional): The language of the input text. Defaults to "english".
        
        Returns:
            float: The proportion of words in the text that are nouns.
        """
        noun_counter=0
        content_word_counter=0

        if (language=="english"):
            for i,row in df_tagged.iterrows():
                if((row[1] == "verb") | (row[1] == "noun") |
                   (row[1]=="adjective") | (row[1]=="adverb")):
                    content_word_counter+=1
                    if((row[1]=="noun") | (row[1]=="noun")):
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
        """
        This function takes a pandas dataframe that contains tagged text, and returns the proportion of content words in the text
        that are verbs. The function supports two languages: English and Spanish.
        
        Args:
            df_tagged (pandas.DataFrame): A dataframe containing tagged text.
            language (str, optional): The language of the input text. Defaults to "english".
        
        Returns:
            float: The proportion of words in the text that are nouns.
        """
        verb_counter=0
        content_word_counter=0

        if (language=="english"):
            for i,row in df_tagged.iterrows():
                if((row[1] == "verb") | (row[1] == "noun") |
                   (row[1]=="adjective") | (row[1]=="adverb")):
                    content_word_counter+=1
                    if(str(row[1]).startswith("verb")):
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
        """
        This function takes a pandas dataframe that contains tagged text, and returns the proportion of content words in the text
        that are adjectives. The function supports two languages: English and Spanish.
        
        Args:
            df_tagged (pandas.DataFrame): A dataframe containing tagged text.
            language (str, optional): The language of the input text. Defaults to "english".
        
        Returns:
            float: The proportion of words in the text that are nouns.
        """
        adj_counter=0
        content_word_counter=0

        if (language=="english"):
            for i,row in df_tagged.iterrows():
                if((row[1] == "verb") | (row[1] == "noun") |
                   (row[1]=="adjective") | (row[1]=="adverb")):
                    content_word_counter+=1
                    if(str(row[1]).startswith("adjective")):
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
        """
        This function takes a pandas dataframe that contains tagged text, and returns the proportion of content words in the text
        that are adverbs. The function supports two languages: English and Spanish.
        
        Args:
            df_tagged (pandas.DataFrame): A dataframe containing tagged text.
            language (str, optional): The language of the input text. Defaults to "english".
        
        Returns:
            float: The proportion of words in the text that are nouns.
        """
        adv_counter=0
        content_word_counter=0

        if (language=="english"):
            for i,row in df_tagged.iterrows():
                if((row[1] == "verb") | (row[1] == "noun") |
                   (row[1]=="adjective") | (row[1]=="adverb")):
                    content_word_counter+=1
                    if(str(row[1]).startswith("adverb")):
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
        """
        This function takes a string of text and uses TreeTagger to annotate each word in the text with its part of speech.
        The function returns a list of (word, tag) tuples. The function supports two languages: English and Spanish.
    
        Args:
            text (str): A string of text to be annotated.
            language (str, optional): The language of the input text. Defaults to "spanish".
    
        Returns:
            list: A list of (word, tag) tuples where each tuple represents a word in the text and its part of speech.
        """
        tt = TreeTagger(path_to_treetagger='C:\TreeTagger',language=language);
        resultado = tt.tag(text);
        return resultado

    def Filter_tagged_words_list(self,tagged_list,word_type='NN'):
        """
        This function takes a list of (word, tag) tuples and returns a filtered list containing only those tuples whose
        tag matches a specified part of speech. The default part of speech is noun (NN).
        
        Args:
            tagged_list (list): A list of (word, tag) tuples.
            word_type (str, optional): The part of speech to filter the list by. Defaults to 'NN'.
        
        Returns:
            list: A list of (word, tag) tuples whose tag matches the specified part of speech.
        """
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
        """
        This function performs hierarchical clustering on a set of vectors, returning a list of clusters.
        
        Args:
            vector_words (list): A list of numpy arrays representing word vectors.
        
        Returns:
            list: A list of clusters, where each cluster is a list of numpy arrays representing word vectors. If there is
            only one vector in the input list, returns NaN.
        """
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


    def psycholinguistics_features(self, data, psycholinguistics_columns, tokens_columns, df):
        '''
        It adds a new column for each psycholinguistics feature in psycholinguistics_columns 
        where it asign to each row the corresponding psycholinguistic values to each token.

        Parameters
        ----------
        data : pandas.DataFrame
            The DataFrame with one row for each different token with the 
            corresponding psycholinguistic values. Obtained from https://www.bcbl.eu/databases/espal/
        psycholinguistics_columns : List of column names
            A list with the names of the columns in data with the 
            psycholinguistic variables.
        columnas_tokens : List of column names
            A list with the names of df where the tokens are.
        df : pandas.DataFrame
            The DataFrame with the lists of tokens.

        Returns
        -------
        df : pandas.DataFrame
            The same DataFrame received but with as many columns more as length of
            psycholinguistics_columns multiplied by length of tokens_columns.
            
        Example:
        Parameters:
            data:
                word|log_frq|sa_num_phon|
                perro|1.85108|4|
                gato|1.59723|4|
                caballo|1.92165|6|
                papa|1.93218|4|
                piso|1.55915|4|
                pintura|1.65155|7|
            psycholinguistics_columns:
                ["log_frq","sa_num_phon"]
            columnas_tokens:
                ["fluency_p","fluency_animals"]
            df:
                codigo|diagnostico|fluency_p|fluency_animals|
                001|AD|["papa","pintura"],["caballo","gato"]
                002|CTR|["piso","pintura","papa"],["perro","gato"]
            
        Returns:
            df:
                codigo|diagnostico|fluency_p|fluency_animals|fluency_p_log_frq|fluency_p_sa_num_phon|fluency_animals_log_frq|fluency_animals_sa_num_phon|
                001|AD|["papa","pintura"]|["caballo","gato"]|[1.93218,1.65155]|[4,7]|[1.92165,1.59723]|[6,4]
                002|CTR|["piso","pintura","papa"]|["perro","gato"]|[1.55915,1.65155,1.93218]|[4,7,4]|[1.55915,1.65155,1.93218]|[4,7,4]
        '''
        for column in tokens_columns:
                
            for psico_column in psycholinguistics_columns:
                df[column+"_"+psico_column] = np.nan
                df[column+"_"+psico_column] = df[column+"_"+psico_column].astype(object)
                list_values = []

                for i,row in df.iterrows():
                    list_values.append([])

                    for words in row[column]:
                        if len(words.split()) > 1:
                            list_values_element = []
                            for word in words.split():
                                list_values_element.append(next(iter(list(set(data[data["word"] == word][psico_column].values))), np.nan))
                            list_values[i].append(np.nanmean(list_values_element))
                        else:
                            list_values[i].append(next(iter(list(set(data[data["word"] == words][psico_column].values))), np.nan))
                df[column+"_"+psico_column] = list_values

        return df
    
    
    
    def get_words_nan_psycholinguistics(self, data, psycholinguistics_columns, tokens_columns, df):
        '''
        It adds a new column for each psycholinguistics feature in psycholinguistics_columns 
        where it asign to each row the corresponding psycholinguistic values to each token.

        Parameters
        ----------
        data : pandas.DataFrame
            The DataFrame with one row for each different token with the 
            corresponding psycholinguistic values. Obtained from https://www.bcbl.eu/databases/espal/
        psycholinguistics_columns : List of column names
            A list with the names of the columns in data with the 
            psycholinguistic variables.
        columnas_tokens : List of column names
            A list with the names of df where the tokens are.
        df : pandas.DataFrame
            The DataFrame with the lists of tokens.

        Returns
        -------
        df : pandas.DataFrame
            The same DataFrame received but with as many columns more as length of
            psycholinguistics_columns multiplied by length of tokens_columns.
            
        Example:
        Parameters:
            data:
                word|log_frq|sa_num_phon|
                perro|1.85108|4|
                gato|1.59723|4|
                caballo|1.92165|6|
                papa|1.93218|4|
                piso|1.55915|4|
                pintura|1.65155|7|
            psycholinguistics_columns:
                ["log_frq","sa_num_phon"]
            columnas_tokens:
                ["fluency_p","fluency_animals"]
            df:
                codigo|diagnostico|fluency_p|fluency_animals|
                001|AD|["papa","pintura"],["caballo","gato"]
                002|CTR|["piso","pintura","papa"],["perro","gato"]
            
        Returns:
            df:
                codigo|diagnostico|fluency_p|fluency_animals|fluency_p_log_frq|fluency_p_sa_num_phon|fluency_animals_log_frq|fluency_animals_sa_num_phon|
                001|AD|["papa","pintura"]|["caballo","gato"]|[1.93218,1.65155]|[4,7]|[1.92165,1.59723]|[6,4]
                002|CTR|["piso","pintura","papa"]|["perro","gato"]|[1.55915,1.65155,1.93218]|[4,7,4]|[1.55915,1.65155,1.93218]|[4,7,4]
        '''
        palabras_nan = set()
        for column in tokens_columns:
                
            for psico_column in psycholinguistics_columns:
                df[column+"_"+psico_column] = np.nan
                df[column+"_"+psico_column] = df[column+"_"+psico_column].astype(object)
                list_values = []

                for i,row in df.iterrows():
                    list_values.append([])

                    for words in row[column]:
                        if len(words.split()) > 1:
                            list_values_element = []
                            for word in words.split():
                                list_values_element.append(next(iter(list(set(data[data["word"] == word][psico_column].values))), np.nan))
                                if str(list_values_element[-1]) == "nan":
                                    palabras_nan.add(word)

                            list_values[i].append(np.nanmean(list_values_element))
                        else:
                            list_values[i].append(next(iter(list(set(data[data["word"] == words][psico_column].values))), np.nan))
                            if str(list_values[i][-1]) == "nan":
                                palabras_nan.add(words)
        return palabras_nan
    
    
    def psycholinguistics_features_synonyms(self, data, psycholinguistics_columns,tokens_columns, df,path, imputed_columns=None):
        '''
        It adds a new column for each psycholinguistics feature in psycholinguistics_columns 
        where it asign to each row the corresponding psycholinguistic values to each token.

        Parameters
        ----------
        data : pandas.DataFrame
            The DataFrame with one row for each different token with the 
            corresponding psycholinguistic values. Obtained from https://www.bcbl.eu/databases/espal/
        psycholinguistics_columns : List of column names
            A list with the names of the columns in data with the 
            psycholinguistic variables.
        columnas_tokens : List of column names
            A list with the names of df where the tokens are.
        df : pandas.DataFrame
            The DataFrame with the lists of tokens.

        Returns
        -------
        df : pandas.DataFrame
            The same DataFrame received but with as many columns more as length of
            psycholinguistics_columns multiplied by length of tokens_columns.
            
        Example:
        Parameters:
            data:
                word|log_frq|sa_num_phon|
                perro|1.85108|4|
                gato|1.59723|4|
                caballo|1.92165|6|
                papa|1.93218|4|
                piso|1.55915|4|
                pintura|1.65155|7|
            psycholinguistics_columns:
                ["log_frq","sa_num_phon"]
            columnas_tokens:
                ["fluency_p","fluency_animals"]
            df:
                codigo|diagnostico|fluency_p|fluency_animals|
                001|AD|["papa","pintura"],["caballo","gato"]
                002|CTR|["piso","pintura","papa"],["perro","gato"]
            
        Returns:
            df:
                codigo|diagnostico|fluency_p|fluency_animals|fluency_p_log_frq|fluency_p_sa_num_phon|fluency_animals_log_frq|fluency_animals_sa_num_phon|
                001|AD|["papa","pintura"]|["caballo","gato"]|[1.93218,1.65155]|[4,7]|[1.92165,1.59723]|[6,4]
                002|CTR|["piso","pintura","papa"]|["perro","gato"]|[1.55915,1.65155,1.93218]|[4,7,4]|[1.55915,1.65155,1.93218]|[4,7,4]
        '''
        
        df_synonyms = self.read_pickle_synonyms_file_fasttext(path)
        new_columns = []

        for i_column, column in enumerate(tokens_columns):
            print(str(i_column) + " de " + str(len(tokens_columns)) + ": " + str(column))
            for i_psico_column,psico_column in enumerate(psycholinguistics_columns):
                print(str(i_psico_column) + " de " + str(len(psycholinguistics_columns)) + ": " + str(psico_column))
                new_column_name = f"{column}_{psico_column}"
                new_column_imputada_name = f"{new_column_name}_imputada"

                new_columns.extend([new_column_name, new_column_imputada_name])

                df[new_column_name] = np.nan
                df[new_column_name] = df[new_column_name].astype(object)

                df[new_column_imputada_name] = np.nan
                df[new_column_imputada_name] = df[new_column_imputada_name].astype(object)
                
                # df[new_column_imputada_lemma_name] = np.nan
                # df[new_column_imputada_lemma_name] = df[new_column_imputada_lemma_name].astype(object)

                # list_values_imputados_lemma = []
                list_values_imputados = []
                list_values = []
                for i_row, row in df.iterrows():
                    if ((i_row%10) == 0):
                        print(str(i_row) + " de " + str(len(df)))

                    # list_values_imputados_lemma.append([])
                    list_values_imputados.append([])
                    list_values.append([])

                    for words in row[column]:
                        # if len(words.split()) > 1:
                        #     list_values_element_imputados = []
                        #     list_values_element = []

                        #     for word in words.split():
                        #         valor = next(iter(set(data.loc[data["word"] == word, psico_column].values)), np.nan)
                        #         list_values_element.append(valor)

                        #         if (column is None) or (column in imputed_columns):
                        #             if str(valor) == "nan":
                        #                 df_fila = df_synonyms[df_synonyms["word"] == word]
                        #                 if df_fila.empty:
                        #                     fila = self.get_synonyms_fasttext(word)
                        #                     df_fila = pd.DataFrame(
                        #                         [[word] + fila], columns=["word"] + ["synonym_" + str(i_syn) for i_syn in
                        #                                                                 range(len(fila))])
                        #                     df_synonyms = pd.concat([df_synonyms, df_fila], ignore_index=True)
                        #                 contador = 1
                        #                 fila = df_fila.values[0]
                        #                 while str(valor) == "nan" and contador < len(fila):
                        #                     valor = next(iter(set(data.loc[data["word"] == fila[contador], psico_column].values)), np.nan)
                        #                     contador += 1

                        #         list_values_element_imputados.append(valor)

                        #     list_values_imputados[-1].append(np.nanmean(list_values_element_imputados))
                        #     list_values[-1].append(np.nanmean(list_values_element))

                        # else:
                            valor = next(iter(set(data.loc[data["word"] == words, psico_column].values)), np.nan)
                            list_values[-1].append(valor)

                            if str(valor) == "nan":
                                df_fila = df_synonyms[df_synonyms["word"] == words]
                                if df_fila.empty:
                                    fila = self.get_synonyms_fasttext(words)
                                    df_fila = pd.DataFrame(
                                        [[words] + fila], columns=["word"] + ["synonym_" + str(i_syn) for i_syn in
                                                                                range(len(fila))])
                                    df_synonyms = pd.concat([df_synonyms, df_fila], ignore_index=True)
                                contador = 1
                                fila = df_fila.values[0]
                                while str(valor) == "nan" and contador < len(fila):
                                    if str(fila[contador]) != "nan":
                                        valor = next(iter(set(data.loc[data["word"] == fila[contador], psico_column].values)), np.nan)
                                    contador += 1
                            list_values_imputados[-1].append(valor)

                df[new_column_imputada_name] = list_values_imputados
                df[new_column_name] = list_values

        df_synonyms.to_pickle(path+"//synonyms_fasttext.pkl")
        return df

    def psycholinguistics_features_optimized(self, data, psycholinguistics_columns, tokens_columns, df):
        # Crear un diccionario para buscar valores de psicolingüística por palabra
        psycholinguistics_dict = {}
        for word in data['word']:
            psycholinguistics_dict[word] = {
                col: data.loc[data['word'] == word, col].values[0]
                for col in psycholinguistics_columns
            }
        
        # Crear nuevas columnas en el DataFrame
        new_columns = []
        total_iterations = len(tokens_columns) * len(psycholinguistics_columns)
        current_iteration = 0
        progress_step = total_iterations // 10  # Imprimir el progreso cada 10% del avance
        
        for column in tokens_columns:
            for psico_column in psycholinguistics_columns:
                new_column_name = f"{column}_{psico_column}"
                df[new_column_name] = df[column].apply(
                    lambda words: [
                        psycholinguistics_dict.get(word, {}).get(psico_column, np.nan)
                        for word in words
                ]
                )
                new_columns.append(new_column_name)
                
                current_iteration += 1
                if current_iteration % progress_step == 0:
                    progress = current_iteration / total_iterations * 100
                    print(f"Progress: {progress:.0f}%")
        
        df[new_columns] = df[new_columns].astype(object)
        
        return df
    def obtain_stadistic_values(self,df,psycholinguistics_columns,tokens_columns,list_statistics = ["promedio","minimo","maximo","std","mediana","curtosis","skewness"]):
        '''
        It obtains the promedio, minimo, maximo, std, mediana, curtosis and skewness
        of each psycholinguistic variable of each fluency task.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame with the psycholinguistic values of each token of
            each fluency task.
        psycholinguistics_columns : List of column names
            A list with the names of the psycholinguistic variables.
        tokens_columns : List of column names
            A list with the names of df where the tokens are.

        Returns
        -------
        df : pandas.DataFrame
            The same DataFrame received but with as many columns more as length of
            psycholinguistics_columns multiplied by length of tokens_columns multiplied
            by the 7 stadistics values (promedio, minimo, maximo, std, mediana, curtosis y skewness.

        '''
        # Inicializa las nuevas columnas
        for column in tokens_columns:
            for psico_column in psycholinguistics_columns:
                if "promedio" in list_statistics:
                    df[column + "_" + psico_column + "_promedio"] = np.nan
                if "minimo" in list_statistics:
                    df[column + "_" + psico_column + "_minimo"] = np.nan
                if "maximo" in list_statistics:
                    df[column + "_" + psico_column + "_maximo"] = np.nan
                if "std" in list_statistics:
                    df[column + "_" + psico_column + "_std"] = np.nan
                if "mediana" in list_statistics:
                    df[column + "_" + psico_column + "_mediana"] = np.nan
                if "curtosis" in list_statistics:
                    df[column + "_" + psico_column + "_curtosis"] = np.nan
                if "skewness" in list_statistics:
                    df[column + "_" + psico_column + "_skewness"] = np.nan
        
        # Función para calcular estadísticas
        def calcular_estadisticas(row, column, psico_column):
            calculation_list = row[column + "_" + psico_column]
            stats = {}
            
            if "promedio" in list_statistics:
                stats["promedio"] = np.nanmean(calculation_list)
            if "minimo" in list_statistics:
                stats["minimo"] = np.nan if len(calculation_list) == 0 else np.nanmin(calculation_list)
            if "maximo" in list_statistics:
                stats["maximo"] = np.nan if len(calculation_list) == 0 else np.nanmax(calculation_list)
            if "std" in list_statistics:
                stats["std"] = np.nanstd(calculation_list)
            if "mediana" in list_statistics:
                stats["mediana"] = np.nanmedian(calculation_list)
            if "curtosis" in list_statistics:
                stats["curtosis"] = kurtosis([x for x in calculation_list if str(x) != 'nan'])
            if "skewness" in list_statistics:
                stats["skewness"] = skew([x for x in calculation_list if str(x) != 'nan'])
                
            return stats
        
        # Aplicar la función a cada fila
        for column in tokens_columns:
            for psico_column in psycholinguistics_columns:
                stats = df.apply(lambda row: calcular_estadisticas(row, column, psico_column), axis=1)
                
                if "promedio" in list_statistics:
                    df[column + "_" + psico_column + "_promedio"] = stats.apply(lambda x: x["promedio"])
                if "minimo" in list_statistics:
                    df[column + "_" + psico_column + "_minimo"] = stats.apply(lambda x: x["minimo"])
                if "maximo" in list_statistics:
                    df[column + "_" + psico_column + "_maximo"] = stats.apply(lambda x: x["maximo"])
                if "std" in list_statistics:
                    df[column + "_" + psico_column + "_std"] = stats.apply(lambda x: x["std"])
                if "mediana" in list_statistics:
                    df[column + "_" + psico_column + "_mediana"] = stats.apply(lambda x: x["mediana"])
                if "curtosis" in list_statistics:
                    df[column + "_" + psico_column + "_curtosis"] = stats.apply(lambda x: x["curtosis"])
                if "skewness" in list_statistics:
                    df[column + "_" + psico_column + "_skewness"] = stats.apply(lambda x: x["skewness"])

        return df
            
    def get_last_txt_download(self,directory):
        files = [f for f in os.listdir(directory) if f.endswith(".txt") and os.path.isfile(os.path.join(directory, f))]
        return max(files, key=lambda x: os.path.getctime(os.path.join(directory, x)))
    
    def espal_feature_extraction(self, words_path,download_path,save_path,list_psycho):
        """
        Extracts linguistic features from a list of words using the ESPAL database.

        Args:
        - words_path (str): The path to a text file containing a list of words to extract features from.
        - download_path (str): The path to the folder where the downloaded files will be stored.
        - save_path (str): The path to the folder where the output files will be saved.

        Returns:
        None.

        This function automates the process of navigating the ESPAL database website to extract various 
        linguistic features for a list of words. The function starts a new Edge browser session, navigates 
        to the ESPAL website, selects various feature checkboxes, uploads a list of words, downloads the feature data, 
        and saves it to a specified folder. The function requires an internet connection and the Edge browser driver 
        to be installed on the local system.
        """

        # start a new Edge browser session
        browser = webdriver.Edge()
        # wait for the file to download
        time.sleep(1)

        # navigate to the target website
        browser.get("https://www.bcbl.eu/databases/espal/")

        time.sleep(5)

        #wait = WebDriverWait(browser, 10)

        element = browser.find_element(By.XPATH, "//form[@name='phonologyForm']/h4[2]/input[@type='radio']")
        element.click()

        # Find the first button in the intro_menu class
        first_button = browser.find_element(By.XPATH, "//div[@class='intro_menu']/p[1]/a")

        # Click on the first button
        first_button.click()

        # wait for the new page to load
        time.sleep(5)

        #### check the log_frq checkbox

        if "log_frq" in list_psycho:
            container = browser.find_element(By.XPATH, '//div[@id="select"]//div[@class="idx_group"]//a[@class="btn_group"][@id="switch1"]')
            container.click()
    
            time.sleep(1)
    
            # locate the checkbox element by its id attribute
            checkbox = browser.find_element(By.XPATH, "//input[@type='checkbox' and @id='log_frq']")
    
            # check the checkbox
            checkbox.click()
    
            time.sleep(1)

        if ("sa_num_phon" in list_psycho) or ("sa_num_syll" in list_psycho):
            #### check the number of phonemes and number of syllabes checkbox
            # expando el contenedor
            container = browser.find_element(By.XPATH, '//div[@id="select"]//div[@class="idx_group"]//a[@class="btn_group"][@id="switch4"]')
            container.click()
    
            time.sleep(1)
    
            if ("sa_num_phon" in list_psycho):
                # locate the checkbox element by its id attribute
                checkbox = browser.find_element(By.XPATH, "//input[@type='checkbox' and @id='num_phon']")
        
                # check the checkbox
                checkbox.click()
                
            if ("sa_num_syll" in list_psycho):
                # locate the checkbox element by its id attribute
                checkbox = browser.find_element(By.XPATH, "//input[@type='checkbox' and @id='num_syll']")
        
                # check the checkbox
                checkbox.click()
    
            time.sleep(1)

        if ("sa_NP" in list_psycho):
            #### check the number of phonological neighborhoods checkbox
            # expando el contenedor
            container = browser.find_element(By.XPATH, '//div[@id="select"]//div[@class="idx_group"]//a[@class="btn_group"][@id="switch5"]')
            container.click()
    
            time.sleep(1)
    
            # locate the checkbox element by its id attribute
            checkbox = browser.find_element(By.XPATH, "//input[@type='checkbox' and @id='NP']")
    
            # check the checkbox
            checkbox.click()
    
            time.sleep(1)

        if ("familiarity" in list_psycho) or ("imageability" in list_psycho) or ("concreteness" in list_psycho):
            #### check the all subjective ratings checkbox
            # expando el contenedor
            container = browser.find_element(By.XPATH, '//div[@id="select"]//div[@class="idx_group"]//a[@class="btn_group"][@id="switchSubjective"]')
            container.click()
    
            time.sleep(1)
    
            if ("familiarity" in list_psycho):
                # locate the checkbox element by its id attribute
                checkbox = browser.find_element(By.XPATH, "//input[@type='checkbox' and @id='familiarity']")
                checkbox.click()
            
            if ("imageability" in list_psycho):
                # locate the checkbox element by its id attribute
                checkbox = browser.find_element(By.XPATH, "//input[@type='checkbox' and @id='imageability']")
                checkbox.click()
    
            if ("concreteness" in list_psycho):
                # locate the checkbox element by its id attribute
                checkbox = browser.find_element(By.XPATH, "//input[@type='checkbox' and @id='concreteness']")
                checkbox.click()
            time.sleep(1)


        #### upload the .txt file
        # locate the file input element by its name attribute
        file_input = browser.find_element(By.XPATH, "//input[@type='file' and @name='words_file']")

        # send the file path to the file input element
        file_input.send_keys(words_path)

        # submit the form
        file_input.submit()

        # wait for the file to download
        time.sleep(5)


        ########## Entro a la página de descarga

        # locate the download button element and click it
        download_link = browser.find_element(By.XPATH,"//td/a[@class='btn_download']")
        download_link.click()

        # wait for the file to download
        time.sleep(5)

        # locate the downloaded file
        last_txt_download = self.get_last_txt_download(download_path)

        # move the downloaded file to the desired path
        desired_path = os.path.expanduser(save_path)
        shutil.move(download_path+"/"+last_txt_download, desired_path)

        # close the browser
        browser.quit()
        
        
    def get_synonyms_fasttext(self, palabra):
        
        synonyms = self.fasttext_reduced.get_nearest_neighbors(palabra, k=20)
        synonyms = [tupla[1] for tupla in synonyms]
        return synonyms
        

    def get_synonyms_wordreference(self, palabra):
        url='https://www.wordreference.com/sinonimos/'
        palabra = palabra.replace('\n','')
        buscar=url+palabra
        resp=requests.get(buscar)
        bs=BeautifulSoup(resp.text,features="html.parser")
        lista = bs.find(class_="trans esp clickable")
        list_sinonimos = []
        if not isinstance(lista,type(None)):
            sinos=str(lista.find('li')).split(',')
            if len(sinos) != 0:
                for i_sino , sino in enumerate(sinos):
                    sinonimo = sino.replace('<li>','').replace('</li>','')
                    sinonimo = sinonimo.replace(" ", "")
                    list_sinonimos.append(sinonimo)
        return list_sinonimos
    
    def load_hugging_translation_model_es_en(self,path):
        cache_directory = path
        model_name = "Helsinki-NLP/opus-mt-es-en"
    
        self.translation_hugging_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_directory)
        self.translation_hugging_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_directory)
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.translation_hugging_model.to(self.device)
            
    def load_hugging_translation_model_chino_english(self,path):
        cache_directory = path
        model_name = "Helsinki-NLP/opus-mt-zh-en"
    
        self.translation_hugging_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_directory)
        self.translation_hugging_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_directory)
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.translation_hugging_model.to(self.device)
    
    def translate_hugging_es_en(self,text):
        input_ids = self.translation_hugging_tokenizer.encode(text, return_tensors="pt")
        outputs = self.translation_hugging_model.generate(input_ids)
        translated_text = self.translation_hugging_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text
    
    def translate_multiple_hugging_es_en(self, texts):        
        max_new_tokens = 512
        batch_division_factor = 1
        
        input_ids = self.translation_hugging_tokenizer.batch_encode_plus(texts, return_tensors="pt", padding=True)["input_ids"]
        
        if torch.cuda.is_available():
            input_ids = input_ids.to(self.device)
        
        batch_size = len(input_ids)
        
        while True:
            try:
                batch_size_divided = batch_size//batch_division_factor
                if batch_size < 1:
                    break

                input_ids_batches = input_ids.split(batch_size_divided)
                combined_outputs = []
                for batch in input_ids_batches:
                    batch_outputs = self.translation_hugging_model.generate(batch, max_new_tokens=max_new_tokens)
                    combined_outputs.append(batch_outputs)
                
                outputs = torch.cat(combined_outputs, dim=1)
                translated_texts = self.translation_hugging_tokenizer.batch_decode(outputs, skip_special_tokens=True)

                return translated_texts
        
            except RuntimeError:
        
                batch_division_factor = batch_division_factor*2
                
    def custom_sent_tokenize(self, text):
        # Divide el texto por puntos
        sentences_by_period = [sentence.strip() for sentence in text.split('.')]
    
        # Divide cada oración por saltos de línea
        sentences = []
        for sentence in sentences_by_period:
            sentences.extend(sentence.split('\n'))

        return sentences
    
    def concatenar_por_numero(self, textos, numeros):
        # Creamos un diccionario para agrupar los textos por número
        grupos = {}
        for texto, numero in zip(textos, numeros):
            if numero not in grupos:
                grupos[numero] = []
            grupos[numero].append(texto)
    
        # Concatenamos los textos de cada grupo separados por espacio
        textos_concatenados = []
        for numero in np.unique(numeros):
            textos_grupo = grupos[numero]
            texto_concatenado = ' '.join(textos_grupo)
            textos_concatenados.append(texto_concatenado)
    
        return textos_concatenados
    
    def translate_text(self,texts):

        print("Numero de textos: " + str(len(texts)))
        translated_segments_concatenated = []
        for i_text,text in enumerate(texts):
            texts_less_limit = []
            # Dividir el texto en oraciones utilizando NLTK
            sentences = self.custom_sent_tokenize(text)

            sentences = [sentence for sentence in sentences if sentence!=""]
            for i_sentence, sentence in enumerate(sentences):
                
                sentence_split = sentence.split()
                sentence_split = [i for i in sentence_split if len(i) <= 100]

                # Verificar si la oración supera el límite de tokens
                if len(sentence_split)!=0:
                    if len(sentence_split) > 256:
                        # Dividir la oración en segmentos más pequeños
                        segments = [sentence[i:i + 256] for i in range(0, len(sentence), 256)]
    
                        texts_less_limit = texts_less_limit + segments
                    else:
                        texts_less_limit.append(" ".join(sentence_split))

            # Traducir cada segmento individualmente
            translated_segments = self.translate_multiple_hugging_es_en(texts_less_limit)
            
            translated_sentence = ' '.join(translated_segments)

            translated_segments_concatenated.append(translated_sentence)
            # Concatenar los segmentos traducidos en una sola respuesta

        return translated_segments_concatenated
    
    def translate_text_chinese_english(self,texts):

        translations = []
        
        for text in texts:
            tokenized_text = self.translation_hugging_tokenizer.prepare_seq2seq_batch([text], return_tensors='pt')
            translation = self.translation_hugging_model.generate(**tokenized_text)
            translated_text = self.translation_hugging_tokenizer.batch_decode(translation, skip_special_tokens=False)[0]
            translations.append(translated_text)
        return translations
            
