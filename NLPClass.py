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

class NLPClass:
    def __init__(self):
        self.numero = 1
        
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
      each string is a token obteined from apply the tokenizer_type.

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
      tokens = [tokenizer(s) for s in text_dataframe]
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
            translated_objects.append(translator.translate(element.replace("-"," "), src=lan_src, dest=lan_dest))
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
                    translation_object = nlp_class.translate([text]) # Get the translation_object with all posible translations
                    while (not has_hyper):
                        translated_synsets = []
                        while (len(translated_synsets)==0):
                            iter_translates+=1
                            translated_word = translation_object[0].extra_data["parsed"][1][0][0][5][0][4][iter_translates][0].lower() # Extract a posible translation
                            translated_synsets = wn.synsets(translated_word.replace(" ","_"))
                            translated_synsets = [x for x in translated_synsets if (".n.") in x.name().lower()] # keep nouns only
                        if (hypernym_check != ''):
                            synset_with_hypernym, _ = nlp_class.get_synset_that_has_hypernym(translated_synsets, hypernym_check = hypernym_check) # check if hypernym_check is part of translated_synsets hypernym tree
                            if (synset_with_hypernym is not None):
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
        total_hypernyms = synset.hypernym_paths()
        for hypernyms in total_hypernyms:
            hypernyms_to_destiny = []
            for hypernym in hypernyms:
                hypernyms_to_destiny.append(hypernym)
                if (hypernym_destiny == hypernym.name()):
                    return hypernyms_to_destiny
        return None

    def get_synset_that_has_hypernym(self, synsets, hypernym_check = "animal.n.01"):
        """
        It receives a list of synsets and return the first synset that has the
        hypernym_check in its hypernyms tree and the synset's hypernyms

        Parameters
        ----------
        synsets : synset list
            A list with synsets.
        hypernym_check : string, optional
            The synset to be searched in the hypernyms tree. 
            The default is "animal.n.01".

        Returns
        -------
        synset_with_hyper : synset
            The first synset whose hypernyms tree contains the hypernym_check.
        hypernyms : synset list
            The synset_with_hyper's hypernyms tree.

        """
        synset_with_hyper = None
        hypernyms = None
        for i in range(0,len(synsets)):
            hypernyms = self.get_hypernyms_to(synsets[i], hypernym_destiny = hypernym_check)
            if (hypernyms is not None):
                synset_with_hyper = synsets[i]
                break
        return synset_with_hyper, hypernyms
    
    def hypernym_min_nodes_distance_from_synset_to_hypernym(self, word, hypernym_check = "animal.n.01"):
        """
        It calculates the number of nodes from the word synset to the hypernym_check 
        in the hypernyms tree.

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
        min_distancia = 100000
        translated_synsets = wn.synsets(word.replace(" ","_"))
        synset_with_hypernym, _ = self.get_synset_that_has_hypernym(translated_synsets, hypernym_check = hypernym_check)
        if synset_with_hypernym is not None:
            total_hypernyms = synset_with_hypernym.hypernym_paths()
            for hypernyms in total_hypernyms:
                distancia = 0
                if wn.synset(hypernym_check) in hypernyms:
                    hypernyms.reverse()
                    for hypernym in hypernyms:
                        if wn.synset(hypernym_check) != hypernym:
                            distancia +=1
                        else:
                            distancia +=1
                            break
                    if distancia<min_distancia:
                        min_distancia = distancia
            return min_distancia
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
        summation = sum((vector_distances-np.mean(vector_distances))*(vector_distances-np.mean(vector_distances)))
        average = summation/(len(vector_distances)-1)
        return average
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        