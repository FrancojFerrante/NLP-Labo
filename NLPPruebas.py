# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 15:43:43 2021

@author: franc
"""

# from nltk.corpus import wordnet as wn

# def hiperonimo_cadena(word, number):
#     global contador
#     if (word=='canine'):
#         contador = contador + 1
#         return
#     for ss in wn.synsets(word):
#         print("hiperonimo: " + str(number) + " -",ss)
#         for hyper in ss.hypernyms():
#             print("hiponimo: " + str(number+1) + " -",hyper)
#             hiperonimo_cadena(hyper.name().split('.')[0],number+1)
        


# contador = 0
# # hiperonimo_cadena('dog', 0)



# dog = wn.synset('chimpanzee.n.01')
# # hypo = lambda s: s.hyponyms()
# hyper = lambda s: s.hypernyms()
# # list(dog.closure(hypo, depth=1)) == dog.hyponyms()
# list(dog.closure(hyper, depth=1)) == dog.hypernyms()
# print(list(dog.closure(hyper)))

import nltk
import os

cwd = 'D://Franco//Doctorado//Laboratorio//NLP'
path = cwd + "//Scripts//MCR//wordnet_spa"
wncr = nltk.corpus.reader.wordnet.WordNetCorpusReader(path, None)
palabra = wncr.synset("perro.n.01")

hyper = lambda s: s.hypernyms()
list(palabra.closure(hyper, depth=1)) == palabra.hypernyms()
print(list(palabra.closure(hyper)))




