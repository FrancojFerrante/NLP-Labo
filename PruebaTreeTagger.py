# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:32:10 2021

@author: fferrante
"""

from treetagger import TreeTagger

tt = TreeTagger(path_to_treetagger='C:\TreeTagger',language="spanish")

tt.get_treetagger_path()
resultado = tt.tag("Querés tomar mate?")

# print(tt.tag("Does this thing even work?"))