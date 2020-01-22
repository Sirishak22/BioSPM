
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:28:27 2019

@author: Krishna Sirisha Motamarry
"""
import xml_to_params
#If demographics and drug expansion is required
# Input file is topics file
xml_to_params.main(filename='crossvalidationtopics.xml',isdemo=True,isDrug=True)
#If demographics and drug expansion is not required
xml_to_params.main(filename='crossvalidationtopics.xml')
#For tfidf and bm25 scores use the below parameters
#xml_to_params.main(filename='crossvalidationtopics.xml', baseline=True, pmidfile='pmidforcrossvalidationsetqrels.txt')