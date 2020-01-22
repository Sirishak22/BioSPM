#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 16:14:38 2019
# Extracts the qrels from the qrel file and creates a pmid file, extracts documents with respects of the pmids
# These document file can be used for feature generation
@author: Krishna Sirisha Motamarry
"""

import find_qrels
qrels=find_qrels.load_qrels('/Users/lowellmilliken/Documents/precision_medicine_contd/lmillik-artpm-c576ced69e03/qrels-final-abstracts-2017.txt')
find_qrels.save_pmids(qrels,'pmidfor2017qrels.txt')
import xml_to_params
xml_to_params.main(filename='topicsonly2017.xml', baseline=True, pmidfile='pmidfor2017qrels.txt')
import get_docs_by_file
get_docs_by_file.main('/Users/lowellmilliken/Documents/precision_medicine_contd/lmillik-artpm-c576ced69e03/pmidfor2017qrels.txt', '/Users/lowellmilliken/Documents/precision_medicine_contd/lmillik-artpm-c576ced69e03/qrel_docs_2017.txt', '/Users/lowellmilliken/Documents/precision_medicine_contd/indexes/medline-ja2018-index-final2')
