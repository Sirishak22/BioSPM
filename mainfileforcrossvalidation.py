#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:57:31 2019

@author: Krishna Sirisha Motamarry
"""
import extract_stat as es
import crossvalidation as val
#import train2017test2018 as val
import os

filename=val.do_cv(unknown_docs_filename='newqueryfeatremoved.txt',otherscore=True,kscorefile=None,scorefile=None,ranker='ListNet',metric='NDCG@10',trainallparam=False,testallparam=False,rparams={'-lr': 0.1, '-epoch': 3000},featurefile='/Users/lowellmilliken/Documents/precision_medicine_contd/lmillik-artpm-c576ced69e03/featureabqueryfestrem.txt')
print(filename)
es.main(filename)
