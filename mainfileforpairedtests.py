#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:14:44 2019

@author: lowellmilliken
"""

import extract_stat as es
import eval_util as eva
es.all_paired(directory='/Users/lowellmilliken/Downloads/trec_eval.9.0/modelcomparisonstats',qfilename='/Users/lowellmilliken/Downloads/trec_eval.9.0/modelcomparisonstats/modelcomparisonquerystats.csv')
eva.do_all_t(outfilename='/Users/lowellmilliken/Downloads/trec_eval.9.0/modelcomparisonstats/modelcomparisonquerypairedstats.csv',statsfile='/Users/lowellmilliken/Downloads/trec_eval.9.0/modelcomparisonstats/modelcomparisonquerystats.csv')
#es.all_paired(directory='/Users/lowellmilliken/Downloads/trec_eval.9.0/featureablationstats',qfilename='/Users/lowellmilliken/Downloads/trec_eval.9.0/featureablationstats/featureablationquerystats.csv')
#eva.do_all_t(outfilename='/Users/lowellmilliken/Downloads/trec_eval.9.0/featureablationstats/featureablationquerypairedstats.csv',statsfile='/Users/lowellmilliken/Downloads/trec_eval.9.0/featureablationstats/featureablationquerystats.csv')