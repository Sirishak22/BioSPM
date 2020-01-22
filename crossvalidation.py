#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:17:46 2019

@author: Sirisha
"""

import random
import learning_to_rank as l2r
import pickle
import os
import parse_results_for_top_N

from learning_to_rank import load_indriscores

cv_dir = 'cv_files_featuresmodifiedfinal5'
# m - meta
# sd - splitdrugs
# f - filter
# t - target
# jd - journal disease
# tl - text length
# is - indri scores
features_template = 's_{}_known_features_{}'
unknown_template = 's_{}_{}_unknown_features_{}'
model_name = '/Users/lowellmilliken/Documents/precision_medicine_contd/lmillik-artpm-c576ced69e03/'+cv_dir + os.sep + '{}_{}_model_{}'
score_filename = '/Users/lowellmilliken/Documents/precision_medicine_contd/lmillik-artpm-c576ced69e03/'+cv_dir + os.sep + '{}_{}_model_{}scores'


def gen_cv_sets():
    cv_sets = [1]*8 + [2]*8 + [3]*8 + [4]*8 + [5]*8 + [6]*8 + [7]*8 + [8]*8 + [9]*8
    random.seed()
    random.shuffle(cv_sets)
    print("cv_sets")
    print(cv_sets)
    return cv_sets


# ListNet rparams = {'-lr': 0.1, '-epoch': 3000}
def do_cv(unknown_docs_filename='topics2017_m_as_ex_tr_nd_ft_nsh_prf-2-20-0.5-0.5_large_gfix_alldocs.txt',
           metric='P@10', program='RankLib', indriscore=False, otherscore=False, ranker='ListNet', rparams=None,
          kscorefile='topics2017_m_as_ex_tr_nd_ft_nsh_prf-2-10-0.5-0.8_basescores_large_gfix_run.txt',
          scorefile='topics2017_m_as_ex_tr_nd_ft_nsh_prf-2-10-0.5-0.8_ob-topics2017_m_as_ex_tr_nd_ft_nsh_prf-2-20-0.5-0.5_large_gfix_large_gfix_run.txt',
          fixparts=True, normscores=False, intval=True,unknownscoresfilename=None,trainallparam=True,testallparam=False,featurefile=None):
    """Creates a ranked list output file in TREC format doing training and cross validation for LeToR.

    :param unknown_docs_filename: name of file containing abstracts from the current Retrieval stage run
    :param meta: Boolean. Use metamap CUIs or not. Requires unknown_docs_filename + '.meta' file containing CUIs for each abstract.
    :param splitdrugs: split drugs into multiple features?
    :param metric: metric to train on. See RankLib help for options.
    :param program: Program to do LeToR with. Default RankLib.
    :param filtered: Filter CUIs to use with meta option. Requires either fterms.pickle or terms_filtered.pickle (for phraseterms) file.
    :param targetproxy: Use proximity to the work 'target' as a feature.
    :param dist: distance threshold for 'target' proximity
    :param journaldisease: Use disease presence in journal name as a feature.
    :param textlen: Use abtract length as a feature.
    :param indriscore: Use the indri score as a feature. Requires Indri scores for the qrel documents called unknown_docs_filename[:-11] + basescores_run.txt and a Indri results file called unknown_docs_filename[:-11] + run.txt
    :param otherscore: Use tf-idf and bm25 scores as a feature. Requires 'qrel_tfidfbase_run.txt' and 'qrel_bm25base_run.txt' as well as unknown_docs_filename[:-11] + 'tfidfbase_run.txt' and unknown_docs_filename[:-11] + 'bm25base_run.txt'
    :param ranker: LeToR ranker to use. See RankLib help for options.
    :param rparams: LeToR parameters in a dictionary. See RankLib help for options. Parameter name including leading '-' is key and parameter value is value.
    :param kscorefile: Alternate score file for use as a feature. This should be scores for the known qrels for training.
    :param scorefile: Alternate score file for use as a feature. This should be scores for the unknown documents for testing.
    :param fixparts: Boolean. Fixed cross-valiation partitions if True.
    :param normscores: Boolean. If True, Indri scores are normalized by (score - minscore)/(maxscore - minscore). Using the '-norm' in rparams with a norm type is preferred. See RankLib help.
    :param phraseterms: Boolean. Use only metamapped CUI terms from original terms that are not unigrams.
    :param intval: Boolean. Use RankLib internal validation. True preferred.
    :param termfile: Explicit set of CUI terms to use. A list in a pickle file.
    :param termkeyfile: Keys for mapping terms in the term file to features. Dict in a pickle file. Key = term. Value = term number (which maps to a feature number).
    :param nodrugs: Boolean. If True, do not use any drug information as a feature.
    """
    unknown_base = unknown_docs_filename[:-11]
    parastr = 'n'

    if indriscore:
        parastr += '_is'
    if otherscore:
        parastr += '_os'
    if normscores:
        parastr += '_ns'

    if scorefile:
        parastr += '_sf'
    if not intval:
        parastr += '_nov'

    

    meta_docs = None
    unknown_meta_docs = None

    if indriscore:
        #basescores = load_indriscores('/Users/lowellmilliken/Downloads/indri-5.14/runquery/outputcrossval_trainbasescores17nd18.txt')
        #unknownscores = load_indriscores('/Users/lowellmilliken/Downloads/indri-5.14/runquery/final5basescorestestmodified.txt')
        #with newest query formulation added
        basescores = load_indriscores('/Users/lowellmilliken/Downloads/indri-5.14/runquery/output_traincrossvalfinal9basescores.txt')
        unknownscores = load_indriscores('/Users/lowellmilliken/Downloads/indri-5.14/runquery/output_testcrossvalfinal9basescores.txt')
    else:
        basescores = None
        unknownscores = None

    if otherscore:
        basetfidfscores = parse_results_for_top_N.load_indri_tfidf_scores(res_filename='qrel_tfidfbase_run.txt')
        basebm25scores = parse_results_for_top_N.load_indri_tfidf_scores(res_filename='qrel_bm25base_run.txt')

        #with newest query formulation added
        unknowntfidfscores = parse_results_for_top_N.load_indri_tfidf_scores(res_filename='/Users/lowellmilliken/Downloads/indri-5.14/runquery/output_testcrossvalfinal9_tfidf_run.txt')
        unknownbm25scores = parse_results_for_top_N.load_indri_tfidf_scores(res_filename='/Users/lowellmilliken/Downloads/indri-5.14/runquery/output_testcrossvalfinal9_bm25_run.txt')

    else:
        basetfidfscores = None
        basebm25scores = None

        unknowntfidfscores = None
        unknownbm25scores = None

    if scorefile:
        kprecscores = load_indriscores(kscorefile, normscores)
        precscores = load_indriscores(scorefile, normscores)
    else:
        kprecscores = None
        precscores = None

    if trainallparam==True:
        topics = l2r.load_topics(topicfile='crossvalidationtopics.xml')
        train_all = cv_dir + os.sep + features_template.format(parastr, 'all')
    

    # if not os.path.exists(train_all) or indriscore:
        known_docs = l2r.load_docs()
    
        l2r.save_all_features(topics, known_docs, train_all, known=True, metadocs=meta_docs,
                              scores=basescores, tfidfscores=basetfidfscores, bm25scores=basebm25scores,
                              precscores=kprecscores)
    if testallparam==True:
       topics = l2r.load_topics(topicfile='topic6.xml')
    # if not os.path.exists(test_all):
       test_all = cv_dir + os.sep + unknown_template.format(unknown_base, parastr, 'all')
       unknown_docs=l2r.load_docs('testdatadocs.txt')
       l2r.save_all_features(topics, unknown_docs, test_all, known=False, metadocs=unknown_meta_docs,scores=unknownscores, tfidfscores=unknowntfidfscores, bm25scores=unknownbm25scores,
                          precscores=precscores)

    cv_file = cv_dir + os.sep + 'cv_sets.txt'
    if fixparts and os.path.exists(cv_file):
        cv_sets = []
        with open(cv_file, 'r') as cvsetfile:
            for line in cvsetfile:
                cv_sets.append(int(line.strip()))
    else:
        cv_sets = gen_cv_sets()
        with open(cv_file, 'w') as cvsetfile:
            for i in cv_sets:
                cvsetfile.write('{}\n'.format(i))

    #all_qnos = list(range(1, 31))
    all_qnos = [1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 79, 80]
    qscores ={}
    pmids = {}
    train_all='/Users/lowellmilliken/Documents/precision_medicine_contd/lmillik-artpm-c576ced69e03/cv_files_featuresmodifiedfinal5/s_n_is_os_known_features_all'
    test_all='/Users/lowellmilliken/Documents/precision_medicine_contd/lmillik-artpm-c576ced69e03/cv_files_featuresmodifiedfinal5/s_modifiedfeature_n_os_unknown_features_all_1500docs'
    for i in range(1, 10):
        training_set=[]
        test_set=[]
        index1=[]
        index2=[]
        model_file = model_name.format(parastr, ranker, i)

        train_filename = '/Users/lowellmilliken/Documents/precision_medicine_contd/lmillik-artpm-c576ced69e03/'+cv_dir + os.sep + features_template.format(parastr, i)
        test_filename = '/Users/lowellmilliken/Documents/precision_medicine_contd/lmillik-artpm-c576ced69e03/'+cv_dir + os.sep + unknown_template.format(unknown_base, parastr, i)
        index1= [j for j, e in enumerate(cv_sets) if e != i]

        for x in index1:
            a=all_qnos[x]
            training_set.append(str(a))

        index2= [k for k, e in enumerate(cv_sets) if e == i]

        for x in index2:
            a=all_qnos[x]
            test_set.append(str(a))


        filter_file(train_all, train_filename, training_set)
        filter_file(test_all, test_filename, test_set)
   
        # if not os.path.exists(model_file) or indriscore:
        l2r.train_model(train_filename, model_file, ranker=l2r.rankers[ranker], metric=metric, program=program, params=rparams, validation=intval, featurefile=featurefile)

        l2r.predict(model_file, test_filename, score_filename.format(parastr, ranker, i), metric=metric, program=program, params=rparams, featurefile=featurefile)

        if program == 'RankLib':
            qscores.update(l2r.load_rankings(score_filename.format(parastr, ranker, i)))
            pmids.update(l2r.load_pmids_from_features(test_filename))
        elif program == 'Quickrank':
            qpmids = l2r.load_pmids_from_features(test_filename)
            qscores.update(l2r.load_quickrank_scores(qpmids, score_filename.format(parastr, ranker, i)))
            pmids.update(qpmids)

    runfilename = '/Users/lowellmilliken/Downloads/trec_eval.9.0/'+unknown_base + 'tvs_L2R_{}_{}_{}_run.txt'.format(ranker, metric, parastr)
    l2r.save_reranked(qscores, pmids, runfilename)

    return runfilename


# create new file with only docs
def filter_file(infilename, outfilename, filter_):
    count = 0
    print('')
    with open(infilename, 'r') as infile, open(outfilename, 'w') as outfile:
        for line in infile:
            count += 1
            print('\rFiltering on {}'.format(count), end='')
            qno = line.split()[1].split(':')[1]
            if qno in filter_:
                outfile.write(line)

