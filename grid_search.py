######################################
# Quick and dirty grid searches for LeToR hyperparameters, not intended for extensive use
# Lowell Milliken
######################################
import crossvalidation as val
import extract_stat as es
#'0.000001',
import os
def do_search():
    # ranker = 'Random Forests'
    ranker = 'ListNet'
    stats = ('P_10', 'recall_1000', 'ndcg_cut_1000', 'map_cut_1000')
    outfilename = 'LN_grid_search_extra_zscoresum.csv'

    with open(outfilename, 'w') as outfile:
        outfile.write('learning rate,epoch,norms,' + ','.join(stats) + '\n')
    #     outfile.write('frate,tree,leaf,shrinkage,norm,' + ','.join(stats) + '\n')

    # frate = 0.2
    nscores = False
    for norm in ('zscore', 'sum'):
        # for frate in (0.3, 0.5, 0.8):
        #     for tree in (1,5,10):
        #         for leaf in (50, 100, 150, 200):

        # tree = 1
        # leaf = 50
        for lr in ('0.00001', '0.0001', '0.001', '0.01', '0.1', '1', '10', '100'):
            for epoch in (1000, 2000, 3000):
                    # params = {'-frate': frate, '-tree': tree, '-leaf': leaf, '-shrinkage': '0.01'}
                    params = {'-lr': lr, '-epoch': epoch}
                    if norm != 'none':
                        params['-norm'] = norm
                    os.chdir('/Users/lowellmilliken/Documents/precision_medicine_contd/lmillik-artpm-c576ced69e03')
                    runfilename = val.do_cv(unknown_docs_filename='LNgridsearch',indriscore=False,otherscore=False,kscorefile=None,scorefile=None,ranker='ListNet',metric='P@10',trainallparam=False,testallparam=False,rparams=params,featurefile='/Users/lowellmilliken/Documents/precision_medicine_contd/lmillik-artpm-c576ced69e03/features1to190.txt')

                    statsfile,statsfile2 = es.create_stats(runfilename)
                    avgstats = es.get_stats(statsfile, stats)

                    with open(outfilename, 'a+') as outfile:
                        # outfile.write('{},{},{},{},{},'.format(frate, tree, leaf, 0.01, norm) + ','.join([avgstats[stat] for stat in stats]) + '\n')
                        outfile.write('{},{},{},'.format(lr, epoch, norm) + ','.join(
                            [avgstats[stat] for stat in stats]) + '\n')


if __name__ == '__main__':
    do_search()