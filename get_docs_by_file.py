# The first main function retrieves documents one by one
# retrieve docs from the Indri index given a file with a list of documents.
# documents from each query should be headed by 'qno:<query number>'
# usage: python <pmidfile> <outputfile> <Indri index directory>

# The second main function can be used to retrieve multiple documents text at a time 
# In order to use the second main function, the dumpindex.cpp file as part of Indri Build Index need to be changed
# The modified file of dumpindex.cpp is also provided along with other code files 

# Krishna Sirisha Motamarry & Lowell Milliken

import subprocess
import sys
import os

def main(doclist, outfilename='docs.txt', index='medline-ja2018-index'):
    docnos = load_pmids(doclist)
    os.chdir("/Users/lowellmilliken/Downloads/indri-5.14/dumpindex/")
    owd = os.getcwd()
    print(owd)
    with open(outfilename, 'w') as docs:
        for queryno, docsids in docnos.items():
            print('writing qno:{}'.format(queryno))
            docs.write('<QUERY qno=\'{}\'>\n'.format(queryno))
            
            for docno in docsids:
                diprocess = subprocess.Popen(['./dumpindex', index, 'di', 'docno', docno],
                                             stdout=subprocess.PIPE)
                output, err = diprocess.communicate()
                dtprocess = subprocess.Popen(['./dumpindex', index, 'dt', output], stdout=subprocess.PIPE,
                                             universal_newlines=True)
                output, err = dtprocess.communicate()
                docs.write(str(output) + str('\n'))

            docs.write('</QUERY>')

    print('done')


"""
def main(doclist, outfilename='docs.txt', index='medline-ja2018-index-final2'):
    docnos = load_pmids(doclist)
    os.chdir("/Users/lowellmilliken/Downloads/indri-5.14/dumpindex")
    with open(outfilename, 'w') as docs:
        for queryno, docsids in docnos.items():
            print('writing qno:{}'.format(queryno))
            docs.write('<QUERY qno=\'{}\'>\n'.format(queryno))
            #cmd= ['./dumpindex', index, 'di', 'docno']
            cmd= ['./dumpindex', index, 'doct', 'docno']
            for ids in docsids:
                cmd.append(ids)
            diprocess = subprocess.Popen(cmd,stdout=subprocess.PIPE,universal_newlines=True)
            output, err = diprocess.communicate()
            docs.write(str(output) + str('\n'))

            docs.write('</QUERY>')

    print('done')
"""
def load_pmids(doclist):
    docnos = {}

    with open(doclist, 'r') as docfile:
        for line in docfile:
            if line.startswith('qno'):
                qno = line.split(':')[1].strip()
                docnos[qno] = []
            else:
                docnos[qno].append(line.strip())

    return docnos


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
