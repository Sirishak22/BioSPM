#!/usr/bin/env python3


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 21:18:46 2019


"""

import xml.etree.ElementTree as ET
import sys
import gzip
import os
import pymetamap
import time
import re
import json
import timer
from multiprocessing import Pool
from Concept import Corpus
import subprocess
import tempfile


def main(xmlfile = 'medline17n0889.xml.gz'):


  #mm_loc = '/Users/lowellmilliken/Documents/precision_medicine/public_mm/bin/metamaplite.sh'
  #mm = pymetamap.MetaMap.get_instance(mm_loc)
  
  print('Uncompressing\n')
  uncompressed_file = gzip.open(xmlfile)

  try:
    print('Parsing XML tree\n')
    tree = ET.parse(uncompressed_file)
    root = tree.getroot()
  except OSError:
    print('could not read xml.gz file')
    return
  finally:
    print('Closing uncompressed file\n')
    uncompressed_file.close()

  sources = []
  source_ids = []
  #start = time.time()
  #start_str = time.strftime('%a-%d-%b-%Y-%H-%M-%S',time.localtime(start))
  newtimer = timer.Timer()
  newtimer.start()
  count = 0
  print('Extracting text, title, and keywords\n')
  for child in root:
    medlinecitation = child.find('MedlineCitation')
    pmid = medlinecitation.find('PMID').text
    article = medlinecitation.find('Article')
    title = article.find('ArticleTitle').text
    abstractchild = article.find('Abstract')

    if abstractchild:
      abstract = ''
      for textchild in abstractchild:
        if textchild.tag == 'AbstractText':
          if textchild.text:
            abstract += ' ' + textchild.text
    else:
      abstract = title

    print(pmid)

    if abstract:
      abstract = abstract.strip()
      abstract = abstract.lstrip('.')
      abstract = abstract.encode('ascii', errors='ignore').decode()
      mm_abstract = abstract

      sources.append(mm_abstract)
      source_ids.append('{}:TEXT_MM'.format(pmid))

    if title:
      title = title.strip()
      title = title.encode('ascii', errors='ignore').decode()
      sources.append(title)
      source_ids.append('{}:TITLE_MM'.format(pmid))

    count += 1
  
  #tot = len(source_ids)
  #i=0
  concepts_dict = {}
  input_file = tempfile.NamedTemporaryFile(mode="wb", delete=False)
  output_file = tempfile.NamedTemporaryFile(mode="r", delete=False)
  filename = input_file.name
  outputfilename = output_file.name
  for identifier, sentence in zip(source_ids, sources):
      input_file.write('{0!r}|{1!r}\n'.format(identifier, sentence).encode('utf8'))

  input_file.flush()    
  #os.system("./metamaplite.sh --inputformat=sldiwi inputfile.txt")
  os.system("./metamaplite.sh --scheduler --inputformat=sldiwi --restrict-to-sts=dsyn,clnd,gngm "+filename+" "+outputfilename)
  #./metamaplite.sh  --inputformat=sldiwi inputfile.txt outputfile.mmi
  output = str(output_file.read())

  concepts = Corpus.load(output.splitlines())

  for concept in concepts:
    conindex = concept.index.strip("'")
    pmid, source = conindex.split(':')
    if pmid not in concepts_dict:
      concepts_dict[pmid] = {}
    concept_value = [concept.score]+[concept.preferred_name]+[concept.cui]+[concept.semtypes]+[str(concept.trigger).replace('"','')]+[concept.pos_info]+[concept.tree_codes]
    if source not in concepts_dict[pmid]:
       concepts_dict[pmid][source] = [concept_value]
    #elif type(concepts_dict[pmid][source]) == list:
        
    else:
         concepts_dict[pmid][source].append(concept_value)
  #concepts = Corpus.load(contents.splitlines())
  #conindex = source_ids[i]
  #pmid, source = conindex.split(':')
  #if pmid not in concepts_dict:
  #      concepts_dict[pmid] = {}
      #if source not in concepts_dict[pmid]:
      
  #concepts_dict[pmid][source] = contents
       
  #concepts, error = mm.extract_concepts(sources, source_ids)#,restrict_to_sts=['neop'])
  #print(concepts)
  #print(error)
  
  #print(concepts_dict['1']['TEXT_MM'])
  #print(concepts_dict["5"]["TEXT_MM"])
  print("Num of keys")
  print(len(concepts_dict.keys()))
  metadir = 'finalmetaoutliterestrict'
  if not os.path.exists(metadir):
    os.makedirs(metadir, exist_ok=True)
  
  #metaoutfile = xmlfile + '.json.gz'
  metaoutfile = metadir + os.sep + os.path.split(xmlfile)[-1][:-7] + '-metamap.json.gz'

  concepts_json = json.dumps(concepts_dict)

  with gzip.open(metaoutfile, 'wt') as out:
    out.write(concepts_json)
    #out.write(str(concepts_dict))

  logdir = 'comparelogsrestrictsemantictypes'
  if not os.path.exists(logdir):
    os.makedirs(logdir, exist_ok=True)

  statsfile = 'comparelogs' + os.sep + os.path.split(xmlfile)[-1][:-7] + '-meta-stats.txt'
  with open(statsfile, 'w') as statsf:
    #end = time.time()
    #end_str = time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime(end))
    #total_time = end - start
    #minutes, seconds = divmod(total_time, 60)
    #hours, minutes = divmod(minutes, 60)
    #total_time_str = '%d:%02d:%02d' % (hours, minutes, seconds)

    newtimer.end()

    statsf.write('\ntotal time: ' + newtimer.total_time_str)
    statsf.write('\ndocument count: %d\n' % (count))
    statsf.write('\naverage time per document: %ds' % (newtimer.total_time / count))

##def multi(directory):
##    files = os.listdir(directory)
##
##    paths = []
##    for file in files:
##        paths.append(os.path.join(directory, file))
##
##    with Pool(processes=4) as pool:
##        pool.apply(main, paths)

if __name__ == '__main__':

  if len(sys.argv) == 2:
    main(sys.argv[1])
  elif len(sys.argv) == 3:
    multi(sys.argv[2])
  else:
    main()
