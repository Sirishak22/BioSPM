#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 22:15:12 2019

@author: Krishna Sirisha Motamarry
"""

#####################

# Extracts relevant fields from xml.gz files in a directory for TREC Precision Medicine track.
#
# Usage: python metamaplitedir.py <input directory> <output directory>
#
#####################
import sys
import tempmetamaplite_restrict
import os
import time


def main(directory):
  files = os.listdir(directory)

  start = time.time()
  start_str = time.strftime('%a-%d-%b-%Y-%H-%M-%S',time.localtime(start))
  logname = 'log-' + start_str + '.txt'

  with open(logname, 'w') as log:
    count = 0
    log.write('starting processing: ' + start_str)
    for filename in files:
      if filename.endswith('.gz'):
        print("File: " + filename)
        tempmetamaplite_restrict.main(directory + os.sep + filename)
        count += 1

    end = time.time()
    end_str = time.strftime('%a, %d %b %Y %H:%M:%S',time.localtime(end))
    total_time = end - start
    minutes, seconds = divmod(total_time, 60)
    hours, minutes = divmod(minutes, 60)
    total_time_str = '%d:%02d:%02d' % (hours, minutes, seconds)

    log.write('\nfinished processing: ' + end_str)
    log.write('\ntotal time: ' + total_time_str)
    log.write('\nfile count: %d\n' % (count))
    log.write('\naverage time per file: %ds' % (total_time/count))

if __name__ == '__main__':
  if len(sys.argv) == 2:
    main(sys.argv[1])
  else:
    print('Incorrect number of arguments. Need directory.')