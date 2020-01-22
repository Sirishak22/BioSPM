#####################
# Krishna Sirisha Motamarry & Lowell Milliken
# Extracts relevant fields from extra abstract directory for TREC Precision Medicine track.
#
# Usage: python text_to_trectext.py <input directory> <output filename>
#
#####################
import sys
import os


def main(directory, outfilename = 'extra_trec_2017.txt'):
    files = os.listdir(directory)
    with open(outfilename, 'w') as textfile:

        for filename in files:
            if filename.endswith('.txt'):
                docid = filename[:-4]
                print(docid)
                filepath = directory + os.sep + filename
                with open(filepath, 'r') as infile:
                    textfile.write('<DOC>\n')
                    textfile.write('<DOCNO>{}</DOCNO>\n'.format(docid))

                    journal = infile.readline()[9:]
                    title = infile.readline()[7:]

                    textfile.write('<TEXT>\n')
                    for line in infile:
                        textfile.write(line + '\n')
                    textfile.write('</TEXT>\n')

                    textfile.write('<TITLE>')
                    textfile.write(title)
                    textfile.write('</TITLE>\n')

                    textfile.write('<JOURNAL>')
                    textfile.write(journal)
                    textfile.write('</JOURNAL>\n')

                    textfile.write('</DOC>\n')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
