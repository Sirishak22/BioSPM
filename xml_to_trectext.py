#####################
# Krishna Sirisha Motamarry & Lowell Milliken
# Extracts relevant fields from xml.gz file for TREC Precision Medicine track.
#
# Usage: python xml_to_trectext.py <input filename> <output directory>
#
#####################
import xml.etree.ElementTree as ET
import sys
import gzip
import os
import time


def main(xmlfile='medline17n0889.xml.gz', outdir='out-text2'):
    print('Uncompressing\n')
    uncompressed_file = gzip.open(xmlfile)

    try:
        print('Parsing XML tree\n')
        tree = ET.parse(uncompressed_file)
        root = tree.getroot()
    finally:
        print('Closing uncompressed file\n')
        uncompressed_file.close()

    filename = outdir + os.sep + os.path.split(xmlfile)[-1][:-7] + '.txt'
    start = time.time()
    count = 0
    print('Extracting text, title, and keywords\n')
    with open(filename, 'w', encoding='utf-8') as textfile:
        for child in root:
            medlinecitation = child.find('MedlineCitation')
            pmid = medlinecitation.find('PMID').text
            article = medlinecitation.find('Article')
            title = article.find('ArticleTitle').text
            abstractchild = article.find('Abstract')
            journal = article.find('Journal').find('Title').text
            authorlist = article.find('AuthorList')
            commentlist=medlinecitation.find('CommentsCorrectionsList')
            pubdate = article.find('Journal').find('JournalIssue').find('PubDate')#.find('Year').text
            pubtypelist = article.find('PublicationTypeList')
            language=article.find('Language').text
            

            keywords = ''
            keywordlist = medlinecitation.find('KeywordList')
            if keywordlist is not None:
                for child in keywordlist:
                    if child.text:
                        keywords += child.text + ','
                keywords = keywords.rstrip(',')

            meshlist = medlinecitation.find('MeshHeadingList')
            if meshlist is not None:
                for meshheading in meshlist:
                    for child in meshheading:
                        if child.text:
                            keywords += child.text + ','
                keywords = keywords.rstrip(',')

            if abstractchild is not None:
                abstract = ''
                for textchild in abstractchild:
                    if textchild.tag == 'AbstractText':
                        if textchild.text:
                            abstract += ' ' + textchild.text
            else:
                abstract = title

            authors = []
            if authorlist:
                for authorchild in authorlist:
                    last = authorchild.find('LastName')
                    first = authorchild.find('ForeName')
                    if last is not None:
                        last = last.text
                    else:
                        last = ''
                    if first is not None:
                        first = first.text
                    else:
                        first = ''

                    author = first + ' ' + last

                    if first or last:
                        authors.append(author)

            authors = ', '.join(authors)
            comments=''
            #comments=0
            if commentlist is not None:
                for commentchild in commentlist:
                    #print(commentchild.attrib['RefType'])
                    comments=comments+commentchild.attrib['RefType']+'|'
                comments=comments.rstrip('|')    
            pubtypes=''
            if pubtypelist is not None:
                for pubtype in pubtypelist:
                    if pubtype.text:
                        pubtypes+=pubtype.text + '|'
                pubtypes = pubtypes.rstrip('|')
                   
                

            print(pmid)
            textfile.write('<DOC>\n')
            textfile.write('<DOCNO>{}</DOCNO>\n'.format(pmid))

            if abstract:
                abstract = abstract.replace('<', '').replace('>', '').strip()
                abstract = abstract.encode('ascii', errors='ignore').decode()
                textfile.write('<TEXT>\n')
                textfile.write(abstract)
                textfile.write('\n</TEXT>\n')

            if title:
                title = title.replace('<', '').replace('>', '').strip()
                title = title.encode('ascii', errors='ignore').decode()
                # re.sub(r'[^\x00-\x7f]', r'', title)
                textfile.write('<TITLE>')
                textfile.write(title)
                textfile.write('</TITLE>\n')

            if keywords:
                keywords = keywords.replace('<', '').replace('>', '').strip()
                keywords = keywords.encode('ascii', errors='ignore').decode()
                textfile.write('<KEYWORDS>')
                textfile.write(keywords)
                textfile.write('</KEYWORDS>\n')

            if journal:
                journal = journal.replace('<', '').replace('>', '').strip()
                journal = journal.encode('ascii', errors='ignore').decode()
                textfile.write('<JOURNAL>')
                textfile.write(journal)
                textfile.write('</JOURNAL>\n')

            if authors:
                authors = authors.replace('<', '').replace('>', '').strip()
                authors = authors.encode('ascii', errors='ignore').decode()
                textfile.write('<AUTHORS>')
                textfile.write(authors)
                textfile.write('</AUTHORS>\n')
            
            if pubtypes:
                pubtypes = pubtypes.replace('<', '').replace('>', '').strip()
                pubtypes = pubtypes.encode('ascii', errors='ignore').decode()
                textfile.write('<PUBTYPES>')
                textfile.write(pubtypes)
                textfile.write('</PUBTYPES>\n')
            
            
            if comments:
                comments = comments.replace('<', '').replace('>', '').strip()
                comments = comments.encode('ascii', errors='ignore').decode()
                textfile.write('<COMMENTS>')
                textfile.write(comments)
                textfile.write('</COMMENTS>\n')
            else:
                comments='NA'
                textfile.write('<COMMENTS>')
                textfile.write(comments)
                textfile.write('</COMMENTS>\n')
            
            if pubdate:
                for child in pubdate:
                    if child.tag == 'Year':
                        pubdate=child.text
                    elif child.tag == 'MedlineDate':
                        pubdate = child.text
                        newpubdate= pubdate.split()
                        pubdate=newpubdate[0]
                
            #if pubdate:
            #    pubdate = pubdate.replace('<', '').replace('>', '').strip()
            #    pubdate = pubdate.encode('ascii', errors='ignore').decode()
            #    textfile.write('<PUBDATE>')
            #    textfile.write(pubdate)
            #    textfile.write('</PUBDATE>\n')
            textfile.write('<PUBDATE>{}</PUBDATE>\n'.format(pubdate))
            textfile.write('<LANGUAGE>{}</LANGUAGE>\n'.format(language))
            #textfile.write('<NUMOFCOMMENTS>{}</NUMOFCOMMENTS>\n'.format(comments))
            
            textfile.write('</DOC>\n')
            count += 1

    end = time.time()
    end_str = time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime(end))
    total_time = end - start
    minutes, seconds = divmod(total_time, 60)
    hours, minutes = divmod(minutes, 60)
    total_time_str = '%d:%02d:%02d' % (hours, minutes, seconds)

    print('\nfinished processing: ' + end_str)
    print('\ntotal time: ' + total_time_str)
    print('\nfile count: %d\n' % (count))
    print('\naverage time per file: %ds' % (total_time / count))


if __name__ == '__main__':

    if len(sys.argv) == 2:
        main(sys.argv[1], sys.argv[2])
    else:
        main()
