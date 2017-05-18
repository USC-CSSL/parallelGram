__author__ = 'joe'

import os
import collections
import numpy as np
import pandas as pd
import pickle
import gzip
from collections import defaultdict
from collections import Counter
import io
import requests
import urllib
import dill
import multiprocessing
import sys

def getFiles(ngram_path):
    """
    Gets list of ngram file names
    :param ngram_path:
    :return: list of ngram file names
    """
    ngram_files = []

    for f in os.listdir(ngram_path):
        if os.path.isfile(os.path.join(ngram_path, f)) and not f.startswith('.'):
            ngram_files.append(f)
    return(ngram_files)


def getWords(wordPath):
    """
    Get list of words from specified file
    :param wordPath:
    :return: Get the words specified in input file
    """
    with open(wordPath, 'rb') as wf:
        words = [w.strip() for w in wf.readline().split(',')]
        return(words)


def getNegWords(wordPath):
    """
    Get list of words from specified file
    :param wordPath:
    :return: Get the words specified in input file
    """
    with open(wordPath, 'rb') as wf:
        lb1 = r'(?<!'
        lb2 = r' )'

        negReg = ''.join([lb1 + nw.strip() + lb2 for nw in wf.readline().split(',')])

        return(negReg)


def chunks(l, nJobs):
    """
    Split l files into nJobs
    :param l:
    :param nJobs:
    :return: Yield an index range for l of n items
    """
    n = len(l)/nJobs
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]

def mergeDicts(dictList):
    """
    Because this program splits the search across jobs,
    we need a way to combine the results. This function does that.
    :param dictList:
    :return: A dictionary containing all co-occurrences
    """
    uniondict = defaultdict(lambda: defaultdict(lambda: 0))

    for d in dictList:
        for k1, v1 in d.items():
            for k2, v2 in v1.items():
                    if type(v2) == int:
                        uniondict[k1][k2] += int(v2)

    return (uniondict)


def gnTotal(url):
    """
    Google provides a count of total ngrams for a given size.
    This function downloads that file and gets totals for each year.
    :param url:
    :return:
    """
    response = urllib.urlopen(url)
    lines = response.readlines()[0].split('\t')
    total_counts = {'year': [], 'total_words': [], 'total_pages': [], 'total_volumes': []}
    for line in lines:
        line = line.split(',')
        if len(line) == 4:
            total_counts['year'].append(line[0])
            total_counts['total_words'].append(line[1])
            total_counts['total_pages'].append(line[2])
            total_counts['total_volumes'].append(line[3])

    totalCountsDf = pd.DataFrame.from_dict(total_counts)
    return(totalCountsDf)



def chunkGrams3(ngram_chunk, baseWords, targetWords, output_path, ngram_path, negReg):
    """
    This function using pattern matching to identify co-occurrences.
    :param ngram_chunk: The files to search through for given job
    :param baseWords: List of base words
    :param targetWords: List of target words
    :param output_path: Where should results be stored?
    :param ngram_path: Where are the ngrams?
    :param negWords: Negation words to exclude by
    :return: Nothing. Dumps results to pickeled file
    """
    wordFreqs = defaultdict(lambda: defaultdict(lambda: 0))
    coFreqs = defaultdict(lambda: defaultdict(lambda: 0))

    wordVols = defaultdict(lambda: defaultdict(lambda: 0))
    coVols = defaultdict(lambda: defaultdict(lambda: 0))

    file_counter = 0
    total_files = len(ngram_chunk) + 1

    baseWords = baseWords + ['behebung']
    targetWords = targetWords
    for ngram_file in ngram_chunk:
        file_counter +=1
        print 'Analyzing file {0} of {1} in chunk'.format(file_counter, total_files)
        cur_file = gzip.open(os.path.join(ngram_path, ngram_file), 'rb')
        for line in cur_file:
            dat = line.split('\t')
            phrase = dat[0].lower()
            if any(baseWord in phrase for baseWord in baseWords):  # If contains baseword then
                if any(targetWord in phrase for targetWord in targetWords):  # If also contains target word, then
                    for bWord in baseWords:
                        reg = negReg + r'\b' + re.escape(bWord) + r'\b'

                        try:

                            re.search(reg, phrase).group(0)
                            wordFreqs[dat[1]][bWord] += int(dat[2])
                            wordVols[dat[1]][bWord] += int(dat[3].replace('\n', ''))

                            for tWord in targetWords:
                                regT = negReg + r'\b' + re.escape(tWord) + r'\b'
                                try:
                                    re.search(regT, phrase).group(0)

                                    wordFreqs[dat[1]][tWord] += int(dat[2])
                                    wordVols[dat[1]][tWord] += int(dat[3].replace('\n', ''))
                                    coFreqs[dat[1]]['_'.join([bWord,tWord])] += int(dat[2])
                                    coVols[dat[1]]['_'.join([bWord,tWord])] += int(dat[3].replace('\n', ''))

                                except:
                                    continue


                        except:
                            continue

                else:
                    for bWord in baseWords:
                        reg = negReg + r'\b' + re.escape(bWord) + r'\b'
                        try:
                            re.search(reg, phrase).group(0)
                            wordFreqs[dat[1]][bWord] += int(dat[2])
                            wordVols[dat[1]][bWord] += int(dat[3].replace('\n', ''))
                        except:
                            continue


            elif any(targetWord in phrase for targetWord in targetWords):
                    for tWord in targetWords:
                        regT = negReg + r'\b' + re.escape(tWord) + r'\b'
                        try:

                            print re.search(regT, phrase).group(0)
                            wordFreqs[dat[1]][tWord] += int(dat[2])
                            wordVols[dat[1]][tWord] += int(dat[3].replace('\n', ''))
                        except:
                            continue


    dictFreqs = mergeDicts(dictList=[wordFreqs,coFreqs])
    dictVols = mergeDicts(dictList=[wordVols,coVols])
    ds = [dictFreqs, dictVols]

    print 'Writing {0}'.format(output_path)
    dill.dump(ds, open(output_path, 'wb'))


def chunkGrams2(ngram_chunk, baseWords, targetWords, output_path, ngram_path):
    """
    This function using pattern matching to identify co-occurrences.
    :param ngram_chunk: The files to search through for given job
    :param baseWords: List of base words
    :param targetWords: List of target words
    :param output_path: Where should results be stored?
    :param ngram_path: Where are the ngrams?
    :return: Nothing. Dumps results to pickeled file
    """
    wordFreqs = defaultdict(lambda: defaultdict(lambda: 0))
    coFreqs = defaultdict(lambda: defaultdict(lambda: 0))

    wordVols = defaultdict(lambda: defaultdict(lambda: 0))
    coVols = defaultdict(lambda: defaultdict(lambda: 0))

    file_counter = 0
    total_files = len(ngram_chunk) + 1

    baseWords = baseWords + ['behebung']
    targetWords = targetWords
    for ngram_file in ngram_chunk:
        file_counter +=1
        print 'Analyzing file {0} of {1} in chunk'.format(file_counter, total_files)
        cur_file = gzip.open(os.path.join(ngram_path, ngram_file), 'rb')
        for line in cur_file:
            dat = line.split('\t')
            phrase = dat[0].lower()
            if any(baseWord in phrase for baseWord in baseWords):  # If contains baseword then
                if any(targetWord in phrase for targetWord in targetWords):  # If also contains target word, then
                    for bWord in baseWords:
                        reg = r'\b' + re.escape(bWord) + r'\b'
                        try:
                            re.search(reg, phrase).group(0)
                            wordFreqs[dat[1]][bWord] += int(dat[2])
                            wordVols[dat[1]][bWord] += int(dat[3].replace('\n', ''))

                            for tWord in targetWords:
                                regT = r'\b' + re.escape(tWord) + r'\b'
                                try:
                                    re.search(regT, phrase).group(0)

                                    wordFreqs[dat[1]][tWord] += int(dat[2])
                                    wordVols[dat[1]][tWord] += int(dat[3].replace('\n', ''))
                                    coFreqs[dat[1]]['_'.join([bWord,tWord])] += int(dat[2])
                                    coVols[dat[1]]['_'.join([bWord,tWord])] += int(dat[3].replace('\n', ''))

                                except:
                                    continue


                        except:
                            continue

                else:
                    for bWord in baseWords:
                        reg = r'\b' + re.escape(bWord) + r'\b'
                        try:
                            re.search(reg, phrase).group(0)
                            wordFreqs[dat[1]][bWord] += int(dat[2])
                            wordVols[dat[1]][bWord] += int(dat[3].replace('\n', ''))
                        except:
                            continue


            elif any(targetWord in phrase for targetWord in targetWords):
                    for tWord in targetWords:
                        regT = r'\b' + re.escape(tWord) + r'\b'
                        try:

                            print re.search(regT, phrase).group(0)
                            wordFreqs[dat[1]][tWord] += int(dat[2])
                            wordVols[dat[1]][tWord] += int(dat[3].replace('\n', ''))
                        except:
                            continue


    dictFreqs = mergeDicts(dictList=[wordFreqs,coFreqs])
    dictVols = mergeDicts(dictList=[wordVols,coVols])
    ds = [dictFreqs, dictVols]

    print 'Writing {0}'.format(output_path)
    dill.dump(ds, open(output_path, 'wb'))



def chunkGrams1(ngram_chunk, baseWords, targetWords, output_path, ngram_path):
    """
    This function using exact matching to identify co-occurrences.
    :param ngram_chunk: The files to search through for given job
    :param baseWords: List of base words
    :param targetWords: List of target words
    :param output_path: Where should results be stored?
    :param ngram_path: Where are the ngrams?
    :return: Nothing. Dumps results to pickeled file
    """
    wordFreqs = defaultdict(lambda: defaultdict(lambda: 0))
    coFreqs = defaultdict(lambda: defaultdict(lambda: 0))

    wordVols = defaultdict(lambda: defaultdict(lambda: 0))
    coVols = defaultdict(lambda: defaultdict(lambda: 0))

    file_counter = 0
    total_files = len(ngram_chunk) + 1

    for ngram_file in ngram_chunk:
        file_counter +=1
        print 'Analyzing file {0} of {1} in chunk'.format(file_counter, total_files)
        cur_file = gzip.open(os.path.join(ngram_path, ngram_file), 'rb')
        for line in cur_file:
            dat = line.split('\t')
            phrase = dat[0].lower()
            if any(baseWord in phrase for baseWord in baseWords):  # If contains baseword then
                if any(targetWord in phrase for targetWord in targetWords):  # If also contains target word, then
                    for bWord in baseWords:
                        if bWord in phrase:
                            wordFreqs[dat[1]][bWord] += int(dat[2])
                            wordVols[dat[1]][bWord] += int(dat[3].replace('\n', ''))

                            for tWord in targetWords:
                                if tWord in phrase:


                                    wordFreqs[dat[1]][tWord] += int(dat[2])
                                    wordVols[dat[1]][tWord] += int(dat[3].replace('\n', ''))
                                    coFreqs[dat[1]]['_'.join([bWord,tWord])] += int(dat[2])
                                    coVols[dat[1]]['_'.join([bWord,tWord])] += int(dat[3].replace('\n', ''))

                else:
                    for bWord in baseWords:
                        if bWord in phrase:

                            wordFreqs[dat[1]][bWord] += int(dat[2])
                            wordVols[dat[1]][bWord] += int(dat[3].replace('\n', ''))



            elif any(targetWord in phrase for targetWord in targetWords):
                    for tWord in targetWords:
                        if tWord in phrase:

                            wordFreqs[dat[1]][tWord] += int(dat[2])
                            wordVols[dat[1]][tWord] += int(dat[3].replace('\n', ''))


    dictFreqs = mergeDicts(dictList=[wordFreqs,coFreqs])
    dictVols = mergeDicts(dictList=[wordVols,coVols])
    ds = [dictFreqs, dictVols]

    print 'Writing {0}'.format(output_path)
    dill.dump(ds, open(output_path, 'wb'))


def main(ngramSyntax = sys.argv[1]):
    """
    This function extracts necessary parameters from the parameters txt file
    and distributes the jobs across workers.
    :param ngramSyntax:
    :return:
    """

    pars = []
    with open(ngramSyntax, 'rb') as ngs:
        for line in ngs:
            pars.append(line.replace('\n', '').strip())


    ngram_path = pars[0]
    nJobs = int(pars[1])
    output_dir = pars[2]
    baseWordPath = pars[3]
    targetWordPath = pars[4]
    coType = pars[5]

    if coType == '3':

        negWordPath = pars[6]
        negReg = getNegWords(negWordPath)

    baseWords = getWords(baseWordPath)
    targetWords = getWords(targetWordPath)

    print('Calculating n-gram cooccurrence with parameters:\n{0}\n{1}\n{2}\n{3}\n{4}\n{5}\n{6}'.format(ngram_path,
                                                                                      nJobs,
                                                                                      output_dir,
                                                                                      baseWordPath,
                                                                                      targetWordPath,
                                                                                      baseWords,
                                                                                      targetWords))

    if coType == '1':
        print 'Cooccurrence search based on pattern matching (type 1)'
    elif coType == '2':
        print 'Cooccurrence search based on exact matching (type 2)'

    elif coType == '3':
        print 'Cooccurrence search based on exact matching with negation exclusion (type 3)'
        print 'Negation words: {0}'.format(negWords)


    ngram_files = sorted(getFiles(ngram_path=ngram_path))
    ngram_chunks = chunks(ngram_files, nJobs=nJobs)

    jobN = 0
    output_paths = []

    output_dir = output_dir

    jobs = []

    for chunk in ngram_chunks:
        jobN += 1
        out_path = output_dir + '/job_' + str(jobN) + '_dict.pickle'
        output_paths.append(out_path)

        if coType == '1':

            p = multiprocessing.Process(target=chunkGrams1, args=(chunk,baseWords,targetWords,out_path,ngram_path,))

        if coType == '2':
            p = multiprocessing.Process(target=chunkGrams2, args=(chunk, baseWords, targetWords, out_path, ngram_path,))

        if coType == '3':
            p = multiprocessing.Process(target=chunkGrams3, args=(chunk, baseWords, targetWords, out_path, ngram_path,negReg,))

        jobs.append(p)

    for p in jobs:
        p.start()

    for p in jobs:
        print 'Joining {0} thread'.format(p)
        p.join()


    return(output_paths)


def getGnDicts(output_paths, ngramSyntax):

    """This function combines the objects saved by the main() function and also
    downloads and combines total ngram counts.
    """
    pars = []
    with open(ngramSyntax, 'rb') as ngs:
        for line in ngs:
            pars.append(line.replace('\n', '').strip())

    output_dir = pars[2]


    ngramFreqDicts = []
    ngramVolDicts = []


    for dicPath in output_paths:
        dic = dill.load(open(dicPath, 'rb'))
        ngramFreqDicts.append(dic[0])
        ngramVolDicts.append(dic[1])


    ngramFreqs = mergeDicts(ngramFreqDicts)
    ngramFreqsDf = pd.DataFrame.from_dict(ngramFreqs, orient='index')

    ngramVols = mergeDicts(ngramVolDicts)
    ngramVolsDf = pd.DataFrame.from_dict(ngramVols, orient='index')

    url="http://storage.googleapis.com/books/ngrams/books/googlebooks-eng-all-totalcounts-20120701.txt"

    gnTotals = gnTotal(url)
    gnTotals.set_index('year', inplace=True)


    ngramFreqsDf = pd.concat([gnTotals, ngramFreqsDf], axis=1)
    ngramVolsDf = pd.concat([gnTotals, ngramVolsDf], axis=1)

    print 'Writing Frequencies to: {0}'.format(output_dir + '/ngramFreqsDf.csv')
    ngramFreqsDf.to_csv(output_dir + '/ngramFreqsDf.csv', sep=',')

    print 'Writing Volumes to: {0}'.format(output_dir + '/ngramVolsDf.csv')
    ngramVolsDf.to_csv(output_dir + '/ngramVolsDf.csv', sep=',')

if __name__ == '__main__':

    output_paths = main(ngramSyntax = sys.argv[1])
    getGnDicts(output_paths=output_paths, ngramSyntax=sys.argv[1])