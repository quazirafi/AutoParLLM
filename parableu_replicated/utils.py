import sys, os, traceback
import sctokenizer
# Natural Language Toolkit: Utility functions
#
# Copyright (C) 2001-2020 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

from itertools import chain

def pad_sequence(
    sequence,
    n,
    pad_left=False,
    pad_right=False,
    left_pad_symbol=None,
    right_pad_symbol=None,
):
    """
    Returns a padded sequence of items before ngram extraction.
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        ['<s>', 1, 2, 3, 4, 5, '</s>']
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        ['<s>', 1, 2, 3, 4, 5]
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [1, 2, 3, 4, 5, '</s>']
    :param sequence: the source data to be padded
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


# add a flag to pad the sequence so we get peripheral ngrams?


def ngrams(
    sequence,
    n,
    pad_left=False,
    pad_right=False,
    left_pad_symbol=None,
    right_pad_symbol=None,
):
    """
    Return the ngrams generated from a sequence of items, as an iterator.
    For example:
        >>> from nltk.util import ngrams
        >>> list(ngrams([1,2,3,4,5], 3))
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]
    Wrap with list for a list version of this function.  Set pad_left
    or pad_right to true in order to get additional ngrams:
        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, None)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
    :param sequence: the source data to be converted into ngrams
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = pad_sequence(
        sequence, n, pad_left, pad_right, left_pad_symbol, right_pad_symbol
    )

    history = []
    while n > 1:
        # PEP 479, prevent RuntimeError from being raised when StopIteration bubbles out of generator
        try:
            next_item = next(sequence)
        except StopIteration:
            # no more data, terminate the generator
            return
        history.append(next_item)
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]
def writeListToFilePreprocess(lstStr,fp,strLanguage):
    lstOutput=[]
    for item in lstStr:
        tokens = sctokenizer.tokenize_str(item, lang=strLanguage)
        lstVal = ['{}'.format(t.token_value) for t in tokens]
        strOut=' '.join(lstVal)
        if not strOut.startswith('# pragma'):
            strOut='None'
        lstOutput.append(strOut)
    f1 = open(fp, 'w')
    f1.write('\n'.join(lstOutput))
    f1.close()

def writeListToFilePreprocessRemoveTemplatePragma(lstStr,fp,strLanguage):
    lstOutput=[]
    for item in lstStr:
        tokens = sctokenizer.tokenize_str(item, lang=strLanguage)
        lstVal = ['{}'.format(t.token_value) for t in tokens]
        strOut=' '.join(lstVal)
        if not strOut.startswith('# pragma'):
            strOut='None'
        elif strOut.startswith('# pragma omp parallel for'):
            # print('go here')
            strOut=strOut.replace('# pragma omp parallel for','').strip()
            if strOut=='':
                strOut='None'
            # print(strOut)
        lstOutput.append(strOut)
    f1 = open(fp, 'w')
    f1.write('\n'.join(lstOutput))
    f1.close()

def removeRedundantNGramsForOpenMPBLEU(lstStr):
    lstOutput=[]
    for item in lstStr:
        outItem=item.replace('# pragma omp parallel for','').strip()
        lstOutput.append(outItem)
    return lstOutput

def removeReorderingIssue(lstStr):
    lstOutput=[]
    for item in lstStr:
        # outItem=item.replace('# pragma omp parallel for','').strip()
        # lstOutput.append(outItem)
        lstTokens=item.split()
        j=0
        lstItemOut=[]
        while(j<len(lstTokens)):
            if lstTokens[j]=='(':
                k=j+1
                while(k<len(lstTokens)):
                    if lstTokens[k]==')':
                        break
                    k+=1
                strSubString=''.join(lstTokens[(j+1):k])
                # print(strSubString)
                lstVarNames=strSubString.split(',')
                lstVarNames=sorted(lstVarNames)
                strNewStringToAPpend='( {} )'.format(' , '.join(lstVarNames))
                # print(strNewStringToAPpend)
                lstItemOut.append(strNewStringToAPpend)
                j=k+1
            else:
                lstItemOut.append(lstTokens[j])
                j+=1
        strItemOut=' '.join(' '.join(lstItemOut).split())
        # print(strItemOut)
        lstOutput.append(strItemOut)
    return lstOutput

def writeListToFile(lstStr,fp):
    f1 = open(fp, 'w')
    f1.write('\n'.join(lstStr))
    f1.close()
def createDirIfNotExist(fopOutput):
    try:
        # Create target Directory
        os.makedirs(fopOutput, exist_ok=True)
        #print("Directory ", fopOutput, " Created ")
    except FileExistsError:
        print("Directory ", fopOutput, " already exists")