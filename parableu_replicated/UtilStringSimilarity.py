from crystalbleu import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu as originBleu
from bleu_ignoring import sentence_bleu  as crystalBLEU, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from nltk import ngrams
import traceback
from Levenshtein import ratio,distance
import sctokenizer
import code_bert_score
import pickle
from calc_code_bleu_local import *

def getValueInDict(dictInfos,key):
    output=None
    if key in dictInfos.keys():
        output= dictInfos[key]
    return output

def getImportantWords(lstTargets):
    dictVocab={}
    for item in lstTargets:
        arrWords=item.split()
        for word in arrWords:
            if word not in dictVocab.keys():
                dictVocab[word]=1
            else:
                dictVocab[word]+=1
    dictVocab=dict(sorted(dictVocab.items(), key=lambda x: x[1],reverse=True))
    return dictVocab


def getTokenizedList(lstInput,strLanguage='python'):
    lstOutput=[]
    for item in lstInput:
        strOut = item
        try:
            tokens = sctokenizer.tokenize_str(item, lang=strLanguage)
            lstVal = ['{}'.format(t.token_value) for t in tokens]
            strOut = ' '.join(lstVal)
        except Exception as e:
            traceback.print_exc()
        lstOutput.append(strOut)

def getSharedNGram(arrTrainTargets,kTopFrequenceNGram,fpTrainTrivialPkl):
    frequencyNGrams = {}
    for j in range(0, len(arrTrainTargets)):
        try:
            lstItemSplitCode = arrTrainTargets[j].split()
            for indexNGram in range(3, 4):
                lstItemNGrams = ngrams(lstItemSplitCode, indexNGram)
                for gr in lstItemNGrams:
                    if gr not in frequencyNGrams.keys():
                        frequencyNGrams[gr] = 1
                    else:
                        frequencyNGrams[gr] += 1
        except Exception as e:
            traceback.print_exc()

    lstFreqNGramKeys = list(frequencyNGrams)[:kTopFrequenceNGram]
    trivially_shared_ngrams = {}
    for item in lstFreqNGramKeys:
        # strAdd=' '.join(list(item))
        trivially_shared_ngrams[item] = frequencyNGrams[item]
    pickle.dump(trivially_shared_ngrams, open(fpTrainTrivialPkl, 'wb'))
    return trivially_shared_ngrams


def getAllSimilarityScoreAtListLevel(lstExpected,lstPredicted,dictExtraInfos):
    dictOutputs={}
    lstPreprocessExpected=lstExpected
    lstPreprocessPredicted=lstPredicted
    if getValueInDict(dictExtraInfos,'isTokenizer')==True:
        strLanguage=dictExtraInfos['strLanguage']
        lstPreprocessExpected=getTokenizedList(lstExpected,strLanguage)
        lstPreprocessPredicted = getTokenizedList(lstPredicted, strLanguage)

    pred_red_results = code_bert_score.score(cands=lstPreprocessPredicted, refs=lstPreprocessExpected, lang=strLanguage)
    lstRedF3 = pred_red_results[3].tolist()
    minLen = min([len(lstPreprocessExpected), len(lstPreprocessPredicted)])
    trivially_shared_ngrams=getSharedNGram(lstPreprocessExpected,dictExtraInfos['topNGrams'], dictExtraInfos['fopOutput']+'trivial_ngrams.pkl')
    rougeObj=dictExtraInfos['rougeObj']
    dictVocab=getImportantWords(lstPreprocessExpected)
    keywords = list(dictVocab.keys())[:dictExtraInfos['topNGrams']]
    lstVocabCodeBLEU=['{}\n{}'.format(it,dictVocab[it]) for it in dictVocab.keys()]
    f1=open(dictExtraInfos['fopOutput']+'keywords.txt','w')
    f1.write('\n'.join(lstVocabCodeBLEU))
    f1.close()
    dictOverallScores={}
    for idx2 in range(0, minLen):
        try:
            itemPred = lstPreprocessPredicted[idx2]
            itemExp = lstPreprocessExpected[idx2]
            lstStrPreds = itemPred.split()
            lstStrExps = itemExp.split()
            scoreCodeBLEU = getCodeBLEUCustomKeyWords([itemPred], [[itemExp]],keywords,strLanguage)
            dictScoreJ = getSimilarityScore(itemExp, itemPred, lstStrExps,
                                            lstStrPreds, trivially_shared_ngrams, rougeObj)
            dictScoreJ['CodeBertScore']=lstRedF3[idx2]
            dictScoreJ['CodeBleu']=scoreCodeBLEU
            dictOverallScores[idx2]=dictScoreJ
        except Exception as e:
            traceback.print_exc()
    return dictOverallScores


def getSimilarityScore(str1,str2,lstStr1,lstStr2,trivially_shared_ngrams,rouge):
    dictItem={}
    # trivially_shared_ngrams=dictConfig['trivially_shared_ngrams']
    # rouge=dictConfig['rouge']
    try:
        bleu_score=originBleu([lstStr1], lstStr2)
        crystalBLEU_score = crystalBLEU(
            [lstStr1], lstStr2, ignoring=trivially_shared_ngrams)
        meteor_sc = meteor_score([lstStr1], lstStr2)
        rouge_scores=rouge.get_scores(str1, str2)
        lev_sim=ratio(str1,str2)
        # print(lev_sim)
        dictItem['o_b']=bleu_score
        dictItem['c_b']=crystalBLEU_score
        dictItem['m']=meteor_sc
        dictItem['r']=rouge_scores
        dictItem['lev_sim']=lev_sim
    except Exception as e:
        traceback.print_exc()
    return dictItem

def edit_distance(string1, string2):
    """Ref: https://bit.ly/2Pf4a6Z"""

    if len(string1) > len(string2):
        difference = len(string1) - len(string2)
        string1[:difference]

    elif len(string2) > len(string1):
        difference = len(string2) - len(string1)
        string2[:difference]

    else:
        difference = 0

    for i in range(len(string1)):
        if string1[i] != string2[i]:
            difference += 1

    return difference

print(edit_distance("kitten", "sitting hello world")) #3
# print(edit_distance("medium jkkjkjkjkjk", "median")) #2
hypothesis = "the #### transcript is a written version of each day 's cnn student news program use this transcript to he    lp students with reading comprehension and vocabulary use the weekly newsquiz to test your knowledge of storie s you     saw on cnn student news"

reference = "this page includes the show transcript use the transcript to help students with reading comprehension and     vocabulary at the bottom of the page , comment for a chance to be mentioned on cnn student news . you must be a teac    her or a student age # # or older to request a mention on the cnn student news roll call . the weekly newsquiz tests     students ' knowledge of even ts in the news"
print(edit_distance(hypothesis, reference))