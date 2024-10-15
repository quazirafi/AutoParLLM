import pandas as pd
from UtilStringSimilarity import *
import pickle
import numpy as np
from scipy import stats
import code_bert_score
from calc_code_bleu_local import *
from statistics import mean
from utils import *



kTopFrequenceNGram=3
isCacheNGram=False
strLanguage='java'
numberCount=90

fpExcel='gpt-human-score.csv'
dfCSVData=pd.read_csv(fpExcel)

lstActual=['{}'.format(it).split('\n')[0] for it in dfCSVData['actual_pragma'].to_list()][:numberCount]
# lstNoGNN=['{}'.format(it).split('\n')[0] for it in dfCSVData['gpt4-no-gnn-gen-pragma'].to_list()][:numberCount]
lstWithGNN=['{}'.format(it).split('\n')[0] for it in dfCSVData['GNN-gen-pragma'].to_list()][:numberCount]
lstId=['{}'.format(it) for it in list(range(1,len(lstActual)+1))]

# print(lstWithGNN)
configTrans=['gnn']
dictPredict={}
dictPredict[configTrans[0]]=lstWithGNN
# dictPredict[configTrans[1]]=lstNoGNN
for con in configTrans:
    fopResult='../metrics_parableu/'
    createDirIfNotExist(fopResult)
    fpPredict = fopResult + 'predict.txt'
    fpS2Predict = fopResult + 'preprocess_predict.txt'
    # writeListToFile(dictPredict[con],fpPredict)
    writeListToFilePreprocess(dictPredict[con], fpPredict, strLanguage)
    fpTrainTarget = fpTestTarget = fopResult + 'target.txt'
    fpS2TestTarget=fopResult + 'preprocess_target.txt'
    # writeListToFile(lstActual, fpTrainTarget)
    # writeListToFile(lstActual, fpTestTarget)
    writeListToFilePreprocess(lstActual, fpTrainTarget, strLanguage)
    writeListToFilePreprocess(lstActual, fpTestTarget, strLanguage)
    fpTrainTrivialPkl = fpTrainTrivialText = fopResult + 'n-gram.pkl'
    fpTestId = fopResult + 'test-id.txt'
    fpSummary = fopResult + 'summary.txt'
    writeListToFile(lstId, fpTestId)
    fpCompareCsv = fopResult + 'details.csv'
    fpS2CompareCsv = fopResult + 'preprocess_details.csv'
    lstHumanScores=dfCSVData['human-{}'.format(con)].to_list()[:numberCount]

    # print(len(lstActual))

    # # We do following steps:
    # 1. Tokenization
    # 2. Preparing trained n-grams
    # 3. Calculating Score

    rougeObj = Rouge()

    try:
        arrLinePreds = []
        if os.path.exists(fpPredict):
            f1 = open(fpPredict, 'r')
            arrLinePreds = f1.read().strip().split('\n')
            f1.close()
        trivially_shared_ngrams = {}
        isAbleSharedNGrams = False
        if isCacheNGram:
            try:
                trivially_shared_ngrams = pickle.load(open(fpTrainTrivialPkl, 'rb'))
                isAbleSharedNGrams = True
            except Exception as e:
                traceback.print_exc()
        if not isAbleSharedNGrams:
            f1 = open(fpTrainTarget, 'r')
            arrTrainTargets = f1.read().strip().split('\n')
            f1.close()
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
            lstStr500 = []
            for qk in trivially_shared_ngrams.keys():
                lstStr500.append('{} {}'.format(qk, trivially_shared_ngrams[qk]))
            f1 = open(fpTrainTrivialText, 'w')
            f1.write('\n'.join(lstStr500))
            f1.close()

        f1 = open(fpTestTarget, 'r')
        arrLineExps = f1.read().strip().split('\n')
        f1.close()
        f1 = open(fpTestId, 'r')
        arrLineIds = f1.read().strip().split('\n')
        f1.close()
        minLen = min([len(arrLineExps), len(arrLinePreds)])
        dictAllLinesCompared = {}
        lstStrCompareCSV = ['Key\tBLEU\tParaBLEU\tCrystalBLEU\tROUGE-1\tROUGE-2\tROUGE-L\tMeteor\tCodeBERTScore']
        f1 = open(fpCompareCsv, 'w')
        f1.write('\n'.join(lstStrCompareCSV) + '\n')
        f1.close()
        f1 = open(fpS2CompareCsv, 'w')
        f1.write('\n'.join(lstStrCompareCSV) + '\n')
        f1.close()
        lstStrCompareCSV = []
        lstS2StrCompareCSV = []
        # print('begin folder  {} {}'.format(index, nameFolder))

        pred_results = code_bert_score.score(cands=arrLineExps, refs=arrLinePreds, lang='c')
        lstF3 = pred_results[3].tolist()

        arrRedLinePreds = removeReorderingIssue(arrLinePreds)
        f1=open(fpS2Predict,'w')
        f1.write('\n'.join(arrRedLinePreds))
        f1.close()
        arrRedLineExps = removeReorderingIssue(arrLineExps)
        f1 = open(fpS2TestTarget, 'w')
        f1.write('\n'.join(arrRedLineExps))
        f1.close()

        pred_red_results = code_bert_score.score(cands=arrRedLineExps, refs=arrRedLinePreds, lang='c')
        lstRedF3 = pred_red_results[3].tolist()

        for idx2 in range(0, minLen):
            try:
                itemId = arrLineIds[idx2]
                itemPred = arrLinePreds[idx2]
                itemExp = arrLineExps[idx2]
                lstStrPreds = itemPred.split()
                lstStrExps = itemExp.split()
                scoreCodeBLEU=getParaBLEU([itemPred],[[itemExp]])
                dictScoreJ = getSimilarityScore(itemExp, itemPred, lstStrExps,
                                                lstStrPreds, trivially_shared_ngrams, rougeObj)
                # dictScoreJ[itemId] = dictScoreJ
                # print('{} {}'.format(dictScoreJ['r'][0]['rouge-l']['f'],''))
                # print(dictScoreJ)
                strLineAdd = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(itemId, dictScoreJ['o_b'],scoreCodeBLEU, dictScoreJ['c_b'],
                                                                         dictScoreJ['r'][0]['rouge-1']['f'],
                                                                         dictScoreJ['r'][0]['rouge-2']['f'],
                                                                         dictScoreJ['r'][0]['rouge-l']['f'],
                                                                         dictScoreJ['m'], lstF3[idx2])
                # print(strLineAdd)
                lstStrCompareCSV.append(strLineAdd)

                itemS2Pred = arrRedLinePreds[idx2]
                itemS2Exp = arrRedLineExps[idx2]
                lstS2StrPreds = itemS2Pred.split()
                lstS2StrExps = itemS2Exp.split()
                scoreS2CodeBLEU = getCodeBLEU([itemS2Pred], [[itemS2Exp]], lang='java')
                dictS2ScoreJ = getSimilarityScore(itemS2Exp, itemS2Pred, lstS2StrExps,
                                                lstS2StrPreds, trivially_shared_ngrams, rougeObj)
                # dictS2ScoreJ[itemId] = dictS2ScoreJ
                # print('{} {}'.format(dictScoreJ['r'][0]['rouge-l']['f'],''))
                strS2LineAdd = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(itemId, dictS2ScoreJ['o_b'],scoreS2CodeBLEU, dictS2ScoreJ['c_b'],
                                                               dictS2ScoreJ['r'][0]['rouge-1']['f'],dictS2ScoreJ['r'][0]['rouge-2']['f'],
                                                                 dictS2ScoreJ['r'][0]['rouge-l']['f'],dictS2ScoreJ['m'],
                                                                  lstRedF3[idx2])
                # print(strLineAdd)
                lstS2StrCompareCSV.append(strS2LineAdd)

                if len(lstStrCompareCSV) % 500 == 0 or idx2 + 1 == minLen:
                    f1 = open(fpCompareCsv, 'a')
                    f1.write('\n'.join(lstStrCompareCSV) + '\n')
                    f1.close()
                    f1 = open(fpS2CompareCsv, 'a')
                    f1.write('\n'.join(lstS2StrCompareCSV) + '\n')
                    f1.close()
                    lstStrCompareCSV=[]
                    lstS2StrCompareCSV=[]
            except Exception as e:
                traceback.print_exc()
                print('{} error'.format(idx2))
                input('ssss')
        dfAllMetrics = pd.read_csv(fpCompareCsv, delimiter='\t')
        lstColumnMetrics=dfAllMetrics.columns.to_list()[1:]
        dictColInfo={}
        # print(lstColumnMetrics)
        # input('aaa')
        for col in lstColumnMetrics:
            # print('name col: {}'.format(col))
            lstVal=np.array(dfAllMetrics[col].to_list())
            meanScore=mean(lstVal)
            spearmanScore=stats.spearmanr(lstVal, lstHumanScores).statistic
            dictColInfo[col]=[lstVal,meanScore,spearmanScore]

        strHead = ('Key\t{}\nSpearman\t{}'.format('\t'.join(lstColumnMetrics),'\t'.join(lstColumnMetrics)))
        strLineAdd = '{}\t{}\n{}\t{}'.format('Score','\t'.join(['{}'.format(dictColInfo[key][1]) for key in dictColInfo.keys()])
                                             ,'Spearman','\t'.join(['{}'.format(dictColInfo[key][2]) for key in dictColInfo.keys()]))


        f1 = open(fpSummary, 'w')
        f1.write(strHead + '\n' + strLineAdd + '\n')
        f1.close()
    except Exception as e:
        traceback.print_exc()
