# Adapted from https://github.com/ATR-DBI/ScanQA/blob/main/scripts/score.py
import json
import sys,os

# import nltk
# nltk.download('omw-1.4')

import numpy as np
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()

sys.path.append(os.path.join(os.getcwd()))

# def get_lemma(ss):
#     return [lemmatizer.lemmatize(token) for token in ss.split()]


def simple_ratio(numerator,denominator): 
    num_numerator=sum([1 if token in numerator else 0 for token in denominator])
    num_denominator=len(denominator)
    return num_numerator/num_denominator


# def tokens_unigram_f_value(ref: str,pred: str)->float:
#     ref_lemma = get_lemma(ref)
#     pred_lemma = get_lemma(pred)
#     precision = simple_ratio(ref_lemma,pred_lemma)
#     recall    = simple_ratio(pred_lemma,ref_lemma)
#     return 2*(recall*precision)/(recall+precision) if recall+precision!=0. else 0


def tokens_score(ref: str,pred: str)->float:
    return 1. if ref==pred else 0.


def evals_json(json_data):
    score_list = [
        'Top1 (EM)',
        # 'Top1 (F-value)'
        ]
    score = {s:[] for s in score_list}
    
    for ins in json_data:
        question_id=ins['question_id']
        question=ins['question']
        ref_answers=ins['answer']
        pred=ins["prediction"]

        # top-1
        answer = pred
        if answer in ref_answers:
            score['Top1 (EM)'].append(1)
            # score['Top1 (F-value)'].append(1)
        else:
            # scores=[tokens_unigram_f_value(answer,ref) for ref in ref_answers]
            score['Top1 (EM)'].append(0)
            # score['Top1 (F-value)'].append(max(scores))
        
    rlt={}
    for k,v in score.items():
        assert len(v)==len(json_data),len(v)
        print(k,np.mean(v)*100)
        rlt[k]=np.mean(v)*100
    return rlt

def eval_pycoco(json_data, use_spice=False):
    score_list = ['Top1 (EM)','Top10 (EM)','Top1 (F-value)','BLEU-1','BLEU-2','BLEU-3','BLEU-4']
    score = {s:[] for s in score_list}
    
    scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
    ]
    if use_spice:
        scorers.append((Spice(), "SPICE"))

    tokenizer = PTBTokenizer()
    # pycocoeval
    gts = {ins['question_id']:[{'caption':ans} for ans in ins['answer']] for ins in json_data}
    # res = {qid:[{'caption':value['answer_top10'][0]}] for qid,value in preds.items()}

    # prediction is a list
    # res = {ins['question_id']:[{'caption':pred} for pred in ins['prediction']] for ins in json_data}
    # prediction is a string
    res = {ins['question_id']:[{'caption':ins['prediction']}] for ins in json_data}

    gts  = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)
    #print(gts,res)
    
    # =================================================
    # Compute scores
    # =================================================
    rlt={}
    for scorer, method in scorers:
        eprint('computing %s score...'%(scorer.method()))
        score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                print("%s: %0.3f"%(m, sc*100))
                rlt[m]=sc*100
        else:
            print("%s: %0.3f"%(method, score*100))
            rlt[method]=score*100
    return rlt

QT=['Place','Number','Color','Object nature','Object','Other']
def qclass1(question):
    lques = question
    if 'Where' in lques:
        return 'Place'
    if 'How many' in lques:
        return 'Number'
    if 'What color' in lques or 'What is the color' in lques:
        return 'Color'
    if 'What shape' in lques:
        #return 'Shape'
        return 'Object nature'
    if 'What type' in lques:
        #return 'Type'
        return 'Object nature'
    if 'What kind' in lques:
        #return 'Kind'
        return 'Object nature'
    if 'What is' in lques:
        return 'Object'
    return 'Other'
            
if __name__=="__main__":

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--json_path", type=str, help="Folder containing the results", required=True)
    parser.add_argument('--use_spice',  help='no spice', action="store_true")
    args = parser.parse_args()

    json_data = json.load(open(args.json_path))

    score=evals_json(json_data)
    print(score)
    #print()
    print(eval_pycoco(json_data, use_spice=args.use_spice))
    print()
    print()