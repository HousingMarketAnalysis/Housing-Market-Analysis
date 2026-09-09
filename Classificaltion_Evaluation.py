import time
import numpy as np
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from nltk.translate.meteor_score import single_meteor_score


def Evaluation(X, Y, ranked_candidates=None):
    if X.dtype != np.str_ or Y.dtype != np.str_:
        candidates = []
        references = []
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                candidates.append(X[i,j].astype('str'))
                references.append(Y[i, j].astype('str'))
    else:
        candidates = X
        references = Y


    # 1. BERTScore F1
    P, R, F1 = bert_score(candidates, references, lang='en', verbose=False)
    bert_f1 = F1.mean().item()

    # 2. BLEU
    bleu_scores = []
    for cand, ref in zip(candidates, references):
        ref_tokens = [ref.split()]
        cand_tokens = cand.split()
        bleu_scores.append(sentence_bleu(ref_tokens, cand_tokens))
    bleu = np.mean(bleu_scores)

    # 3. ROUGE-L F1
    rouge = Rouge()
    rouge_f1_scores = []
    for cand, ref in zip(candidates, references):
        scores = rouge.get_scores(cand, ref)
        rouge_f1_scores.append(scores[0]['rouge-l']['f'])
    rouge_l_f1 = np.mean(rouge_f1_scores)

    # 4. METEOR (fix: tokenize first)
    meteor_scores = []
    for cand, ref in zip(candidates, references):
        meteor_scores.append(single_meteor_score(ref.split(), cand.split()))
    meteor = np.mean(meteor_scores)

    # 5. Exact Match
    em_scores = [int(cand.strip() == ref.strip()) for cand, ref in zip(candidates, references)]
    em = np.mean(em_scores)

    # 6. F1 score (word-level overlap)
    f1_scores = []
    for cand, ref in zip(candidates, references):
        cand_tokens = cand.split()
        ref_tokens = ref.split()
        common = set(cand_tokens) & set(ref_tokens)
        precision = len(common) / len(cand_tokens) if cand_tokens else 0
        recall = len(common) / len(ref_tokens) if ref_tokens else 0
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)
    f1_avg = np.mean(f1_scores)

    # 7. Response Time
    start = time.time()
    _ = [cand for cand in candidates]  # simulate processing
    end = time.time()
    response_time = (end - start) / len(candidates)

    # 8. Mean Reciprocal Rank (MRR)
    mrr_list = []
    if ranked_candidates is not None:
        for rank_list, ref in zip(ranked_candidates, references):
            rr = 0
            for i, cand in enumerate(rank_list, start=1):
                if cand.strip() == ref.strip():
                    rr = 1 / i
                    break
            mrr_list.append(rr)
        mrr = np.mean(mrr_list)
    else:
        mrr = 0  # fallback if ranking not provided

    metrics_array = np.array([bert_f1, bleu, rouge_l_f1, meteor, em, f1_avg, response_time, mrr])
    return metrics_array
