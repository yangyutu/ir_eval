import logging
from typing import Dict, List
import math
# reference: https://github.com/eXascaleInfolab/pytrec_eval/blob/master/pytrec_eval/metrics.py

def recall(
    qrels: Dict[str, Dict[str, float]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
    relevance_score_thresh: float = 0.0
) -> Dict[str, float]:
    
    recalls = {}
    
    for k in k_values:
        recalls[f"Recall@{k}"] = 0.0
        
    k_max = max(k_values)
    
    logging.info("\n")
    valid_qrel_count = 0
    
    for query_id, doc_scores in results.items():
        # sort docs by predicted scores descendly
        top_hits = sorted(doc_scores.items(), key=lambda item:item[1], reverse=True)[0:k_max]
        query_relevant_docs = [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > relevance_score_thresh]
        if len(query_relevant_docs) == 0:
            continue
        valid_qrel_count += 1
        for k in k_values:
            retrieved_doc_score_at_k = top_hits[:k]
            retrieved_doc_TP = [doc_id for doc_id, _ in retrieved_doc_score_at_k if qrels[query_id].get(doc_id,0) > relevance_score_thresh]
            #denominator = min(len(query_relevant_docs), k)
            denominator = len(query_relevant_docs)
            recalls[f"Recall@{k}"] += len(retrieved_doc_TP) / denominator
            
    for k in k_values:
        recalls[f"Recall@{k}"] = round(recalls[f"Recall@{k}"] / valid_qrel_count, 5)
        logging.info(f'Recall@{k}: {recalls[f"Recall@{k}"]}')
    
    return recalls

def hole(
    qrels: Dict[str, Dict[str, float]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int]
) -> Dict[str, float]:
    """
    hole rate compute the number of document without annotation
    """
    
    holes = {}
    for k in k_values:
        holes[f"Hole@{k}"] = 0.0

    annotated_corpus = set()
    for _, doc_score in qrels.items():
        for doc_id, _ in doc_score.items():
            annotated_corpus.add(doc_id)
            
    k_max = max(k_values)
    logging.info("\n")

    for _, doc_scores in results.items():
        top_hits = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]
        for k in k_values:
            hole_docs = [row[0] for row in top_hits[0:k] if row[0] not in annotated_corpus]
            hole[f"Hole@{k}"] += len(hole_docs) / k      
    for k in k_values:
        holes[f"Hole@{k}"] = round(holes[f"Hole@{k}"] / len(qrels), 5)
        logging.info(f'Hole@{k}: {holes[f"Hole@{k}"]}')
    
    return holes
    
def precision(
    qrels: Dict[str, Dict[str, float]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
    relevance_score_thresh: float = 0.0
) -> Dict[str, float]:
    
    precisions = {}
    
    for k in k_values:
        precisions[f"Precision@{k}"] = 0.0
        
    k_max = max(k_values)
    
    logging.info("\n")
    valid_qrel_count = 0
    
    for query_id, doc_scores in results.items():
        # sort docs by predicted scores descendly
        top_hits = sorted(doc_scores.items(), key=lambda item:item[1], reverse=True)[0:k_max]
        query_relevant_docs = [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > relevance_score_thresh]
        if len(query_relevant_docs) == 0:
            continue
        valid_qrel_count += 1
        for k in k_values:
            retrieved_doc_score_at_k = top_hits[:k]
            retrieved_doc_TP = [doc_id for doc_id, _ in retrieved_doc_score_at_k if qrels[query_id].get(doc_id,0) > relevance_score_thresh]
            denominator = k
            precisions[f"Precision@{k}"] += len(retrieved_doc_TP) / denominator
            
    for k in k_values:
        precisions[f"Precision@{k}"] = round(precisions[f"Precision@{k}"] / valid_qrel_count, 5)
        logging.info(f'Precision@{k}: {precisions[f"Precision@{k}"]}')
    
    return precisions


def ndcg(
    qrels: Dict[str, Dict[str, float]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
    relevance_score_thresh: float = 0.0
) -> Dict[str, float]:
    
    ndcgs = {}
    
    for k in k_values:
        ndcgs[f"NDCG@{k}"] = 0.0
        
    k_max = max(k_values)
    
    logging.info("\n")
    valid_qrel_count = 0
    
    for query_id, doc_scores in results.items():
        # sort docs by predicted scores descendly
        top_hits = sorted(doc_scores.items(), key=lambda item:item[1], reverse=True)[0:k_max]
        query_relevant_docs = [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > relevance_score_thresh]
        query_predict_doc_rank = [(doc_id, qrels[query_id].get(doc_id, 0)) for doc_id, _ in top_hits]
        
        query_perfect_doc_rank = sorted(qrels[query_id].items(), key=lambda item:item[1], reverse=True)[0:k_max]
        if len(query_relevant_docs) == 0:
            continue
        valid_qrel_count += 1
        for k in k_values:
            query_predict_doc_rank_at_k = query_predict_doc_rank[:k]
            query_perfect_doc_rank_at_k = query_perfect_doc_rank[:k]
            dcg = query_predict_doc_rank_at_k[0][1] + sum([doc_relScore[1] / math.log2(rank)
                                            for rank, doc_relScore in enumerate(query_predict_doc_rank_at_k[1:], start=2)])
            
            ideal_dcg = query_perfect_doc_rank_at_k[0][1] + sum([doc_relScore[1] / math.log2(rank)
                                            for rank, doc_relScore in enumerate(query_perfect_doc_rank_at_k[1:], start=2)])
            
            
            ndcg_val = dcg / ideal_dcg if ideal_dcg > 0 else 0
            
            ndcgs[f"NDCG@{k}"] += ndcg_val
            
    for k in k_values:
        ndcgs[f"NDCG@{k}"] = round(ndcgs[f"NDCG@{k}"] / valid_qrel_count, 5)
        logging.info(f'NDCG@{k}: {ndcgs[f"NDCG@{k}"]}')
    
    return ndcgs
