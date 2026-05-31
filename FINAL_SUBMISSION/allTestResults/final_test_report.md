# Final Test Report

## Overview
This report summarizes retrieval, answer quality, and RAG reliability tests.

## Retrieval Metrics
{
  "recall@1": 1.0,
  "recall@3": 1.0,
  "recall@5": 1.0,
  "hit_rate@1": 1.0,
  "hit_rate@3": 1.0,
  "hit_rate@5": 1.0,
  "mrr": 1.0,
  "total_questions": 10
}

## Answer Quality Metrics
{
  "exact_match": 0.0,
  "f1": 0.07526962001497396,
  "bleu": 0.0,
  "rouge_1": 0.11310653806977336,
  "rouge_2": 0.019857459112677327,
  "rouge_l": 0.0991382841015194,
  "total_questions": 10
}

## RAG Reliability Metrics
{
  "source_match_accuracy": 1.0,
  "ood_fallback_accuracy": 0.3333333333333333,
  "total_questions": 13
}

## Key Observations
- Retrieval metrics are expected to be strong for in-domain questions.
- OOD fallback accuracy reflects how often the model abstains appropriately.

## Strengths
- Local offline pipeline with reproducible evaluation.

## Weaknesses
- OOD detection remains challenging in System 1.

## Recommended Next Steps
- Consider reranking, better embeddings, or calibrated confidence in System 2.