# BEIR scifact

5183 documents, 300 test queries, k=10.

| retriever | chunk size | hit_rate@10 | recall@10 | ndcg@10 | mrr |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | 300 | 0.753 | 0.729 | 0.590 | 0.556 |
| bm25 | 800 | 0.767 | 0.744 | 0.629 | 0.600 |
