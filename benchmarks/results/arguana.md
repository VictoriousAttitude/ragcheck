# BEIR arguana

8674 documents, 1401 test queries, k=10.

| retriever | chunk size | hit_rate@10 | recall@10 | ndcg@10 | mrr |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | 300 | 0.484 | 0.484 | 0.171 | 0.080 |
| bm25 | 800 | 0.693 | 0.693 | 0.293 | 0.171 |
