# BEIR fiqa

57600 documents, 648 test queries, k=10.

| retriever | chunk size | hit_rate@10 | recall@10 | ndcg@10 | mrr |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | 300 | 0.384 | 0.228 | 0.180 | 0.235 |
| bm25 | 800 | 0.429 | 0.262 | 0.210 | 0.268 |
