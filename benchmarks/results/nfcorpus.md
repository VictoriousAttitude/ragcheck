# BEIR nfcorpus

3633 documents, 323 test queries, k=10.

| retriever | chunk size | hit_rate@10 | recall@10 | ndcg@10 | mrr |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | 300 | 0.644 | 0.131 | 0.252 | 0.475 |
| bm25 | 800 | 0.656 | 0.138 | 0.259 | 0.470 |
