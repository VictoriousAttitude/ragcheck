# BEIR scidocs

25657 documents, 1000 test queries, k=10.

| retriever | chunk size | hit_rate@10 | recall@10 | ndcg@10 | mrr |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | 300 | 0.438 | 0.132 | 0.129 | 0.242 |
| bm25 | 800 | 0.464 | 0.149 | 0.141 | 0.253 |
