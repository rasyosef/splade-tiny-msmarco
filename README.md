# SPLADE Tiny MSMARCO

This repo contains Python code to train SPLADE sparse retrieval models based on BERT-Tiny (4M params), BERT-Mini (11M params), and BERT-Small (28.8M params) by distilling a Cross-Encoder on the MSMARCO dataset. The cross-encoder used was [ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2). 

The tiny SPLADE models beat `BM25` by `65.6 - 76.3%` on the MSMARCO benchmark. Even though `splade-mini` and `splade-tiny` are `6-15x` smaller than Naver's official [splade-v3-distilbert](https://huggingface.co/naver/splade-v3-distilbert), they retain `80-85%` of it's performance on MSMARCO, all while producing sparser embedding vectors with `34-45%` fewer active dimensions. The tiny SPLADE models are small enough to be used without a GPU on a dataset of a few thousand documents. 

You can download the models from the following huggingface collection.

- Models: https://huggingface.co/collections/rasyosef/splade-tiny-msmarco-687c548c0691d95babf65b70
- Distillation Dataset: https://huggingface.co/datasets/yosefw/msmarco-train-distil-v2

## Performance

The splade models were evaluated on 55 thousand queries and 8.84 million documents from the [MSMARCO](https://huggingface.co/datasets/microsoft/ms_marco) dataset.

||Size (# Params)|Embedding Type|MRR@10 (MS MARCO dev)|Recall@10|Corpus Active Dims|
|:-|:------------|:-------------|:--------------------|:--------|:-----------------|
|**`BM25`**|-|-|18.0|37.8|-|
|**`rasyosef/splade-tiny`**|4.4M|sparse|30.9|55.4|127.1|
|**`rasyosef/splade-mini`**|11.2M|sparse|33.2|58.8|106.5|
|**`rasyosef/splade-small`**|28.8M|sparse|35.2|62.1|179.2|
|**`naver/splade-v3-distilbert`**|67.0M|sparse|38.7|66.8|192.3|
|**`Snowflake/snowflake-arctic-embed-s`**|33.2M |dense|33.7|60.7|384|
|**`intfloat/e5-small-v2`**|33.4M|dense|34.4|61.8|384|
|**`Snowflake/snowflake-arctic-embed-m-v1.5`**|109.0M|dense|35.2|63.6|768|

## Sample Inference Code

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.

```python
from sentence_transformers import SparseEncoder

# Download from the 🤗 Hub
model = SparseEncoder("rasyosef/splade-tiny")

# Run inference
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# (3, 30522)

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[39.7253,  7.1662,  0.0000],
#         [ 7.1662, 27.0255,  0.1385],
#         [ 0.0000,  0.1385, 26.3539]])

# Let's decode our embeddings to be able to interpret them
decoded = model.decode(embeddings, top_k=10)
for decoded, sentence in zip(decoded, sentences):
    print(f"Sentence: {sentence}")
    print(f"Decoded: {decoded}")
    print()
```

```
Sentence: The weather is lovely today.
Decoded: [('today', 2.543731451034546), ('lovely', 2.1207380294799805), ('weather', 2.043243646621704), ('summers', 2.0363612174987793), ('cool', 1.8053990602493286), ('darling', 1.4539366960525513), ('now', 1.3975915908813477), ('beautiful', 1.3838205337524414), ('nice', 1.2771646976470947), ('worthy', 1.2120126485824585)]

Sentence: It's so sunny outside!
Decoded: [('outside', 2.2667503356933594), ('sunny', 2.188624382019043), ('cool', 1.8421072959899902), ('so', 1.8326992988586426), ('ahead', 1.439140796661377), ('darling', 1.3871415853500366), ('it', 1.2396169900894165), ('across', 0.9793394804000854), ('sunshine', 0.9226517081260681), ('rocky', 0.8372038006782532)]

Sentence: He drove to the stadium.
Decoded: [('drove', 2.0859971046447754), ('stadium', 2.0446298122406006), ('he', 1.7063332796096802), ('team', 1.4266990423202515), ('move', 1.3472365140914917), ('jumped', 1.1752349138259888), ('driving', 1.1558808088302612), ('ride', 1.1327213048934937), ('run', 1.0909342765808105), ('drive', 1.0640281438827515)]
```
