# SPLADE Tiny MSMARCO

This repo contains Python code to train SPLADE sparse retrieval models based on BERT-Tiny (4M params) and BERT-Mini (11M params) by distilling a Cross-Encoder on the MSMARCO dataset. The cross-encoder used was [ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2). 

The tiny SPLADE models beat `BM25` by `65.6 - 76.3%` on the MSMARCO benchmark. While these models are `6-15x` smaller than Naver's official `splade-v3-distilbert`, they posess `80-85%` of it's performance on the MSMARCO benchmark. The tiny SPLADE models are small enough to be used without a GPU on a dataset of a few thousand documents. 

You can download the models from the following huggingface collection.

- Models: https://huggingface.co/collections/rasyosef/splade-tiny-msmarco-687c548c0691d95babf65b70
- Distillation Dataset: https://huggingface.co/datasets/yosefw/msmarco-train-distil-v2

## Performance

The splade models were evaluated on 55 thousand queries and 8 million documents from the [MSMARCO](https://huggingface.co/datasets/microsoft/ms_marco) dataset.

||Size (# Params)|MRR@10 (MS MARCO dev)|
|:---|:----|:-------------------|
|`BM25`|-|18.6|-|-|
|`rasyosef/splade-tiny`|4.4M|30.8|
|`rasyosef/splade-mini`|11.2M|32.8|
|`naver/splade-v3-distilbert`|67.0M|38.7|

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
# tensor([[30.0340,  6.7401,  0.0545],
#        [ 6.7401, 24.6187,  0.3883],
#        [ 0.0545,  0.3883, 19.0308]])

# Let's decode our embeddings to be able to interpret them
decoded = model.decode(embeddings, top_k=10)
for decoded, sentence in zip(decoded, sentences):
    print(f"Sentence: {sentence}")
    print(f"Decoded: {decoded}")
    print()
```

```
Sentence: The weather is lovely today.
Decoded: [('today', 2.360283851623535), ('weather', 2.1073269844055176), ('lovely', 1.9814589023590088), ('beautiful', 1.5150052309036255), ('morning', 1.4209978580474854), ('nice', 1.3863301277160645), ('summer', 1.3625599145889282), ('wonderful', 1.190387487411499), ('cozy', 0.9536394476890564), ('now', 0.8671116232872009)]

Sentence: It's so sunny outside!
Decoded: [('sunny', 2.3126473426818848), ('outside', 2.303380250930786), ('so', 2.0214669704437256), ('weather', 1.5352340936660767), ('it', 1.2967190742492676), ('inside', 1.1312494277954102), ('today', 0.954143762588501), ('south', 0.9383463263511658), ('summer', 0.7259277105331421), ('sandy', 0.663248598575592)]

Sentence: He drove to the stadium.
Decoded: [('stadium', 2.2020630836486816), ('drove', 2.0168163776397705), ('he', 1.5912643671035767), ('drive', 1.3636406660079956), ('walked', 1.0576399564743042), ('crowd', 0.9113193154335022), ('pulled', 0.8589119911193848), ('.', 0.8554607033729553), ('ran', 0.6225103735923767), ('team', 0.5971767902374268)]
```
