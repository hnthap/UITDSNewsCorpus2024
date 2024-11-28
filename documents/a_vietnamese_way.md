<!-- markdownlint-disable MD033 -->

# A Vietnamese Way

November 28, 2024

## Explained in Detail

| Filter | English-specific |
| ------ | ---------------- |
| Ending punctuation | No. |
| Summary-article domain difference | No. |
| Summary word count | Yes. See [note 1](#note-1). |
| Summary must has at least one entity | Yes. See [note 2](#note-2). |
| Entity precision (hard) | Yes. See [note 2](#note-2). |
| Entity precision (soft) | Yes. See [note 2](#note-2). |
| Simhash | No. |
| Quotation | No. |
| Title-title similarity | Yes. See [note 3](#note-3). |
| Summary-title similarity | Yes. See [note 3](#note-3). |
| BERTScore (BERT) | Yes. See [note 4](#note-4). |
| BERTScore (BART) | Yes. See [note 4](#note-4). |
| MINT | No. |

### Note 1

Appendix C.1.3 (Title and Content Filter):

> We apply a length filter to ensure each article’s
> title is between 5 and 25 words and the main text
> contains at least 50 words. We also remove exact
> duplicates when two articles contain the same main
> text or when two articles have the same title and
> the first 200 characters of the main text are also
> identical. This process resulted in 35 million news
> articles.

I concluded that the CCSum authors did not base on any basics to choose the constraint value (5 to 25 words for title, and at least 50 words for main text), because nowhere did they state that.

I propose these options for Vietnamese:

- If counting words (e.g., "an ninh" as 1 word), the constraint is the same as CCSum.
- If counting syllables (e.g., "an ninh" as 2 syllables), the constraint is: 5 to 40 syllables for title, and at least 80 syllables for main text. (We can modify the constraint, then evaluate to determine which dataset is the best.)

### Note 2

The CCSum authors do not mention how they find the entities from text, neither in the paper nor in the code. But they implement a function `get_entities()` in the file `entity_util.py`, that returns a list of entities from some specified text. This function's first argument is the `nlp` object created in this code:

```python
# File: ccsum/text_util.py

nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser"])
nlp.add_pipe("sentencizer")
```

```python
# File: ccsum/entity_util.py

def get_entities(nlp, texts, n_process=16):
    docs = nlp.pipe(texts, n_process=n_process, batch_size=64)
    entities = []
    entity_types = []
    for doc in tqdm.tqdm(docs, total=len(texts)):
        entities.append(get_entities_from_doc(doc))
        entity_types.append(get_entity_types_from_doc(doc))
    return entities, entity_types
```

I concluded that they loaded the English-specific `en_core_web_sm` tokenizer from the [spaCy package](https://spacy.io/usage/models), which also supports Vietnamese (the package [simply use `pyvi`](https://github.com/explosion/spaCy/blob/master/spacy/lang/vi/__init__.py)).

I propose to use this package, too, but this is untested for Vietnamese. I will try to use it, and if it does not work, I will try something else.

### Note 3

The title-title and summary-title filters use cosine similarity of [sentenceBERT](https://arxiv.org/abs/1908.10084) embeddings.

There is a straightforward Vietnamese solution using the package [sentence-transformer](https://huggingface.co/keepitreal/vietnamese-sbert):

```python
from sentence_transformers import SentenceTransformer

sentences = ["Cô giáo đang ăn kem", "Chị gái đang thử món thịt dê"]

model = SentenceTransformer('keepitreal/vietnamese-sbert')
embeddings = model.encode(sentences)
print(embeddings)
```

### Note 4

The CCSum authors calculate [BERTScore](https://github.com/Tiiiger/bert_score) using pre-trained weights `bert-large-uncased`. 

The `bert_score` package supports multiple languages, [based on specific models](https://github.com/Tiiiger/bert_score?tab=readme-ov-file#default-layers). I don't know if there are available models for Vietnamese.

I propose to investigate the list of models to find some that support Vietnamese, then use one or some of them in our pipeline.
