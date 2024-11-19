# CCSum Official Repo Explained

Link to the official repo of CCSum: [hnthap/CCSum](https://github.com/hnthap/CCSum) (forked)

Table of Contents:

- [CCSum Official Repo Explained](#ccsum-official-repo-explained)
  - [CCSum Generation Pipeline](#ccsum-generation-pipeline)
    - [Overall pipeline](#overall-pipeline)
    - [Heuristics filters](#heuristics-filters)
    - [Factual consistency filters](#factual-consistency-filters)
    - [Coverage filters](#coverage-filters)
    - [Abstractiveness filters](#abstractiveness-filters)
  - [CCSum Evaluation Pipeline](#ccsum-evaluation-pipeline)
  - [Dependencies](#dependencies)
  - [File Structure](#file-structure)
    - [`filtering.py`](#filteringpy)
    - [`data_util.py`](#data_utilpy)
    - [`data.py`](#datapy)
    - [`entity_util.py`](#entity_utilpy)
    - [`factuality.py`](#factualitypy)
    - [`llm_filter.py`](#llm_filterpy)
    - [`llm_models.py`](#llm_modelspy)
    - [`mint.py`](#mintpy)
    - [`regex_util.py`](#regex_utilpy)
    - [`retrieval.py`](#retrievalpy)
    - [`similarity_util.py`](#similarity_utilpy)
    - [`text_util.py`](#text_utilpy)

## CCSum Generation Pipeline

### Overall pipeline

**Step 1.** Clustering: [`data.py`](#datapy), [`data_util.py`](#data_utilpy), [`text_util.py`](#text_utilpy)

$$\text{CommonCrawl-News Corpus} \xrightarrow{\text{$n$-day windows}} \text{Article Cluster } C$$

**Step 2.** Generating article-summary pairs: [`data_util.py`](#data_utilpy), [`retrieval.py`](#retrievalpy)

$$\text{Article Cluster } C \xrightarrow{\text{sentence-BERT}}\, \xrightarrow{\text{soft-cluster}} \text{Candidate Pairs } \hat{\mathcal{D}}$$

**Step 3.** Filtering: [`filtering.py`](#filteringpy)

$$\text{Candidate Pairs } \hat{\mathcal{D}} \xrightarrow{\text{Filters } \mathcal{F}} \text{CCSum } \mathcal{D}$$

### Heuristics filters

- $f_{\text{domain}}$: The summary must come from a domain different than the article's domain. [`filtering.py`](#filteringpy)
- $f_{\text{entity count}}$: The summary must contain at least one entity. [`filtering.py`](#filteringpy)
- $f_{\text{word count}}$: The summary must contain at least 25 words. [`filtering.py`](#filteringpy)
- $f_{\text{punctuation}}$: The summary must end with proper punctuations. [`filtering.py`](#filteringpy)
- $f_{\text{not in}}$: The summary must not be included in the corresponding article. [`filtering.py`](#filteringpy)

### Factual consistency filters

- $f_{\text{ep}}$ (Entity Precision filter): The summary's entities must also appear in the corresponding article.  [`entity_util.py`](#entity_utilpy), [`filtering.py`](#filteringpy), [`text_util.py`](#text_utilpy)
- $f_{\text{BS-P}}$ (BERTScore Precision filter): The summary and its corresponding article must be factually consistent. [`factuality.py`](#factualitypy), [`filtering.py`](#filteringpy)
- $f_{\text{quo}}$: The summary's quoted text must present as in the article. [`filtering.py`](#filteringpy), [`regex_util.py`](#regex_utilpy)

### Coverage filters

- $f_{\text{BS-R}}$ (BERTScore Recall filter) [`factuality.py`](#factualitypy)
- $f_{\text{t-t}}$ (Title-title similarity filter): The title of the summary's original article and the title of the paired article must be semantically similar. [`filtering.py`](#filteringpy)
- $f_{\text{s-t}}$ (Summary-title similarity filter): The summary and the title of the paired article must be semantically similar. [`filtering.py`](#filteringpy)

### Abstractiveness filters

- $f_{\text{MINT}}$ (MINT filter) [`filtering.py`](#filteringpy), [`mint.py`](#mintpy)
- $f_{\text{Simhash}}$ (Simhash filter) [`filtering.py`](#filteringpy)

## CCSum Evaluation Pipeline

> 🚧 This section is under construction.

## Dependencies

- `sentence-transformer` (used in [`data_util.py`](#data_utilpy))
- `pseudo_ref_research` (used in [`data.py`](#datapy))
- `bert_score` (used in [`factuality.py`](#factualitypy))
- `simhash` (used in [`filtering.py`](#filteringpy))
- `transformers` (used in [`llm_models.py`](#llm_modelspy))
- `spacy`, `pylcs` (used in [`mint.py`](#mintpy), [`similarity_util.py`](#similarity_utilpy))
- `faiss` (used in [`retrieval.py`](#retrievalpy))
- Other dependencies: `numpy`, `pandas`, `scipy`, `sklearn`.

## File Structure

```text
ccsum
|---- ccsum
|      |---- __init__.py
|      |---- data_util.py
|      |---- data.py
|      |---- entity_util.py
|      |---- factuality.py
|      |---- filtering.py
|      |---- llm_filter.py
|      |---- llm_models.py
|      |---- mint.py
|      |---- regex_util.py
|      |---- retrieval.py
|      |---- similarity_util.py
|      |---- text_util.py
|
|---- data
|---- setup.py
```

### `filtering.py`

| Function | Description |
| -------- | ----------- |
| ⭐ `apply_filters` | Apply all filters at once. The ultimate CCSum filtering function. |
| `ending_punctuation` | Apply $f_{\text{punctuation}}$. |
| `summary_title_different_domains` | Apply $f_{\text{domain}}$. |
| `summary_word_count` | Apply $f_{\text{word count}}$. |
| `summary_at_least_one_entity` | Apply $f_{\text{entity count}}$. |
| `summary_not_in_article` | Apply $f_{\text{entity count}}$. |
| `entity_precision_hard` | Apply the *hard* $f_{\text{ep}}$. |
| `entity_precision_soft` | Apply the *soft* $f_{\text{ep}}$. |
| `simhash_distance` | Calculate the distance between the main text of the summary and its corresponding article, for $f_{\text{Simhash}}$. |
| `simhash_filter` | Apply $f_{\text{Simhash}}$. |
| `quotation_filter` | Apply $f_{\text{quo}}$. |
| `title_title_sim` | Apply $f_{\text{t-t}}$. |
| `summary_title_similarity` | Apply $f_{\text{s-t}}$. |
| `bert_scores_bert_filter` | Apply $f_{\text{BS-P}}$ and $f_{\text{BS-R}}$, sequentially, using the BERT model. |
| `bert_scores_bart_filter` | Apply $f_{\text{BS-P}}$ and $f_{\text{BS-R}}$, sequentially, using the BART model. |
| `mint_filter` | Apply $f_{\text{MINT}}$. |

### `data_util.py`

This file contains some data utility functions.

| Function | Description |
| -------- | ----------- |
| `encode_sentence_transformer` | Encode the articles to embeddings using the `sentence-transformers/all-mpnet-base-v2` model. |
| `index_by_date_and_sort_index` | Index and sort a data frame according to the date field. |
| `sliding_window` | Create the $n$-day windows based on the publication date field. |

### `data.py`

This file contains functions needed for manipulating data.

This file provides the class `SummaryArticlePairDataset` to store the summary-article pairs efficiently.

This file also provides some utility functions:

| Function | Description |
| -------- | ----------- |
| `load_jsonl` | Load a list of objects from a JSONL file. (UTF-8 encoding) |
| `load_json` | Load an object from a JSON file. (UTF-8 encoding) |
| `load_json_from_filelist` | Load a list of objects from a list of JSON files. |
| `load_jsonl_from_dir` | Load a list of objects from a directory of JSON files. |
| `csv_to_jsonl` | Convert a CSV file to a JSONL file. |
| `list_to_jsonl` | Save a Python `list` to a JSONL file. |
| `dict_to_json` | Save a Python `dict` to a JSON file. |
| `get_content_between_quotes` | Find all the texts between double quotes from provided text. |
| `get_content_between_outermost_quotes` | Find the one text between the first and last double quotes from provided text. |
| `escape_quotes` | Replace `"` with `\\"` to escape the double quotes. |

### `entity_util.py`

| Function | Description |
| -------- | ----------- |
| `normalize_entity` | Rewrite some pre-defined entity names, e.g, replacing "US" with "United States". |
| `get_entity_from_doc` | Retrieve the entity names from a `doc` object. See more: [`text_util.py`](#text_utilpy). |
| `get_entities` | Retrieve all entities from a list of texts. See more: [`text_util.py`](#text_utilpy). |
| `evaluate_entity_precision_constraint` | Measure a score used to filter with $f_{\text{ep}}$, on a list of allowed entity types. (so-called hard entity precision) |
| `evaluate_entity_precision` | Measure a score used to filter with $f_{\text{ep}}$. (so-called soft entity precision) |

### `factuality.py`

| Function | Description |
| -------- | ----------- |
| `evaluate_bert_score` | Evaluate BERTScore on two lists of candidates and references for $f_{\text{BS-P}}$ and $f_{\text{BS-R}}$. |
| `evaluate_bert_score_multibatch` | Evaluate BERTScore on two lists of candidates and references, on multiple batches, for $f_{\text{BS-P}}$ and $f_{\text{BS-R}}$. |

### `llm_filter.py`

> 🚧 This section is under construction.

### `llm_models.py`

> 🚧 This section is under construction.

### `mint.py`

> 🚧 This section is under construction.

| Function | Description |
| -------- | ----------- |
| `ngrams_all` | Create a set of 1-grams, 2-grams, ..., n-grams from the provided list of tokens. |
| `ngram_overlaps` | 🚧 |
| `mint` | 🚧 |
| `process_batch` | Calculate the MINT score of each pair in specified batch. |
| `evaluate_mint` | 🚧 |
| `evaluate_mint_on_df` | 🚧 |

### `regex_util.py`

| Function | Description |
| -------- | ----------- |
| `clean_sentence` | ⚠️ English-specific! Make a sentence cleaner in some specific ways. |
| `remove_byline` | Remove some unnecessary phrases. |
| `get_quotes` | Extract quoted strings. |
| `evaluate_quote_precision` | Evaluate a quote precision score for $f_{\text{quo}}$. |

### `retrieval.py`

| Function | Description |
| -------- | ----------- |
| `retrieve_soft_clusters` | Perform soft clustering on a specific window. |
| `retrieve_from_one_doc` | Search for the top 10 summaries that are most similar to the document embedding. |

### `similarity_util.py`

| Function | Description |
| -------- | ----------- |
| `get_summary_article_similarity_at_sentence_level` | Yes. |
| `get_similarity` | Calculate cosine similarity scores between two lists of texts. |

### `text_util.py`

| Function | Description |
| -------- | ----------- |
| `construct_summary_article_pairs` | Construct $\hat{\mathcal{D}}$, i.e., construct all summary-article pairs from all clusters. |
| `construct_summary_article_pairs_within_cluster` | Construct all summary-article pairs within a cluster. |
| `extract_lead_sentence` | Extract the first sentence of the specified article. |
| `evaluate_entity_precision` | Evaluate the entity precision. See more: [`entity_util.py`](#entity_utilpy) |
