# Agentic Retrieval Benchmark

[ English ](./README.md) | [ 中文 ](./README_zh.md)

## TL;DR
- A reproducible benchmark for LLM-augmented retrieval across Multi-CPR and LexRAG.
- Supports rewriting with DeepSeek-V3 and glm-4-plus, plus embeddings from Gemini, GLM, and Qwen models.
- Uses MongoDB Atlas for storage, indexing, and evaluation of BM25 and vector retrieval pipelines.
- Main finding: dense retrieval usually outperforms BM25; rewriting improves Recall on LexRAG by more than 50% but can slightly hurt some Multi-CPR settings, possibly because the model has already been fine-tuned on the corresponding dataset.

## Introduction
With the breakthroughs of large language models (LLMs), combining them with traditional text retrieval has formed **LLM-augmented text retrieval**. We define it as applying LLMs to key retrieval stages such as query rewriting and result re-ranking, while leveraging traditional methods, to improve retrieval performance. This new paradigm can effectively alleviate problems in classical retrieval such as query ambiguity, semantic gaps, and the difficulty of multi-turn dialogue retrieval.

Current research in this area has two main shortcomings: (1) Studies are scattered and lack unified evaluation standards, making it difficult to compare methods objectively. (2) Comparison experiments are incomplete: many works do not systematically compare against traditional baselines such as BM25 and basic dense retrieval, and they rarely quantify the impact of embedding model choices or prompt design, leading to poor reproducibility.

To address these gaps, this project aims to build a standardized, reproducible benchmark for LLM-augmented text retrieval. Using unified data and pipelines, we compare traditional retrieval with "rewrite + dense retrieval" approaches and quantify how rewriting prompts and embedding models affect performance, providing a standardized reference for future research.

This project includes:

- Test data from two open datasets:
  - Multi-CPR: three application scenarios (medical, e-commerce, video); single-turn QA format
  - LexRAG: Chinese legal-consultation scenarios; multi-turn dialogue format
- Text rewriting tools based on two models:
  - DeepSeek-V3
  - glm-4-plus
- Text embedding tools based on three models:
  - Gemini-embedding-001
  - glm-embedding-3
  - Qwen text-embedding-4
- Storage, indexing, and retrieval evaluation utilities built on MongoDB Atlas

We use the classic BM25 algorithm as the lexical baseline. Dense retrieval uses cosine similarity as the baseline. The retrieval schemes are illustrated below:

![Retrieval method diagram](./retrieval_method.png)

Experimental results show that dense retrieval is overall better than BM25, especially for long texts. The choice of embedding model has a significant impact; models trained on specific datasets or formats often outperform untuned counterparts. Query rewriting also has a clear effect on results. On the LexRAG dataset, rewriting improves Recall by more than 50%, while on Multi-CPR some rewriting prompts slightly reduce performance, possibly because the model has already been fine-tuned on the corresponding dataset or optimized for that input format, and rewriting may divert model attention.

## Usage

### Requirements
- MongoDB Atlas Server
- Python == 3.10
- openai == 2.15.0
- zai-sdk == 0.2.0
- pymongo == 4.15.5
- jieba == 0.42.1

### Setup
Use Docker to deploy a MongoDB Atlas container locally. Reference: `https://www.mongodb.com/zh-cn/docs/atlas/cli/current/atlas-cli-deploy-docker/`

Create a database from the MongoDB console and modify the corresponding parameters in `mongodb_config.py` according to your database settings.

Install required Python packages:
```bash
pip install -r requirements.txt
```

### Data Preparation
Use the script `dataset_download.py` to pull our datasets from Hugging Face, or download them manually and place them in the specified directories.

```python
python ./src/dataset_download.py
```

Hugging Face link: `https://huggingface.co/datasets/PrismShadow/AgenticSearch`

From each of the three Multi-CPR scenarios we sample roughly 4,000 queries and about 10,000 passages, using the provided indices as ground truth.

For the LexRAG dataset, we select the "dialogue history + latest question" scenario, meaning each query consists of all previous turns in the conversation plus the latest user question.

Data has already been cleaned and preprocessed and can be used directly as input for rewriting and evaluation scripts.

- Query data: `./data/rawData/xxx_query.txt`
- Passage data: `./data/rawData/xxx_subset.tsv`
- Ground-truth labels/indices: `./data/qrelData/xxx_dev.tsv`

### Rewrite
The script input is a txt or tsv file containing the raw queries or passages for one scenario. The output is a json file where each object has:
- `raw_text` (original text)
- `rewrite_text` (rewritten text)
- `raw_embedding` (empty at this stage)
- `rewrite_embedding` (empty at this stage)

Examples:

Using ZAI SDK (default model: glm-4-plus) for rewriting:
```python
# --mode must match the input file type; choose prompts according to whether the input is query or passage
python src/rewrite/rewrite_zai.py --input data/rawData/medical_query.txt --output data/rewriteData/medical_query_zai.json --mode query
```

Using OpenAI SDK (default model: DeepSeek-V3) for rewriting:
```python
# --mode must match the input file type; choose prompts according to whether the input is query or passage
python src/rewrite/rewrite_openai.py --input data/rawData/medical_query.txt --output data/rewriteData/medical_query_openai.json --mode query
```

### Embedding
The embedding scripts take a rewritten json file (containing raw and rewritten text) and output a json file where each object contains:
- `raw_text`
- `rewrite_text`
- `raw_embedding`
- `rewrite_embedding`

Usage of the three embedding scripts:

Using OpenAI SDK (default model: gemini-embedding-001):
```python
python src/embedding/embedding_openai.py --input data/rewriteData/medical_query_openai.json --output data/embeddedData/medical_query_gemini.json
```

Using OpenAI SDK (default model: qwen text-embedding-4):
```python
python src/embedding/embedding_qwen.py --input data/rewriteData/medical_query_qwen.json --output data/embeddedData/medical_query_qwen.json
```

Using ZAI SDK (default model: glm-embedding-3):
```python
python src/embedding/embedding_zai.py --input data/rewriteData/medical_query_zai.json --output data/embeddedData/medical_query_zai.json
```

### Import to Database
Use `mongodb_setup.py` to import data into MongoDB Atlas. `raw-collection` stores raw passages and their embeddings; `rewrite-collection` stores rewritten passages and embeddings:
```python
python mongodb_setup.py \
    --input data/embeddedData/medical_passage_gemini.json \
    --raw-collection medical_raw_gemini \
    --rewrite-collection medical_rewrite_gemini \
    --keep-existing False # keep existing collections; default False (overwrite)
```

By default, the raw collection uses `raw_text` as the text index and `vec_raw_embedding` as the vector index; the rewrite collection uses `rewrite_text` and `vec_rewrite_embedding` respectively.

### Evaluate
We provide evaluation scripts for BM25 and dense retrieval, both using MongoDB Atlas APIs.

Metrics: MRR@10, Recall@1, Recall@5, Recall@10. During evaluation, Top-K candidates are set to 10 for all methods.

For BM25, we use jieba for tokenization before building the index. BM25 script:
```python
python bm25.py \
    --query-file data/embeddedData/medical_query_gemini.json \
    --qrels-file data/qrelData/medical_dev.tsv \
    --raw-collection medical_raw_gemini \
    --rewrite-collection medical_rewrite_gemini \
    --raw-index bm25_raw_text \
    --rewrite-index bm25_rewrite_text
```

Dense (vector) retrieval uses cosine similarity. Parameters mirror those in `bm25.py`:
```python
python vector_cos.py \
    --query-file data/embeddedData/medical_query_gemini.json \
    --qrels-file data/qrelData/medical_dev.tsv \
    --raw-collection medical_raw_gemini \
    --rewrite-collection medical_rewrite_gemini \
    --raw-index vec_raw_embedding \
    --rewrite-index vec_rewrite_embedding
```

## Result
All experimental results are shown below.

### Medical
|                             | Embedding Model        | Rewrite Model | MRR@10 | R@1    | R@5    | R@10   |
|-----------------------------|------------------------|---------------|--------|--------|--------|--------|
| BM25(raw-raw)               | None                   | None          | 0.4117 | 0.3680 | 0.4650 | 0.5160 |
| BM25(rewrite-rewrite)       | None                   | DeepSeekV3    | 0.4716 | 0.4090 | 0.5610 | 0.6100 |
| BM25(raw-rewrite)           | None                   | DeepSeekV3    | 0.4098 | 0.3610 | 0.4790 | 0.5240 |
| BM25(rewrite-raw)           | None                   | DeepSeekV3    | 0.3789 | 0.3270 | 0.4490 | 0.5050 |
| Vector(raw-raw)             | Gemini embedding-001   | None          | 0.6387 | 0.5930 | 0.7030 | 0.7460 |
| Vector(rewrite-rewrite)     | Gemini embedding-001   | DeepSeekV3    | 0.5238 | 0.4560 | 0.6080 | 0.6590 |
| Vector(raw-rewrite)         | Gemini embedding-001   | DeepSeekV3    | 0.5259 | 0.4670 | 0.6010 | 0.6450 |
| Vector(rewrite-raw)         | Gemini embedding-001   | DeepSeekV3    | 0.6178 | 0.5590 | 0.6950 | 0.7420 |
| Vector(rewrite-raw)(no COT) | Gemini embedding-001   | DeepSeekV3    | 0.5950 | 0.5390 | 0.6670 | 0.7170 |
| Vector(raw-raw)             | Qwen text-embedding-4  | None          | **0.6695** | **0.6230** | **0.7310** | 0.7700 |
| Vector(rewrite-raw)         | Qwen text-embedding-4  | DeepSeekV3    | 0.6464 | 0.5890 | 0.7140 | **0.7710** |
| Vector(rewrite-raw)(no COT) | Qwen text-embedding-4  | DeepSeekV3    | 0.6415 | 0.5880 | 0.7110 | 0.7580 |
| Vector(raw-raw)             | GLM embedding-3        | None          | 0.5179 | 0.4600 | 0.5950 | 0.6460 |
| Vector(rewrite-raw)         | GLM embedding-3        | DeepSeekV3    | 0.5897 | 0.5283 | 0.6677 | 0.7257 |

### Ecom
|                             | Embedding Model        | Rewrite Model | MRR@10 | R@1    | R@5    | R@10   |
|-----------------------------|------------------------|---------------|--------|--------|--------|--------|
| BM25(raw-raw)               | None                   | None          | 0.7048 | 0.6160 | 0.8230 | 0.8700 |
| BM25(rewrite-rewrite)       | None                   | DeepSeekV3    | 0.6945 | 0.6120 | 0.8100 | 0.8560 |
| BM25(raw-rewrite)           | None                   | DeepSeekV3    | 0.7013 | 0.6240 | 0.8020 | 0.8590 |
| BM25(rewrite-raw)           | None                   | DeepSeekV3    | 0.6837 | 0.5890 | 0.8120 | 0.8670 |
| Vector(raw-raw)             | Gemini embedding-001   | None          | 0.7799 | 0.7090 | 0.8730 | 0.9040 |
| Vector(rewrite-rewrite)     | Gemini embedding-001   | DeepSeekV3    | 0.7554 | 0.6770 | 0.8630 | 0.8900 |
| Vector(raw-rewrite)         | Gemini embedding-001   | DeepSeekV3    | 0.7822 | 0.7110 | 0.8740 | 0.9060 |
| Vector(rewrite-raw)         | Gemini embedding-001   | DeepSeekV3    | **0.7952** | **0.7290** | 0.8830 | 0.9070 |
| Vector(rewrite-raw)(no COT) | Gemini embedding-001   | DeepSeekV3    | 0.7880 | 0.7120 | 0.8870 | 0.9090 |
| Vector(raw-raw)             | Qwen text-embedding-4  | None          | 0.7782 | 0.7070 | 0.8730 | 0.9050 |
| Vector(rewrite-raw)         | Qwen text-embedding-4  | DeepSeekV3    | 0.7928 | 0.7210 | 0.8850 | 0.9150 |
| Vector(rewrite-raw)(no COT) | Qwen text-embedding-4  | DeepSeekV3    | 0.7924 | 0.7170 | **0.8900** | **0.9220** |
| Vector(raw-raw)             | GLM embedding-3        | None          | 0.7031 | 0.6080 | 0.8270 | 0.8660 |
| Vector(rewrite-raw)         | GLM embedding-3        | DeepSeekV3    | 0.7707 | 0.6943 | 0.8657 | 0.9010 |

### Video
|                             | Embedding Model        | Rewrite Model | MRR@10 | R@1    | R@5    | R@10   |
|-----------------------------|------------------------|---------------|--------|--------|--------|--------|
| BM25(raw-raw)               | None                   | None          | 0.7154 | 0.6330 | 0.8360 | 0.8690 |
| BM25(rewrite-rewrite)       | None                   | DeepSeekV3    | 0.5834 | 0.4980 | 0.6950 | 0.7650 |
| BM25(raw-rewrite)           | None                   | DeepSeekV3    | 0.6768 | 0.5900 | 0.7960 | 0.8340 |
| BM25(rewrite-raw)           | None                   | DeepSeekV3    | 0.6267 | 0.5220 | 0.7640 | 0.8180 |
| Vector(raw-raw)             | Gemini embedding-001   | None          | 0.6649 | 0.6040 | 0.7440 | 0.7810 |
| Vector(rewrite-rewrite)     | Gemini embedding-001   | DeepSeekV3    | 0.6120 | 0.5440 | 0.7030 | 0.7550 |
| Vector(raw-rewrite)         | Gemini embedding-001   | DeepSeekV3    | 0.6722 | 0.6070 | 0.7580 | 0.7930 |
| Vector(rewrite-raw)         | Gemini embedding-001   | DeepSeekV3    | 0.6574 | 0.5850 | 0.7530 | 0.7980 |
| Vector(rewrite-raw)(no COT) | Gemini embedding-001   | DeepSeekV3    | 0.6849 | 0.6150 | 0.7790 | 0.8150 |
| Vector(raw-raw)             | Qwen text-embedding-4  | None          | **0.7691** | **0.6990** | **0.8590** | **0.8930** |
| Vector(rewrite-raw)         | Qwen text-embedding-4  | DeepSeekV3    | 0.7399 | 0.6720 | 0.8330 | 0.8660 |
| Vector(rewrite-raw)(no COT) | Qwen text-embedding-4  | DeepSeekV3    | 0.7316 | 0.6670 | 0.8150 | 0.8410 |
| Vector(raw-raw)             | GLM embedding-3        | None          | 0.4964 | 0.4360 | 0.5790 | 0.6100 |
| Vector(rewrite-raw)         | GLM embedding-3        | DeepSeekV3    | 0.6721 | 0.5973 | 0.7677 | 0.7960 |

### Law
|                             | Embedding Model        | Rewrite Model | MRR@10 | R@1    | R@5    | R@10   |
|-----------------------------|------------------------|---------------|--------|--------|--------|--------|
| BM25(raw-raw)               | None                   | None          | 0.1044 | 0.0611 | 0.1595 | 0.2246 |
| BM25(rewrite-rewrite)       | None                   | DeepSeekV3    | 0.2137 | 0.1428 | 0.3087 | 0.3900 |
| BM25(raw-rewrite)           | None                   | DeepSeekV3    | 0.0823 | 0.0450 | 0.1280 | 0.1878 |
| BM25(rewrite-raw)           | None                   | DeepSeekV3    | 0.2346 | 0.1678 | 0.3218 | 0.3909 |
| Vector(raw-raw)             | Gemini embedding-001   | None          | 0.1887 | 0.1097 | 0.2935 | 0.3892 |
| Vector(rewrite-rewrite)     | Gemini embedding-001   | DeepSeekV3    | 0.2845 | 0.1975 | 0.4023 | 0.4971 |
| Vector(raw-rewrite)         | Gemini embedding-001   | DeepSeekV3    | 0.1490 | 0.0860 | 0.2347 | 0.3231 |
| Vector(rewrite-raw)         | Gemini embedding-001   | DeepSeekV3    | **0.3391** | **0.2533** | **0.4562** | 0.5274 |
| Vector(rewrite-raw)(no COT) | Gemini embedding-001   | DeepSeekV3    | 0.3259 | 0.2385 | 0.4467 | 0.5183 |
| Vector(raw-raw)             | Qwen text-embedding-4  | None          | 0.1497 | 0.0860 | 0.2318 | 0.3209 |
| Vector(rewrite-raw)         | Qwen text-embedding-4  | DeepSeekV3    | 0.2788 | 0.2005 | 0.3828 | 0.4604 |
| Vector(rewrite-raw)(no COT) | Qwen text-embedding-4  | DeepSeekV3    | 0.2687 | 0.1912 | 0.3759 | 0.4551 |
| Vector(raw-raw)             | GLM embedding-3        | None          | 0.1768 | 0.1054 | 0.2717 | 0.3769 |
| Vector(rewrite-raw)         | GLM embedding-3        | DeepSeekV3    | 0.3214 | 0.2290 | 0.4445 | **0.5326** |

## Data Examples

### Multi-CPR
Multi-CPR data is single-turn Q&A.

#### Medical
`query` is the patient's question; `passage` is the doctor's response. Example:
```
query: An adult pulled open a child's eyelid and it turned red?what are the consequences?

passage: This is likely a fingernail scratch that damaged the conjunctiva and caused bleeding. It usually heals in about a week. You can apply erythromycin eye ointment to prevent infection. First use a cold compress for 24 hours, then warm compress. Generally it's minor, so please don't worry too much. Because a child's eyes are delicate, for safety you can take the child to the hospital ophthalmology department for examination and treatment.
```

#### Ecom
`query` is a product keyword; `passage` is the full product title. Example:
```
query: wall primer

passage: Dulux white latex universal anti-alkali anti-mold interior/exterior wall primer for home use
```

#### Video
`query` is the video search keyword; `passage` is the complete video title plus extra info. Example:
```
query: cartoon collection Pleasant Goat and Big Big Wolf

passage: Pleasant Goat and Big Big Wolf full series for kids: Pleasant Goat, Beauty Goat, Warm Goat, Lazy Goat, Small Grey, Boiling Goat, Slow Goat, Grey Wolf, Red Wolf 2018 Pleasant Goat and Big Big Wolf
```

### LexRAG
A sample query?passage pair (the <dialog history> part contains earlier turns; the latest question is the user's current ask):
```
query: <dialog history> In the second year of my car installment plan I did not buy insurance through the financing company. Do they have the right to repossess the car? Monthly payments are on time. According to Article 428 of the Civil Code, the car loan contract and the car insurance contract should be handled separately. If you pay on time and do not breach the installment agreement, the company has no right to seize the car. As for insurance, unless the contract explicitly requires purchasing through the financing company, they cannot seize the car on that basis. <latest question> Should the contract specify how insurance must be purchased?

passage: Article 507 of the Civil Code: If a contract is invalid, void, revoked, or terminated, the clause on resolving disputes remains effective.
```

## Rewrite Prompt
To study factors affecting query?passage matching, we tried multiple rewriting strategies. Prompts focus on adding helpful information. To guide model thinking, we rewrite both queries and passages into a "question?answer" style so the model captures intent and adds relevant details.

The currently best-performing rewrite prompts are:

```python
REWRITE_QUERY_SYSTEM_PROMPT = (
    '''
    # Role
    You are a multi-domain retrieval assistant. Handle one line of text each time.

    # Workflow
    1 Keyword extraction
    1.1 Based on user intent, extract 1 core keyword from the original text.
    1.2 From that core keyword, imagine 3-5 other possible search keywords users might use.

    2 Question generation
    2.1 Generate a rewritten question centered on the core keyword. Do not break the original collocations; avoid deleting information.
    2.2 Using the supplemental keywords, add 2 follow-up questions that ask for different content from the rewritten question.

    3 Answer generation
    3.1 In 1 sentence, based on the core keyword and user intent, provide relevant information or common knowledge.

    # Notes
    - Input may be a single question or multi-turn dialogue in the format: <dialog history> xxx <latest question> yyy, where yyy is the newest question.
    - For multi-turn dialogue, ensure the core keyword comes from the latest question, and rewrite toward the latest question.
    - Do NOT fabricate exact numbers, drug names, dosages, statute numbers, platform names, etc.
    - Avoid deleting non-trivial keywords; prefer expansion over reduction.
    - Output must contain two sections, "Chain of Thought" and "Final Output", separated by headings <Chain of Thought> and <Final Output>. The chain of thought should briefly list reasoning steps.
    - The final output must not include labels like "original text" or "rewritten question".

    # Output format (single-line plain text)
    <Chain of Thought> keyword extraction (core keyword; supplemental keywords) 2 question generation 3 answer generation <Final Output: question: rewritten question; supplemental question A; supplemental question B answer: answer to rewritten question + additional info>
    '''
)

REWRITE_PASSAGE_SYSTEM_PROMPT = (
    '''
    # Role
    You are a multi-domain retrieval assistant. Handle one piece of text each time.

    # Workflow
    1 Original text analysis
    1.1 Identify the nature/function of the text.
    1.2 Based on that function, infer the author's intent.

    2 Keyword extraction
    2.1 From the text and intent, extract 1 core retrieval keyword.
    2.2 From that core keyword, imagine 3-5 other possible search keywords.

    3 Question generation
    3.1 From the core keyword, reason what user intent would retrieve this text; generate 1 rewritten question emphasizing rare terms or key attributes, avoiding overly high overlap with the original text.
    3.2 Using the supplemental keywords as subjects, add 1-2 extended questions with different subjects from the rewritten question.

    4 Answer generation
    4.1 Prefer retaining the original text; supplement relevant information based on rare terms. Do NOT fabricate model numbers, values, drug names, dosages, statute numbers, platform names, etc.

    # Notes
    - The core keyword should be what best fits the author's intent and what users would likely search, not merely the broadest summary term.
    - Do not delete original text; only expand.
    - Do NOT fabricate specific numbers, drug names, dosages, statute numbers, platform names, etc.
    - Output must contain two sections, "Chain of Thought" and "Final Output", separated by headings <Chain of Thought> and <Final Output>. The chain of thought should briefly list reasoning steps.
    - The final output must not include labels like "original text" or "rewritten question".

    # Output format (single-line plain text)
    <Chain of Thought> 1 text analysis (text type/function + author intent) 2 keyword extraction (core keyword; supplemental keywords) 3 question generation 4 answer generation <Final Output: question: rewritten question; supplemental question A; supplemental question B answer: original text answering the rewritten question + extended info>
    '''
)
```
