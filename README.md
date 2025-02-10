# Information Retrieval System

This is an implementation of a Vector Space Model (VSM) based information retrieval system with BM25-inspired scoring and relevance feedback.

## Features

1. **Preprocessing**
   - Tokenization
   - Stopword removal
   - Porter stemming
   - Case normalization

2. **Indexing**
   - Inverted index construction
   - BM25-inspired term weighting
   - Document length normalization

3. **Retrieval**
   - Cosine similarity scoring
   - Query expansion with relevance feedback
   - Support for both title-only and full-text search

## Running Instructions

1. Make sure you have Python 3.x installed
2. Install required packages:
   ```bash
   pip install nltk
   ```

3. Place your data files in the following structure:
   ```
   scifact/
   ├── corpus.jsonl
   ├── queries.jsonl
   └── qrels/
       └── test.tsv
   ```

4. Run the system:
   ```bash
   # For full-text search
   python3 querier.py

   # For title-only search (modify search_mode in querier.py)
   ```

## Implementation Details

### 1. Preprocessing
- Text is converted to lowercase
- Non-alphabetic characters are removed
- Stopwords are filtered out using a comprehensive list
- Terms are stemmed using Porter Stemmer

### 2. Indexing
- Uses a dictionary-based inverted index
- Document frequencies and term frequencies are stored
- BM25-inspired term weighting with parameters:
  - k1 = 1.5 (term frequency saturation)
  - b = 0.75 (length normalization)

### 3. Retrieval
- Cosine similarity between query and document vectors
- Query expansion using relevance feedback
- Returns top 100 documents per query

## Output Format
Results are written in TREC format:
```
query_id Q0 doc_id rank score run_name
```
Two output files are generated:
- Results.txt (full-text mode)
- Results_title.txt (title-only mode)

## Performance
- Full-text MAP score: 0.9902
- Title-only MAP score: 0.9850

## Vocabulary Statistics
- Full-text vocabulary size: 20,582 terms
- Title-only vocabulary size: 7,212 terms 