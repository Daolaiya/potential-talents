<a id="Introduction"></a>
# Potential Talents

A small NLP project that ranks candidate job titles by their fitness to a set of
keyphrases (e.g. *"aspiring human resources"*), supports re-ranking when a
candidate is "starred", and ships a tiny Gradio demo app.

> All work lives in a single notebook: [`notebook_pt.ipynb`](./notebook_pt.ipynb).

---

## Table of contents

1. [Project brief](#project-brief)
2. [Project structure](#project-structure)
3. [Setup](#setup)
4. [Running the notebook](#running-the-notebook)
5. [Approach](#approach)
6. [Results at a glance](#results-at-a-glance)
7. [Known limitations & next steps](#known-limitations--next-steps)

---

## Project brief

> The text below is the original problem statement and is preserved verbatim
> so the notebook can be evaluated against it.

**Background:**
- As a talent sourcing and management company, we are interested in finding talented individuals for sourcing these candidates to technology companies. Finding talented candidates is not easy, for several reasons. The first reason is one needs to understand what the role is very well to fill in that spot, this requires understanding the client's needs and what they are looking for in a potential candidate. The second reason is one needs to understand what makes a candidate shine for the role we are in search for. Third, where to find talented individuals is another challenge.
- The nature of our job requires a lot of human labor and is full of manual operations. Towards automating this process we want to build a better approach that could save us time and finally help us spot potential candidates that could fit the roles we are in search for. Moreover, going beyond that for a specific role we want to fill in we are interested in developing a machine learning powered pipeline that could spot talented individuals, and rank them based on their fitness.
- We are right now semi-automatically sourcing a few candidates, therefore the sourcing part is not a concern at this time but we expect to first determine best matching candidates based on how fit these candidates are for a given role. We generally make these searches based on some keyphrases such as "full-stack software engineer", "engineering manager" or "aspiring human resources" based on the role we are trying to fill in. These keyphrases might change, and you can expect that specific keyphrases will be provided to you.
- Assuming that we were able to list and rank fitting candidates, we then employ a review procedure, as each candidate needs to be reviewed and then determined how good a fit they are through manual inspection. This procedure is done manually and at the end of this manual review, we might choose not the first fitting candidate in the list but maybe the 7th candidate in the list. If that happens, we are interested in being able to re-rank the previous list based on this information. This supervisory signal is going to be supplied by starring the 7th candidate in the list. Starring one candidate actually sets this candidate as an ideal candidate for the given role. Then, we expect the list to be re-ranked each time a candidate is starred.

**Data Description:**
- The data comes from our sourcing efforts. We removed any field that could directly reveal personal details and gave a unique identifier for each candidate.

**Attributes:**
- `id` — unique identifier for candidate (numeric)
- `job_title` — job title for candidate (text)
- `location` — geographical location for candidate (text)
- `connection` — number of connections candidate has, `500+` means over 500 (text)

**Output (desired target):**
- `fit` — how fit the candidate is for the role? (numeric, probability between 0–1)
    - keyphrases: *"Aspiring human resources"* or *"seeking human resources"*

**Download Data:**
- https://docs.google.com/spreadsheets/d/117X6i53dKiO7w6kuA1g1TpdTlv1173h_dPlJt5cNNMU/edit?usp=sharing
- A local copy is included as [`data_raw.csv`](./data_raw.csv) (104 rows).

**Goal(s):**
- Predict how fit the candidate is based on their available information (variable `fit`).

**Success Metric(s):**
- Rank candidates based on a fitness score.
- Re-rank candidates when a candidate is starred.

**Bonus(es):**
- We are interested in a robust algorithm, tell us how your solution works and show us how your ranking gets better with each starring action.
- How can we filter out candidates which in the first place should not be in this list?
- Can we determine a cut-off point that would work for other roles without losing high potential candidates?
- Do you have any ideas that we should explore so that we can even automate this procedure to prevent human bias?

---

## Project structure

```
PT/
├── notebook_pt.ipynb     # main analysis + modeling + demo app
├── data_raw.csv          # source data (104 candidates)
├── requirements.txt      # Python dependencies
└── README.md             # this file
```

---

## Setup

> Tested with **Python 3.11**.

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. download the NLTK data the notebook needs (one-time)
python -c "import nltk; [nltk.download(p) for p in ('stopwords','punkt','punkt_tab','wordnet','omw-1.4')]"
```

The first execution of the Word2Vec cell downloads the
`word2vec-google-news-300` model (~1.6 GB) via `gensim.downloader` and caches it
under `~/gensim-data/`. The sentence-transformer cell similarly downloads
`all-MiniLM-L6-v2` (~90 MB) on first use.

---

## Running the notebook

```bash
jupyter notebook notebook_pt.ipynb
```

Run the cells top-to-bottom. The final modeling cell launches a Gradio app
(`iface.launch(share=True)`). Set `share=False` if you do not want a public
tunnel.

---

## Approach

The notebook follows a standard text-similarity pipeline:

1. **Cleaning** — abbreviation expansion (`HR → Human Resources`, …), special
   character removal, deaccenting, stopword removal and lowercasing. Stemming
   and lemmatization are defined but intentionally **not** applied (they
   produced non-interpretable stems such as `resouc`).
2. **Ranking** — for each of three encoders, embed every cleaned job title and
   each keyphrase, then rank candidates by mean cosine similarity:
   - **TF-IDF** (`sklearn.feature_extraction.text.TfidfVectorizer`)
   - **Word2Vec** averaged token embeddings (`word2vec-google-news-300`)
   - **Sentence-Transformer** sentence embeddings (`all-MiniLM-L6-v2`)
3. **Filtering / cutoff** — per-method threshold chosen by visual inspection
   of the similarity histogram.
4. **Re-ranking after starring** — appending the starred candidate's
   (cleaned) job title to the keyphrase list and recomputing similarities.
5. **Demo app** — Gradio wrapper using the sentence-transformer encoder; a
   user enters comma-separated keyphrases and gets a ranked dataframe back.

---

## Results at a glance

- 104 rows / 51 exact duplicates / 0 missing values (after dropping the empty
  `fit` target).
- All three methods consistently surface the *"aspiring human resources
  professional"* / *"seeking human resources …"* titles at the top.
- Starring the candidate at the median rank (position 53) repeatedly drives
  their rank into the top-10 within 1–2 starring iterations across all three
  methods.

---

## Known limitations & next steps

- The dataset is tiny (104 rows, with 51 dupes). Any quantitative comparison
  between methods is suggestive at best.
- The `fit` column is empty in the source data, so this is **unsupervised**
  ranking; we have no ground truth to measure precision/recall.
- Cutoff thresholds are picked visually per method; a learned or calibrated
  cutoff would be more defensible.
- Re-ranking is done by appending starred titles to the keyphrase list and
  taking a simple mean. Weighted aggregation, a learning-to-rank model
  (e.g. RankNet, LambdaMART), or LLM-based re-ranking are natural extensions
  noted in the notebook's conclusion.
