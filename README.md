# VibeCheck

**Read the vibe of every review.** VibeCheck turns thousands of customer reviews into product decisions. It labels every review as positive, neutral or negative (TF-IDF + Logistic Regression), groups products into categories (TF-IDF + K-Means), and writes a short best-and-worst report per category with an LLM (NVIDIA Nemotron).

**Live app:** https://group5-ywvncykekhs7cbznt2hwef.streamlit.app

---

## Overview
Companies get far more reviews than anyone can read, and star averages hide the problems: 93% of the reviews in our data are positive, so a 4.6-star product can still have a recurring defect. VibeCheck is built for product and customer-experience teams. It is Group 5's solution to the Ironhack AI Engineering project *NLP Automated Customer Reviews*.

| Feature (app tab) | What it does |
|---|---|
| **Check a review** | Labels one review and shows the words that drove the result |
| **Analyze your reviews** | Upload a CSV of reviews: every review is labelled, and every product gets a health status (*Customers happy*, *Mixed feedback*, *Needs attention*) with its most praised and most complained-about words |
| **Category report** | An AI-written report on the strongest and weakest products in a category, checked against the data |
| **Model performance** | Test-set metrics and confusion matrix |

## Dataset
- **Source:** *Consumer Reviews of Amazon Products* by Datafiniti on Kaggle (file `1429_1.csv`).
- **Size:** 34,660 reviews of Amazon devices and accessories; **34,625 reviews of 38 products** after cleaning.
- **Labels from stars:** 1–2 = negative, 3 = neutral, 4–5 = positive. The result is very imbalanced: **93.3% positive, 4.3% neutral, 2.3% negative**.
- **Preprocessing:** invalid ratings dropped; review title + text joined and lowercased into `clean_text`; product names rebuilt from the `keys` column (most `name` values were missing).
- **Split:** 80/20, stratified: 27,700 training and 6,925 test reviews.
- The raw data is not in the repo (`Data/` is in `.gitignore`). Notebook 01 downloads it with `kagglehub`. `CSVs/reviews_slim.csv.gz` is a small compressed copy that the report writer uses.

## Method
1. **Sentiment (Task 1):** TF-IDF (1–3-grams, `min_df=2`) + Logistic Regression (`C=3`, `class_weight="balanced"`), tuned with `GridSearchCV` on **macro F1**. Cross-validated macro F1 of the candidates: Naive Bayes 0.34, Linear SVM 0.61, Logistic Regression 0.62, and **0.64 after tuning**. We optimise macro F1 because accuracy rewards ignoring the rare classes.
2. **Categories (Task 2):** TF-IDF (English stop words removed) on each product's name + Amazon category text → **K-Means, k = 5**: Tablets, Echo & Smart Speakers, Fire TV, Kindle, Chargers & Cables.
3. **Reports (Task 3):** pandas builds a fact sheet per category (products with 10+ reviews: rating, % negative, % recommend, two real complaints). A fill-in-the-template prompt asks **NVIDIA Nemotron** (`nemotron-3.5-lightning-30b-a3b`, via the NVIDIA API) to write the report. `check_article` then counts the sections, the product names and any **number that isn't in the facts**.
4. **App (Task 4):** Streamlit, deployed on Streamlit Community Cloud.

## Results
Test set: 6,925 reviews.

| | Always "positive" (baseline) | **Our model** |
|---|---|---|
| Accuracy | 93.3% | **92.4%** |
| Macro F1 | 0.32 | **0.64** |
| Weighted F1 | 0.90 | **0.93** |
| Negative reviews found (recall) | 0% | **61%** |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0.55 | 0.61 | 0.58 | 162 |
| Neutral | 0.32 | 0.44 | 0.37 | 300 |
| Positive | 0.97 | 0.95 | 0.96 | 6,463 |

Confusion matrix: [`Reports/02_confusion_matrix_tfidf.png`](Reports/02_confusion_matrix_tfidf.png). The 5 categories hold 17,596 (Tablets), 7,262 (Echo), 5,068 (Fire TV), 4,137 (Kindle) and 562 (Chargers) reviews. One report per category: [`Reports/articles.md`](Reports/articles.md).

## Demo
Open the [live app](https://group5-ywvncykekhs7cbznt2hwef.streamlit.app):
- **Check a review:** paste any review.
- **Analyze your reviews:** upload a CSV with a review column and, optionally, a product column.
- **Category report:** generate a fresh report (about 20–30 seconds).

## How to run
```bash
git clone https://github.com/Beholdingmars/VibeCheck.git
cd VibeCheck
pip install -r requirements.txt
export NVIDIA_API_KEY=...        # or put it in .streamlit/secrets.toml (never commit it)
streamlit run app.py
```
To rebuild the models and reports, run the notebooks in `Source/` in order (01 → 04).

## Project structure
```
VibeCheck/
├── app.py                  Streamlit app
├── article_writer.py       Category reports: facts, prompt, NVIDIA call, checks
├── vibecheck_theme.py      Brand styles shared by the app
├── .streamlit/config.toml  App theme
├── requirements.txt
├── Models/                 sentiment_model.joblib (TF-IDF + Logistic Regression)
├── CSVs/                   reviews_slim.csv.gz (data for the reports)
├── Reports/                confusion matrix, product categories, articles.md
└── Source/                 01 clean → 02 sentiment → 03 clustering → 04 reports
```

## Key findings and limitations
- **Stars don't separate products:** the 15 most-reviewed products all average between 4.4 and 4.8 stars. The words in the reviews do separate them.
- **Accuracy misleads** on imbalanced data: the "always positive" baseline beats us on accuracy but finds no negative reviews.
- **Neutral is the hard class** (F1 0.37): 3-star reviews mix praise and complaints.
- English Amazon device reviews only; other domains would need retraining.
- Categories come from product names, so a product with an unusual name can land in the wrong group.
- The report writer depends on an external API: it needs a key, takes 20–30 seconds, and can still round numbers. The check flags those.
- **Next:** a pretrained transformer (e.g. DistilBERT) for the neutral class, topic-level sentiment (battery, screen, price), and reports written straight from an uploaded CSV.

## Credits
- Data: Datafiniti, *Consumer Reviews of Amazon Products* (Kaggle).
- Libraries: scikit-learn, pandas, Plotly, Streamlit, OpenAI Python client.
- LLM: NVIDIA Nemotron via the NVIDIA API.
- Brief: Ironhack AI Engineering, *NLP Automated Customer Reviews*.
- Team: Group 5, Jose Silvera & Salma Charfi.
