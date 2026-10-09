import joblib
import numpy as np
import pandas as pd
import streamlit as st
import article_writer as aw
import plotly.graph_objects as go
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import vibecheck_theme as vc

st.set_page_config(page_title="VibeCheck", page_icon="✓", layout="wide")
st.markdown(
    """
    <style>
    /* Make the tab bar use the full width */
    .stTabs [data-baseweb="tab-list"],
    .stTabs [role="tablist"] {
        width: 100%;
        display: flex;
    }
    /* Push the last tab (Model performance) to the far right */
    .stTabs [data-baseweb="tab-list"] > button:last-of-type,
    .stTabs [role="tablist"] > [role="tab"]:last-of-type {
        margin-left: auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
vc.apply_theme()

COLORS = vc.SENTIMENT
ORDER = ["negative", "neutral", "positive"]
EXAMPLES = {
    "Positive example": "Great sound quality and the battery lasts all day. Totally worth the price!",
    "Neutral example": "It works as described. Nothing special, but it does the job.",
    "Negative example": "Stopped charging after two weeks. Customer service never answered.",
}


@st.cache_resource
def load_model():
    return joblib.load("Models/sentiment_model.joblib")


model = load_model()
vectorizer, clf = model[:-1], model[-1]          # TF-IDF part and Logistic Regression part
feature_names = vectorizer.get_feature_names_out()


def donut(labels, values, center_text):
    pairs = sorted(zip(labels, values), key=lambda x: ORDER.index(str(x[0]).lower()))
    labels = [str(l).capitalize() for l, _ in pairs]
    values = [v for _, v in pairs]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.68, sort=False, direction="clockwise",
        marker=dict(colors=[COLORS[l.lower()] for l in labels], line=dict(color=vc.COLORS["ground"], width=3)),
        textinfo="percent", textfont=dict(family=vc.MONO, color=vc.COLORS["ground"]),
        hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        showlegend=True, legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
        annotations=[dict(text=center_text, x=0.5, y=0.5, showarrow=False,
                          font=dict(size=22, family=vc.FONT, color=vc.COLORS["text"]))],
    )
    return vc.style_chart(fig, height=360)


def word_contributions(text, class_name, top_n=8):
    """Which words in this review pushed the model toward class_name."""
    x = vectorizer.transform([text])
    class_idx = list(clf.classes_).index(class_name)
    contrib = x.multiply(clf.coef_[class_idx]).tocsr()
    idx, vals = contrib.indices, contrib.data
    order = np.argsort(-np.abs(vals))[:top_n]
    return pd.DataFrame({"word": feature_names[idx[order]], "weight": vals[order]}).sort_values("weight")


def contributions_chart(df, class_name):
    color = COLORS[class_name.lower()]
    fig = go.Figure(go.Bar(
        x=df["weight"], y=df["word"], orientation="h",
        marker=dict(color=[color if w > 0 else vc.COLORS["line"] for w in df["weight"]], cornerradius=6),
        hovertemplate="%{y}: %{x:+.2f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(title=dict(text=f"← against {class_name}   ·   towards {class_name} →",
                              font=dict(family=vc.MONO, size=12, color=vc.COLORS["muted"])),
                   zeroline=True, showgrid=False),
        yaxis=dict(showgrid=False, tickfont=dict(family=vc.MONO, size=13, color=vc.COLORS["text"])),
    )
    return vc.style_chart(fig, height=320)


# Thresholds for the verdict (tune these to your data)
RECOMMEND_MIN_POS = 0.70      # at least 70% positive ...
RECOMMEND_MAX_NEG = 0.15      # ... and at most 15% negative
REJECT_MIN_NEG = 0.30         # 30%+ negative -> not recommended
VERDICT_STYLE = {
    "recommend": ("Customers happy", COLORS["positive"]),
    "mixed": ("Mixed feedback", COLORS["neutral"]),
    "reject": ("Needs attention", COLORS["negative"]),
}


def key_terms(texts, class_name, top_n=4):
    """Words that most drive reviews of this class (e.g. what people complain about)."""
    if len(texts) == 0:
        return []
    X = vectorizer.transform(texts)
    class_idx = list(clf.classes_).index(class_name)
    scores = np.asarray(X.sum(axis=0)).ravel() * clf.coef_[class_idx]
    picked = []
    for i in np.argsort(-scores):
        if scores[i] <= 0 or len(picked) == top_n:
            break
        if not set(feature_names[i].split()) & ENGLISH_STOP_WORDS:     # skip filler words like "and", "my"
            picked.append(feature_names[i])
    return picked


def product_verdict(group):
    n = len(group)
    share = group["sentiment"].value_counts(normalize=True)
    pos, neu, neg = (float(share.get(c, 0)) for c in ORDER[::-1])
    praise = key_terms(group.loc[group["sentiment"] == "positive", "_text"], "positive")
    complaints = key_terms(group.loc[group["sentiment"] == "negative", "_text"], "negative")

    praise_txt = f" Most praised: **{', '.join(praise)}**." if praise else ""
    complaint_txt = f" Main complaints: **{', '.join(complaints)}**." if complaints else ""

    if pos >= RECOMMEND_MIN_POS and neg <= RECOMMEND_MAX_NEG:
        return "recommend", (f"{pos:.0%} of {n:,} reviews are positive and only {neg:.0%} negative.{praise_txt}")
    if neg >= REJECT_MIN_NEG:
        return "reject", (f"{neg:.0%} of {n:,} reviews are negative.{complaint_txt}")
    return "mixed", (f"{pos:.0%} positive, {neu:.0%} neutral, {neg:.0%} negative "
                     f"across {n:,} reviews.{praise_txt}{complaint_txt}")

# ---------------- Sidebar ----------------
tfidf = vectorizer[-1] if hasattr(vectorizer, "__getitem__") else vectorizer
ngrams = getattr(tfidf, "ngram_range", None)
class_weight = getattr(clf, "class_weight", None)
C = getattr(clf, "C", None)

with st.sidebar:
    st.header("About the model")

    with st.expander("🧠 Sentiment model", expanded=True):
        st.markdown(
            f"""
- **Method:** TF-IDF features + Logistic Regression
- **Vocabulary:** {len(feature_names):,} terms
- **N-grams:** {f"{ngrams[0]}–{ngrams[1]} words" if ngrams else "n/a"}
- **Class weight:** {class_weight or "none"}  ·  **C:** {C}
- **Classes:** negative · neutral · positive
"""
        )
        st.caption("Each review is turned into word-importance scores (TF-IDF); "
                   "the classifier learns which words point to each sentiment.")

    with st.expander("📊 Data"):
        st.markdown(
            """
- **Source:** Amazon consumer reviews
- **Labels from star rating:** 1–2 ★ negative · 3 ★ neutral · 4–5 ★ positive
- **Test set:** 6,925 reviews (held out, never seen in training)
- **Class balance (test):** 93% positive · 4% neutral · 2% negative
"""
        )

    with st.expander("✍️ Category report"):
        st.markdown(
            f"""
- **Model:** NVIDIA {aw.MODEL.split('/')[-1]} via NVIDIA API
- **Grounding:** the report is written only from our review statistics
  (ratings, % negative, % recommend, common complaints)
- **Checks:** sections present, products named, no invented numbers
"""
        )

    with st.expander("⚠️ Limitations"):
        st.markdown(
            """
- Strong on positive reviews, weaker on **neutral** (often confused with positive)
- Data is ~93% positive, so accuracy looks higher than real-world performance
- Bag-of-words model: misses sarcasm and context ("not bad at all")
- English reviews only
- Generated reports can still contain mistakes: always review before sharing
"""
        )

    with st.expander("🛠 Tech stack"):
        st.markdown("Python · scikit-learn · pandas · Plotly · Streamlit · NVIDIA NIM API")

vc.header("Read the vibe of every review. Turn customer feedback into product decisions.")
tab1, tab2, tab3, tab4 = st.tabs(["Check a review", "Analyze your reviews", "Category report", "Model performance"])

# ---------------- Tab 1: single review ----------------
with tab1:
    if "review" not in st.session_state:
        st.session_state.review = EXAMPLES["Negative example"]

    vc.eyebrow("Try an example")
    cols = st.columns(len(EXAMPLES))
    for col, (name, text) in zip(cols, EXAMPLES.items()):
        if col.button(name, width="stretch"):
            st.session_state.review = text

    review = st.text_area("Or paste a customer review", key="review", height=120)

    if st.button("Check the vibe", type="primary") and review.strip():
        probs = model.predict_proba([review])[0]
        pred = str(model.classes_[probs.argmax()])
        conf = probs.max()

        left, right = st.columns([1, 1.15], gap="large")
        with left:
            vc.result_card(pred, {str(c): float(p) for c, p in zip(model.classes_, probs)})
            if conf < 0.5:
                st.warning("Mixed signals: the model is unsure about this one. Worth a human read.")
        with right:
            vc.eyebrow("What drove this result")
            contrib = word_contributions(review, pred)
            if contrib.empty:
                st.info("None of these words appear in our training data, so there is nothing to explain.")
            else:
                st.plotly_chart(contributions_chart(contrib, pred), width="stretch")
                st.caption("The words that pushed the result the most.")

# ---------------- Tab 2: batch ----------------
with tab2:
    vc.eyebrow("Your customer reviews")
    st.write("Upload a CSV export of your reviews, one review per row. VibeCheck labels every review "
             "and shows which products customers love and which need attention.")
    file = st.file_uploader("CSV file", type="csv")
    if file:
        data = pd.read_csv(file)
        c1, c2 = st.columns(2)
        text_col = c1.selectbox("Review text column", data.columns)
        product_col = c2.selectbox("Product column (optional)",
                                   ["(none - all reviews are one product)"] + list(data.columns))

        if st.button("Analyze all reviews", type="primary"):
            data["_text"] = data[text_col].fillna("").astype(str)
            probs = model.predict_proba(data["_text"])
            data["sentiment"] = model.classes_[probs.argmax(axis=1)]
            data["confidence"] = probs.max(axis=1).round(2)

            # --- Overall summary ---
            counts = data["sentiment"].value_counts()
            m = st.columns(4)
            m[0].metric("Reviews", f"{len(data):,}")
            for col, label in zip(m[1:], ORDER):
                n = int(counts.get(label, 0))
                col.metric(label.capitalize(), f"{n:,}", f"{n / len(data):.0%}", delta_color="off")

            left, right = st.columns([1, 1.4], gap="large")
            left.plotly_chart(donut(counts.index, counts.values, f"<b>{len(data):,}</b><br>reviews"),
                              width="stretch")
            right.dataframe(data[[text_col, "sentiment", "confidence"]], height=360, width="stretch")

            # --- Recommendation per product ---
            st.subheader("Product health")
            if product_col.startswith("(none"):
                groups = [("All uploaded reviews", data)]
            else:
                top_products = data[product_col].value_counts().head(20).index   # 20 most-reviewed
                groups = [(p, data[data[product_col] == p]) for p in top_products]

            verdicts = []
            for name, group in groups:
                kind, text = product_verdict(group)
                verdicts.append({"product": name, "verdict": kind, "reviews": len(group)})
                label, color = VERDICT_STYLE[kind]
                with st.container(border=True):
                    vc.pill(label, color)
                    st.markdown(f"#### {name}")
                    st.markdown(text)

            st.download_button("Download labelled reviews (CSV)",
                               data.drop(columns="_text").to_csv(index=False).encode("utf-8"),
                               "sentiment_results.csv", "text/csv")

# ---------------- Tab 4: performance ----------------
with tab4:
    vc.eyebrow("Held-out test set · 6,925 reviews")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", "92.4%")
    m2.metric("Macro F1", "0.64")
    m3.metric("Weighted F1", "0.93")
    m4.metric("Test reviews", "6,925")

    st.info("The test set is ~93% positive, so accuracy looks high. "
            "Macro F1 treats all three classes equally and is the fairer score. "
            "Neutral is the hardest class: it is often confused with positive.")

    cm = np.array([[99, 29, 34],
                   [40, 131, 129],
                   [41, 255, 6167]])
    cm_pct = cm / cm.sum(axis=1, keepdims=True)          # % of each true class
    text = [[f"{p:.0%}<br>({n:,})" for p, n in zip(pr, nr)] for pr, nr in zip(cm_pct, cm)]

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.subheader("Confusion matrix")
        fig = go.Figure(go.Heatmap(
            z=cm_pct, x=[f"Pred {l}" for l in ORDER], y=[f"True {l}" for l in ORDER],
            colorscale=[[0, vc.COLORS["surface"]], [0.5, "#1F5A4D"], [1, "#2C8A73"]], zmin=0, zmax=1,
            text=text, texttemplate="%{text}", textfont=dict(family=vc.MONO, color=vc.COLORS["text"]),
            showscale=False, xgap=3, ygap=3,
            hovertemplate="%{y} → %{x}: %{z:.0%}<extra></extra>",
        ))
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(vc.style_chart(fig, height=380), width="stretch")
        st.caption("Each row shows where the reviews of that true class ended up (row = 100%).")

    with right:
        st.subheader("Per-class scores")
        scores = pd.DataFrame({
            "Class": ["Negative", "Neutral", "Positive"],
            "Precision": [0.55, 0.32, 0.97],
            "Recall": [0.61, 0.44, 0.95],
            "F1": [0.58, 0.37, 0.96],
            "Support": [162, 300, 6463],
        })
        st.dataframe(scores, hide_index=True, width="stretch")

# ---------------- Tab 3: article writer ----------------
@st.cache_data(show_spinner=False)
def generate_article(category, model):          # model in the key -> new cache if you switch model
    return aw.write_article(aw.prompt_v2(category))


with tab3:
    st.write("Get a short report on the strongest and weakest products in a category: what customers praise, "
             "what they complain about and which product needs attention first. "
             f"Written by NVIDIA **{aw.MODEL.split('/')[-1]}** from our review statistics.")

    category = st.selectbox("Category", aw.categories())

    with st.expander("The data behind this report"):
        st.dataframe(aw.product_table(aw.df[aw.df["category"] == category]), use_container_width=True)

    if st.button("Write report", type="primary"):
        try:
            with st.spinner("Writing the report… about 20–30 seconds"):
                article = generate_article(category, aw.MODEL)
        except Exception as e:
            st.error("The report writer did not answer. Try again in a minute.")
            with st.expander("Technical details"):
                st.exception(e)          # full traceback with file names and line numbers
            st.stop()

        check = aw.check_article(article, category)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Words", check["words"])
        c2.metric("Sections", check["sections"])
        c3.metric("Products covered", check["product names"])
        c4.metric("Numbers not in data", len(check["invented numbers"]))
        if check["invented numbers"]:
            st.warning(f"Numbers not found in the data: {', '.join(check['invented numbers'])}")

        with st.container(border=True):
            st.markdown(article)

        st.download_button("Download report (.md)", article.encode("utf-8"),
                           f"article_{category.lower().replace(' ', '_')}.md", "text/markdown")

vc.footer()
