import os
import re
import pandas as pd
from openai import OpenAI

DATA_PATH = os.path.join(os.path.dirname(__file__), "CSVs", "reviews_slim.csv.gz")
df = pd.read_csv(DATA_PATH)
# the source data spells "Kindle" as "kind lee" in some product names
for col in ("product_name", "short_name"):
    df[col] = df[col].str.replace(r"\bkind lee\b", "kindle", regex=True)

MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
MIN_REVIEWS = 10          # 👉 use the same value as in your notebook

def complaints(category_df, product, n=2):
    unhappy = category_df[(category_df["short_name"] == product) & (category_df["rating"] <= 3) & (category_df["n_words"] >= 10)]
    examples = unhappy["clean_text"].sample(min(n, len(unhappy)), random_state=42)
    return [text[:150] for text in examples]

def product_table(category_df):
    table = category_df.groupby("short_name").agg(
        reviews=("rating", "size"),
        avg_rating=("rating", "mean"),
        pct_negative=("negative", "mean"),
        pct_recommend=("recommend", "mean"),
    ).round(2)
    table = table[table["reviews"] >= MIN_REVIEWS]
    return table.sort_values("avg_rating", ascending=False)


def pick_products(table):
    if len(table) < 2:
        return list(table.index[:3]), None
    worst = table["avg_rating"].idxmin()
    top = table.drop(worst).sort_values("reviews", ascending=False).index[:3]
    return list(top), worst

def product_line(table, product):
    row = table.loc[product]
    line = f"{product}: {row['avg_rating']}/5 from {row['reviews']:.0f} reviews, {row['pct_negative']}% negative"
    if pd.notna(row["pct_recommend"]):
        line += f", {row['pct_recommend']}% recommend it"
    return line

def facts_text(category):
    category_df = df[df["category"] == category]
    table = product_table(category_df)
    top, worst = pick_products(table)

    lines = [f"Category: {category} ({len(category_df)} reviews)", "", "TOP PRODUCTS (most reviewed first):"]
    for product in top:
        lines.append("- " + product_line(table, product))
        for text in complaints(category_df, product):
            lines.append(f'    complaint: "{text}"')
    lines.append("")
    if worst is None:
        lines.append("WORST PRODUCT: none")
    else:
        lines.append("WORST PRODUCT: " + product_line(table, worst))
        for text in complaints(category_df, worst):
            lines.append(f'    complaint: "{text}"')
    return "\n".join(lines)


def prompt_v2(category):
    category_df = df[df["category"] == category]
    top, worst = pick_products(product_table(category_df))

    template = [f"# The best and worst {category.lower()}", "[intro of two sentences]", "", "## Top picks"]
    for product in top:
        template.append(f"**{product}**: [one or two sentences: what buyers like, and how it compares with the others]")
    template += ["", "## Common complaints", "[two or three sentences: the problems unhappy buyers mention]", ""]
    if worst is not None:
        template += [f"## Product to avoid: {worst}", "[one or two sentences: why it is the weakest]", ""]
    template += ["## Verdict", "[one sentence]"]

    return (facts_text(category) + "\n\n"
            "Fill in this article template. Replace every [...] with your own text and keep everything else as it is.\n"
            "RULES: Use only the facts above. Do not invent prices, features or numbers. "
            "Do not copy the complaints, say them in your own words. "
            "If the product to avoid has a rating of 4 or more, say it is the weakest of a good group, not a bad product.\n\n"
            + "\n".join(template))


def check_article(article, category):
    category_df = df[df["category"] == category]
    top, worst = pick_products(product_table(category_df))
    names = top + ([worst] if worst is not None else [])
    sections = ["top picks", "common complaints", "verdict"]
    numbers_in_facts = set(re.findall(r"\d+(?:\.\d+)?", facts_text(category)))
    numbers_in_article = set(re.findall(r"\d+(?:\.\d+)?", article.replace(",", "")))
    return {
        "words": len(article.split()),
        "sections": f"{sum(section in article.lower() for section in sections)}/{len(sections)}",
        "product names": f"{sum(name in article for name in names)}/{len(names)}",
        "invented numbers": sorted(numbers_in_article - numbers_in_facts),
    }


# ---------------- NVIDIA API ----------------
def _api_key():
    try:
        import streamlit as st
        return st.secrets["NVIDIA_API_KEY"]
    except Exception:
        return os.environ.get("NVIDIA_API_KEY")


client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=_api_key())


def clean(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = re.sub(r"\s*\[(?:intro|one|two|three)[^\]]*\]", "", text)   # template [...] the model left in
    text = re.sub(r"(?m)^\s*\[[^\]\n]{3,80}\]\s*$\n?", "", text)      # any other placeholder left on its own line
    text = text.strip()
    if text.lower().startswith("the best and worst"):                  # make the title a heading
        text = "# " + text
    return text


def write_article(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": "You are a writer for a product review blog."},
                  {"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.2,
        top_p=0.9,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return clean(response.choices[0].message.content)


def categories():
    return sorted(df["category"].dropna().unique())