"""
============================================================================
 CUSTOMER REVIEWS INTELLIGENCE  ·  Streamlit + MLOps single-file deployment
============================================================================
 Run:   streamlit run app.py
 Deps:  see requirements.txt (streamlit, pandas, numpy, textblob, plotly,
        wordcloud, matplotlib, streamlit-option-menu)

 ARCHITECTURE LAYERS (split into src/ modules for a full repo — see README):
   1. CONFIG            constants, lexicons, thresholds, paths
   2. STYLING           presentation layer (injected CSS + Plotly theme)
   3. PREPROCESSING     cleaning, tokenization, n-grams
   4. SENTIMENT MODEL   swappable inference layer (TextBlob implementation)
   5. ANALYTICS ENGINE  features: aspects, emotions, DISTINCT themes
   6. DATA STORE        CSV-backed persistence + duplicate removal
   7. SERVING           Streamlit UI / tabs
============================================================================
"""

import os
import re
from collections import Counter

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from textblob import TextBlob
from wordcloud import WordCloud
from streamlit_option_menu import option_menu
from xquik_import import detect_review_text_column, prepare_reviews, read_review_csv

# ============================================================================
# 1. CONFIG
# ============================================================================
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "reviews.csv")
POS_T, NEG_T = 0.4, 0.0           # polarity classification thresholds
                                  # > +0.4 positive · < 0 negative · 0..0.4 = neutral

COL = {"pos": "#10b981", "neu": "#94a3b8", "neg": "#f43f5e",
       "indigo": "#6366f1", "purple": "#a855f7", "pink": "#ec4899",
       "red": "#dc2626", "rose": "#e11d48", "darkred": "#991b1b", "wine": "#7f1d1d"}

STOPWORDS = set("the a an and or but if then of to in on for with at by from up about into over after under again further here there all any both each few more most other some such only own same than very can will just should now i me my we our you your he she it its they them their this that these those am is are was were been being have has had having do does did doing would could ought as also because while during before between out off down too s t m re ve ll d one get got really even much many well way thing things lot bit when what who which how why where would could should i'm you're it's don't didn't was were".split())
VERBS = set("is are was were be been being have has had do does did get got make made go went come came use used buy bought order ordered try tried take took give gave see saw say said know knew think thought want wanted need needed work worked feel felt look looked find found tell told ask asked seem seemed put receive received send sent return returned wait waited call called pay paid keep kept let leave left bring brought arrived arrive stopped working".split())
ADJECTIVES = set("good great excellent amazing awesome fantastic wonderful perfect bad terrible awful horrible poor nice beautiful happy disappointed disappointing slow fast quick easy hard difficult expensive cheap friendly rude helpful unhelpful reliable comfortable clean dirty fresh stale broken best worst better worse new old big small high low long short hot cold warm fine okay decent mediocre lovely pleasant unpleasant impressive premium flimsy sturdy durable smooth rough loud quiet bright dark soft firm tasty bland delicious gorgeous stunning outstanding superb exceptional brilliant incredible frustrating annoying confusing reasonable affordable overpriced faulty defective damaged useless solid neat reliable five star stars".split())
NON_ASPECT = STOPWORDS | VERBS | ADJECTIVES

EMOTIONS = {
    "Joy": "happy joy delighted thrilled love loved enjoy enjoyed pleased glad excited wonderful fantastic cheerful satisfied smile blissful ecstatic grateful fun awesome amazing".split(),
    "Trust": "reliable trust trusted dependable satisfied confident professional honest recommend consistent secure quality loyal assured".split(),
    "Anger": "angry anger furious mad rage hate hated outraged irate annoyed frustrated frustrating livid fuming infuriating".split(),
    "Sadness": "sad disappointed disappointing unhappy upset regret sorry miserable heartbroken letdown dissatisfied depressing gloomy".split(),
    "Fear": "worried worry afraid scared anxious nervous concerned fear uneasy doubtful hesitant risky unsafe".split(),
    "Surprise": "surprised surprising shocked shocking unexpected astonished amazed wow stunned unbelievable".split(),
    "Disgust": "disgusting gross nasty revolting repulsive sick foul vile filthy dirty".split(),
}
EMO_LOOKUP = {}
for _e, _ws in EMOTIONS.items():
    for _w in _ws:
        EMO_LOOKUP.setdefault(_w, []).append(_e)

SAMPLE = [
    "Absolutely love this product! The quality is excellent and delivery was super fast. Highly recommend to everyone.",
    "Terrible experience. The item arrived broken and customer service was rude and unhelpful. Avoid at all costs.",
    "Decent value for the price. Nothing amazing but it works fine and does the job.",
    "The staff were incredibly friendly and helpful. Best customer service I have experienced in years.",
    "Very disappointed with the quality. It looks cheap and stopped working after a week. Waste of money.",
    "Fast shipping and great packaging. The product exceeded my expectations. Will buy again!",
    "Customer support was slow to respond and the issue was never really resolved. Frustrating.",
    "Beautiful design and very comfortable to use. The battery life is amazing and it feels premium.",
    "Overpriced for what you get. The material feels flimsy and the delivery was delayed by two weeks.",
    "Excellent quality and the price was very reasonable. Shipping was prompt and everything arrived perfectly.",
    "The app is buggy and crashes constantly. Confusing interface and very difficult to navigate.",
    "Wonderful service from start to finish. The team was professional and the product is fantastic.",
    "Mediocre at best. The food was bland and the service was slow. Would not return.",
    "Super reliable and durable. I have used it daily for months and it still works perfectly.",
    "Horrible customer service. They were rude and refused to give me a refund. Complete scam.",
    "The room was clean and comfortable but the staff were a bit unfriendly. Mixed feelings overall.",
    "Amazing product, fast delivery, friendly support. Everything was perfect. Five stars!",
    "Stopped working after two days. Defective and the company would not respond to my emails.",
    "Pretty good overall. The quality is decent and the price is affordable. Happy with my purchase.",
    "Awful experience. Long wait times, expensive prices, and unhelpful staff. Very frustrating.",
    "The product quality is outstanding and the customer service was exceptional. Highly recommended!",
    "Disappointing. The item was damaged and the delivery was extremely late. Poor experience.",
    "Great value and reliable performance. The setup was easy and the support team was helpful.",
    "The service was incredibly slow and the food arrived cold. Overpriced and not worth it.",
    "Love the design and the quality is premium. Fast shipping and excellent customer support.",
    "Faulty product, broke immediately. Customer service was rude. Worst purchase ever.",
    "Nice and comfortable, good quality for the price. Delivery was a little slow but acceptable.",
    "Fantastic experience! The staff were friendly, the product is amazing, and delivery was fast.",
    "Terrible quality and the price is way too high. Felt like a complete ripoff. Avoid.",
    "Reliable and efficient. The product works great and the support team was very responsive.",
    "The quality is good but the customer service is slow. Mixed experience but mostly positive.",
    "Excellent product! Beautiful, durable, and reasonably priced. Could not be happier with it.",
    "Broken on arrival and the refund process was a nightmare. Frustrating and disappointing.",
    "Smooth and intuitive to use. Great design and the price is very affordable. Recommend it.",
    "Poor quality, slow shipping, and rude staff. A disappointing and frustrating experience overall.",
]

# ============================================================================
# 2. STYLING  (presentation layer)
# ============================================================================
st.set_page_config(page_title="Reviews Intelligence", page_icon="✦",
                   layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
html, body, [class*="css"], .stApp, button, input, textarea { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.stApp {
  background:
    radial-gradient(1100px 550px at 8% -12%, #fef2f2 0%, rgba(254,242,242,0) 55%),
    radial-gradient(1000px 520px at 112% -4%, #fff1f2 0%, rgba(255,241,242,0) 50%),
    #fafafa;
}
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1180px; }
.hero-title {
  font-size: 3.1rem; font-weight: 900; letter-spacing: -0.03em; line-height: 1.05;
  background: linear-gradient(95deg, #b91c1c, #dc2626 45%, #f43f5e);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-sub { color: #64748b; font-size: 1.05rem; margin-top: .35rem; font-weight: 500; }
.pill {
  display: inline-block; padding: 5px 13px; border-radius: 999px; font-size: .78rem;
  font-weight: 700; background: rgba(220,38,38,.10); color: #b91c1c; margin-right: 8px;
  border: 1px solid rgba(220,38,38,.20);
}
.pill.gray { background: rgba(100,116,139,.10); color:#475569; border-color: rgba(100,116,139,.18); }
.metric-card {
  background: rgba(255,255,255,.85); backdrop-filter: blur(12px);
  border-radius: 22px; padding: 20px 22px; height: 100%;
  box-shadow: 0 12px 34px rgba(2,6,23,.06); border: 1px solid rgba(2,6,23,.05);
  transition: transform .2s ease, box-shadow .2s ease;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 18px 44px rgba(2,6,23,.10); }
.metric-label { font-size:.7rem; letter-spacing:.13em; text-transform:uppercase; color:#94a3b8; font-weight:800; }
.metric-value { font-size: 2.4rem; font-weight: 900; line-height: 1.1; margin-top: 6px; letter-spacing:-.02em; }
.metric-sub { font-size:.78rem; color:#94a3b8; margin-top: 4px; font-weight:500; }
.panel {
  background: rgba(255,255,255,.85); backdrop-filter: blur(12px);
  border-radius: 24px; padding: 26px 28px; margin-bottom: 6px;
  box-shadow: 0 12px 34px rgba(2,6,23,.05); border: 1px solid rgba(2,6,23,.05);
}
.panel h3 { font-weight: 800; color:#0f172a; font-size:1.15rem; margin: 0 0 4px 0; letter-spacing:-.01em; }
.panel p.cap { color:#94a3b8; font-size:.82rem; margin: 0 0 16px 0; font-weight:500; }
.theme-card {
  border-radius: 18px; padding: 16px 18px; margin-bottom: 12px;
  border: 1px solid rgba(2,6,23,.05);
}
.theme-card.pos { background: linear-gradient(180deg, #ecfdf5, #f0fdfa); border-color:#bbf7d0; }
.theme-card.neg { background: linear-gradient(180deg, #fff1f2, #fef2f2); border-color:#fecdd3; }
.theme-rank {
  display:inline-flex; align-items:center; justify-content:center; width:26px; height:26px;
  border-radius:50%; background:#fff; font-weight:900; font-size:.85rem; margin-right:10px;
  box-shadow: 0 2px 8px rgba(2,6,23,.08);
}
.theme-head { font-weight:800; font-size:1.02rem; text-transform:capitalize; color:#0f172a; }
.theme-meta { font-size:.74rem; font-weight:700; color:#64748b; }
.theme-quote { font-size:.82rem; color:#475569; font-style:italic; margin-top:8px; line-height:1.5; }
.review-card { border-radius:16px; padding:14px 16px; margin-bottom:10px; border:1px solid rgba(2,6,23,.06); }
.review-meta { display:flex; justify-content:space-between; font-size:.72rem; color:#94a3b8; font-weight:700; margin-bottom:6px; }
.review-text { color:#334155; font-size:.9rem; line-height:1.55; }
.stButton>button {
  border-radius: 14px; font-weight: 700; border: none; padding: .55rem 1.1rem;
  background: linear-gradient(95deg,#dc2626,#b91c1c); color:#fff;
  box-shadow: 0 8px 22px rgba(220,38,38,.32); transition: transform .15s ease;
}
.stButton>button:hover { transform: translateY(-2px); }

/* ---- SIDEBAR (red theme) ---- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #991b1b 0%, #7f1d1d 62%, #5f1414 100%);
}
section[data-testid="stSidebar"] * { color: #fee2e2; }

/* primary sidebar buttons: crisp white on red */
section[data-testid="stSidebar"] .stButton>button { width:100%; background:#ffffff;
  box-shadow:0 6px 16px rgba(0,0,0,.22); }
section[data-testid="stSidebar"] .stButton>button,
section[data-testid="stSidebar"] .stButton>button * { color:#b91c1c !important; }
section[data-testid="stSidebar"] .stButton>button:hover { background:#fff5f5; }

/* download button: outlined secondary */
section[data-testid="stSidebar"] .stDownloadButton>button { width:100%; font-weight:700;
  border-radius:14px; background:rgba(255,255,255,.14);
  border:1px solid rgba(255,255,255,.45); box-shadow:none; }
section[data-testid="stSidebar"] .stDownloadButton>button,
section[data-testid="stSidebar"] .stDownloadButton>button * { color:#ffffff !important; }
section[data-testid="stSidebar"] .stDownloadButton>button:hover { background:rgba(255,255,255,.24); }

/* file uploader: force the WHOLE widget dark/legible on the red sidebar.
   (Streamlit 1.58 gives the uploaded-file chip no test-id, so we override the
   entire subtree: every inner box -> transparent, every text -> white.) */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] * { color:#ffffff !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploader"] div[class] {
  background-color: transparent !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploader"] small { color:#fee2e2 !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploader"] svg { fill:#ffffff !important; color:#ffffff !important; }
/* keep the dropzone itself a visible translucent panel (wins by being more specific) */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
  background: rgba(255,255,255,.10) !important; border: 1.5px dashed rgba(255,255,255,.6) !important;
  border-radius: 14px !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
  background: rgba(255,255,255,.18) !important; border:1px solid rgba(255,255,255,.5) !important;
  border-radius:10px !important; font-weight:700 !important; }
/* "file ready" success banner inside the red sidebar */
section[data-testid="stSidebar"] [data-testid="stAlert"],
section[data-testid="stSidebar"] [role="alert"] {
  background: rgba(255,255,255,.16) !important; border-radius:12px !important; }
section[data-testid="stSidebar"] [data-testid="stAlert"] *,
section[data-testid="stSidebar"] [role="alert"] * { color:#ffffff !important; }

[data-testid="stMetricValue"] { font-weight: 900; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def style_fig(fig, h=320, legend=False):
    fig.update_layout(
        height=h, margin=dict(l=8, r=8, t=10, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#475569", size=13),
        showlegend=legend,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,.15)", zeroline=False)
    return fig


def metric_card(label, value, sub="", color="#0f172a"):
    return (f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value" style="color:{color}">{value}</div>'
            f'<div class="metric-sub">{sub}</div></div>')


# ============================================================================
# 3. PREPROCESSING
# ============================================================================
def tokenize(text):
    return re.findall(r"[a-z']+", str(text).lower())

def content_tokens(text, min_len=3, extra_stop=frozenset()):
    return [w for w in tokenize(text)
            if w not in STOPWORDS and w not in extra_stop and len(w) >= min_len]

def aspect_tokens(text, min_len=4, extra_stop=frozenset()):
    return [w for w in tokenize(text)
            if w not in NON_ASPECT and w not in extra_stop and len(w) >= min_len]

def bigrams(token_lists, extra_stop=frozenset(), sentiment_only=False):
    """Adjacent word pairs. Drops pairs touching a stopword / ignored term.
    If sentiment_only, keeps only pairs where at least one word carries sentiment
    (an adjective or a clearly polar word) — used for praise/complaint phrases."""
    c = Counter()
    for toks in token_lists:
        for a, b in zip(toks, toks[1:]):
            if (a in STOPWORDS or b in STOPWORDS or len(a) < 3 or len(b) < 3
                    or a in extra_stop or b in extra_stop):
                continue
            if sentiment_only and not (is_sentiment_word(a) or is_sentiment_word(b)):
                continue
            c[f"{a} {b}"] += 1
    return c

def normalize(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()

@st.cache_data(show_spinner=False)
def detect_proper_nouns(reviews_tuple, min_count=5, cap_ratio=0.6):
    """Heuristically find brand / place / person names (e.g. a restaurant name).
    A token is a proper noun if, ignoring sentence-initial position, it is
    capitalised in most of its occurrences across the corpus. These carry no
    sentiment signal, so we strip them from keywords, phrases, aspects & themes."""
    cap, tot = Counter(), Counter()
    for r in reviews_tuple:
        for sent in re.split(r"[.!?]+", str(r)):
            words = re.findall(r"[A-Za-z']+", sent)
            for i, w in enumerate(words):
                lw = w.lower()
                tot[lw] += 1
                if i > 0 and w[:1].isupper():     # capitalised mid-sentence
                    cap[lw] += 1
    return frozenset(
        w for w, t in tot.items()
        if t >= min_count and cap[w] / t >= cap_ratio and w not in STOPWORDS)

_SENT_CACHE = {}
def is_sentiment_word(w):
    """True if the word is an adjective in our lexicon or has clear TextBlob polarity."""
    if w in ADJECTIVES:
        return True
    if w not in _SENT_CACHE:
        _SENT_CACHE[w] = abs(TextBlob(w).sentiment.polarity) >= 0.1
    return _SENT_CACHE[w]


# ============================================================================
# 4. SENTIMENT MODEL  (swappable inference layer)
# ============================================================================
class SentimentModel:
    """Abstract interface — swap the implementation without touching the app."""
    name = "base"
    def predict(self, text: str) -> dict:
        raise NotImplementedError

class TextBlobModel(SentimentModel):
    """Lexicon/pattern model with a noun-correction layer.

    TextBlob mis-scores some domain nouns (famously 'chicken' = 'cowardly', -0.6),
    which drags genuine praise negative. We recompute polarity from TextBlob's own
    per-word assessments but DROP any assessment whose words are all bare nouns,
    keeping real sentiment words ('delicious', 'terrible', 'cold food'). Subjectivity
    is left untouched."""
    name = "TextBlob (noun-corrected)"
    def predict(self, text: str) -> dict:
        blob = TextBlob(str(text))
        subj = float(blob.sentiment.subjectivity)
        polars = []
        for words, pol, *_ in blob.sentiment_assessments.assessments:
            if all(w not in NON_ASPECT for w in words):   # every word is a bare noun -> misfire
                continue
            polars.append(pol)
        polarity = float(sum(polars) / len(polars)) if polars else 0.0
        return {"polarity": polarity, "subjectivity": subj}

@st.cache_resource
def get_model() -> SentimentModel:
    return TextBlobModel()

def classify(p):
    return "Positive" if p > POS_T else ("Negative" if p < NEG_T else "Neutral")


# ============================================================================
# 5. ANALYTICS ENGINE
# ============================================================================
@st.cache_data(show_spinner=False)
def analyze(reviews_tuple):
    model = get_model()
    ignore = detect_proper_nouns(reviews_tuple)     # brand / restaurant / people names
    df = pd.DataFrame({"text": list(reviews_tuple)})
    pred = df["text"].apply(model.predict)
    df["polarity"] = pred.apply(lambda d: d["polarity"])
    df["subjectivity"] = pred.apply(lambda d: d["subjectivity"])
    df["sentiment"] = df["polarity"].apply(classify)
    df["toks"] = df["text"].apply(tokenize)
    df["content"] = df["text"].apply(lambda t: content_tokens(t, extra_stop=ignore))
    df["aspects"] = df["text"].apply(lambda t: set(aspect_tokens(t, extra_stop=ignore)))
    df["length"] = df["toks"].apply(len)
    df.attrs["ignore"] = ignore
    return df

def word_freq(df, n=120):
    c = Counter()
    for toks in df["content"]:
        c.update(set(toks))
    return dict(c.most_common(n))

def context_polarity(df, words):
    """Average review polarity for each word, i.e. how the word is *used* in this
    corpus — far more reliable than a single word's standalone lexicon score."""
    acc = {w: [0.0, 0] for w in words}
    wset = set(words)
    for toks, pol in zip(df["content"], df["polarity"]):
        for w in set(toks) & wset:
            acc[w][0] += pol
            acc[w][1] += 1
    return {w: (s / n if n else 0.0) for w, (s, n) in acc.items()}

def emotion_breakdown(df):
    counts = {e: 0 for e in EMOTIONS}
    for toks in df["toks"]:
        for w in toks:
            for e in EMO_LOOKUP.get(w, []):
                counts[e] += 1
    total = sum(counts.values()) or 1
    return sorted([{"emotion": e, "count": c, "pct": 100 * c / total}
                   for e, c in counts.items()], key=lambda x: -x["count"])

def aspect_sentiment(df, n=8, min_share=0.01):
    c = Counter()
    for s in df["aspects"]:
        c.update(s)
    threshold = max(3, int(len(df) * min_share))
    out = []
    for term, cnt in c.most_common(40):
        if cnt < threshold:
            continue
        mask = df["aspects"].apply(lambda s: term in s)
        out.append({"aspect": term, "count": int(cnt),
                    "polarity": float(df.loc[mask, "polarity"].mean())})
        if len(out) >= n:
            break
    return sorted(out, key=lambda x: -x["polarity"])

def aspect_corpus_polarity(full_df):
    """Average polarity of every aspect across the WHOLE corpus, so we know each
    aspect's true leaning (e.g. 'chicken' is net-positive, 'wait' net-negative)."""
    acc = {}
    for aset, pol in zip(full_df["aspects"], full_df["polarity"]):
        for a in aset:
            s = acc.setdefault(a, [0.0, 0])
            s[0] += pol
            s[1] += 1
    return {a: s / n for a, (s, n) in acc.items()}

def extract_themes(sub_df, full_df, n=5, direction="pos"):
    """
    DISTINCT themes via greedy, non-overlapping clustering — but an aspect only
    qualifies as a theme if its TRUE corpus-wide leaning matches the direction.
    So a net-positive dish like 'chicken' can never surface as a complaint, and
    the representative quote always reflects the theme's sentiment.
      - pick the most frequent qualifying aspect in the remaining pool
      - claim every review containing it (removing them from the pool)
      - recompute on what's left, repeat
    """
    if sub_df.empty:
        return []
    lean = aspect_corpus_polarity(full_df)
    ok = ((lambda a: lean.get(a, 0) > POS_T) if direction == "pos"
          else (lambda a: lean.get(a, 0) < NEG_T))
    pool = sub_df.copy()
    used = set()
    themes = []
    base = len(sub_df)
    for _ in range(n):
        remaining = pool[~pool.index.isin(used)]
        if remaining.empty:
            break
        freq = Counter()
        for s in remaining["aspects"]:
            freq.update(a for a in s if ok(a))
        if not freq:
            break
        keyword, cnt = freq.most_common(1)[0]
        if cnt < 2:
            break
        members = remaining[remaining["aspects"].apply(lambda s: keyword in s)]
        idx = (members["polarity"].idxmax() if direction == "pos"
               else members["polarity"].idxmin())
        themes.append({
            "label": keyword,
            "count": int(len(members)),
            "pct": 100 * len(members) / base,
            "avg": float(members["polarity"].mean()),
            "rep": members.loc[idx, "text"],
            "idx": list(members.index),
        })
        used.update(members.index)
    return themes


# ============================================================================
# 6. DATA STORE  (persistence + duplicate removal)
# ============================================================================
def dedup(reviews):
    """Remove exact duplicates after normalization. Returns (clean, removed)."""
    seen, clean = set(), []
    for r in reviews:
        r = str(r).strip()
        if not r:
            continue
        k = normalize(r)
        if k in seen:
            continue
        seen.add(k)
        clean.append(r)
    return clean, len(reviews) - len(clean)

class DataStore:
    """CSV-backed review store. In production, swap for SQLite / a managed DB."""
    def __init__(self, path=DB_PATH):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def load(self):
        if os.path.exists(self.path):
            try:
                return pd.read_csv(self.path)["review"].dropna().astype(str).tolist()
            except Exception:
                return []
        return []

    def save(self, reviews):
        pd.DataFrame({"review": reviews}).to_csv(self.path, index=False)

STORE = DataStore()


def set_reviews(reviews, persist=True):
    clean, removed = dedup(reviews)
    st.session_state.reviews = clean
    st.session_state.removed = removed
    if persist:
        STORE.save(clean)

def is_duplicate(text):
    return normalize(text) in {normalize(r) for r in st.session_state.get("reviews", [])}


# session bootstrap
if "reviews" not in st.session_state:
    persisted = STORE.load()
    st.session_state.reviews = persisted
    st.session_state.removed = 0
    st.session_state.source = "saved database" if persisted else None


# ============================================================================
# 7. SERVING  (Streamlit UI)
# ============================================================================
def detect_column(df):
    return detect_review_text_column(df)

LOGO_HTML = """
<div style="display:flex;align-items:center;gap:11px;margin:2px 0 6px 0;">
  <div style="width:42px;height:42px;border-radius:12px;background:#ffffff;
       display:flex;align-items:center;justify-content:center;
       box-shadow:0 8px 20px rgba(0,0,0,.30);">
    <svg width="23" height="23" viewBox="0 0 24 24" fill="none">
      <rect x="3"  y="13" width="4" height="8"  rx="1.5" fill="#dc2626"/>
      <rect x="10" y="8"  width="4" height="13" rx="1.5" fill="#ef4444"/>
      <rect x="17" y="3"  width="4" height="18" rx="1.5" fill="#b91c1c"/>
    </svg>
  </div>
  <div style="font-size:1.22rem;font-weight:800;color:#ffffff;letter-spacing:-.01em;line-height:1.05;">
     Sentiment&nbsp;Analysis</div>
</div>
"""

# ---- SIDEBAR -------------------------------------------------------------
with st.sidebar:
    st.markdown(LOGO_HTML, unsafe_allow_html=True)
    st.markdown("---")
    up = st.file_uploader("Upload reviews or Xquik export", type=["csv"])
    if up is not None:
        st.success(f"✓ {up.name} ready — click **Load dataset**")
        try:
            raw = read_review_csv(up)
        except ValueError as error:
            st.warning(str(error))
        else:
            chosen = detect_column(raw)      # prefer known review/comment headers
            if chosen is None:
                st.warning("No review, comment, text, or Tweet Text column found.")
            if st.button("Load dataset"):
                try:
                    set_reviews(prepare_reviews(raw))
                    st.session_state.source = up.name
                    st.rerun()
                except ValueError as error:
                    st.warning(str(error))
    if st.session_state.reviews:
        if st.button("Clear database"):
            set_reviews([])
            st.session_state.source = None
            st.rerun()
        st.download_button(
            "⬇  Download database",
            data=pd.DataFrame({"review": st.session_state.reviews}).to_csv(index=False),
            file_name="sentiment_database.csv",
            mime="text/csv",
        )
    st.markdown("---")
    st.caption(f"**Reviews:** {len(st.session_state.reviews):,}")
    if st.session_state.get("removed"):
        st.caption(f"**Duplicates removed:** {st.session_state.removed}")

# ---- EMPTY STATE ---------------------------------------------------------
if not st.session_state.reviews:
    st.markdown('<div class="hero-title">Customer Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("⬅ Use the sidebar to **upload a reviews CSV**, then click **Load dataset** to begin.")
    st.stop()

# ---- ANALYSIS ------------------------------------------------------------
df = analyze(tuple(st.session_state.reviews))
ignore = df.attrs.get("ignore", frozenset())     # proper nouns to drop from phrases
total = len(df)
counts = df["sentiment"].value_counts().to_dict()
for k in ["Positive", "Neutral", "Negative"]:
    counts.setdefault(k, 0)
pct = {k: 100 * counts[k] / total for k in counts}
avg_pol, avg_subj = df["polarity"].mean(), df["subjectivity"].mean()
net = pct["Positive"] - pct["Negative"]

# ---- HERO ----------------------------------------------------------------
st.markdown('<div class="hero-title">Customer Sentiment Analysis</div>', unsafe_allow_html=True)
pills = f'<span class="pill">{total:,} reviews analyzed</span>'
if st.session_state.get("removed"):
    pills += f'<span class="pill gray">{st.session_state.removed} duplicates removed</span>'
st.markdown(f'<div style="margin:16px 0 26px 0">{pills}</div>', unsafe_allow_html=True)

# ---- NAV -----------------------------------------------------------------
nav = option_menu(
    None,
    ["Overview", "Sentiment", "Words", "Emotions", "Reviews", "Add Review", "Architecture"],
    icons=["grid-1x2-fill", "graph-up-arrow", "hash", "emoji-smile-fill",
           "chat-square-quote-fill", "plus-circle-fill", "stack"],
    orientation="horizontal", default_index=0,
    styles={
        "container": {"padding": "6px", "background-color": "rgba(255,255,255,.7)",
                      "border-radius": "16px", "box-shadow": "0 8px 24px rgba(2,6,23,.05)"},
        "icon": {"font-size": "14px"},
        "nav-link": {"font-size": "13px", "font-weight": "700", "color": "#64748b",
                     "border-radius": "12px", "margin": "0 2px", "--hover-color": "#fef2f2"},
        "nav-link-selected": {"background": "linear-gradient(95deg,#dc2626,#b91c1c)",
                              "color": "white", "font-weight": "700"},
    },
)
st.markdown("<br>", unsafe_allow_html=True)


def panel_open(title, cap=""):
    st.markdown(f'<div class="panel"><h3>{title}</h3>'
                + (f'<p class="cap">{cap}</p>' if cap else ""), unsafe_allow_html=True)


# ---- drill-down: match reviews + popup with download ---------------------
def match_phrase(frame, phrase):
    """Reviews containing a keyword (1 word) or an exact bigram (2 words)."""
    parts = phrase.split()
    if len(parts) == 1:
        w = parts[0]
        return frame[frame["toks"].apply(lambda t: w in t)]
    a, b = parts[0], parts[1]
    return frame[frame["toks"].apply(
        lambda t: any(t[i] == a and t[i + 1] == b for i in range(len(t) - 1)))]

def match_emotion(frame, emotion):
    words = set(EMOTIONS[emotion])
    return frame[frame["toks"].apply(lambda t: bool(set(t) & words))]

@st.dialog("Matching reviews", width="large")
def show_reviews(heading, matched):
    st.markdown(f"#### {heading}")
    if matched is None or matched.empty:
        st.info("No matching reviews found.")
        return
    st.caption(f"{len(matched):,} review(s)")
    out = matched[["text", "sentiment", "polarity", "subjectivity"]]
    st.download_button("⬇  Download these reviews (CSV)", out.to_csv(index=False),
                       file_name="reviews_subset.csv", mime="text/csv",
                       use_container_width=True)
    st.markdown("---")
    for _, r in matched.sort_values("polarity").iterrows():
        txt = (r["text"][:400] + "…") if len(r["text"]) > 400 else r["text"]
        bg = ("#ecfdf5" if r["sentiment"] == "Positive"
              else "#fff1f2" if r["sentiment"] == "Negative" else "#f8fafc")
        st.markdown(
            f'<div class="review-card" style="background:{bg}">'
            f'<div class="review-meta"><span>{r["sentiment"]}</span>'
            f'<span>polarity {r.polarity:.2f} · subjectivity {r.subjectivity:.2f}</span></div>'
            f'<div class="review-text">{txt}</div></div>', unsafe_allow_html=True)

def clickable_bar(fig, key, frame, height=300):
    """Render a bar chart whose bars open a popup of the matching reviews."""
    ev = st.plotly_chart(style_fig(fig, height), use_container_width=True,
                         on_select="rerun", key=key)
    try:
        pts = ev["selection"]["points"]
    except (TypeError, KeyError, AttributeError):
        pts = []
    seen = f"_dlg_{key}"                      # per-chart guard avoids reopen loops
    if not pts:
        st.session_state.pop(seen, None)      # cleared selection -> allow re-view
        return
    label = pts[-1].get("y") or pts[-1].get("label") or pts[-1].get("x")
    if label and st.session_state.get(seen) != label:
        st.session_state[seen] = label
        show_reviews(f"Reviews containing “{label}”", match_phrase(frame, label))

# ===================== OVERVIEW =====================
if nav == "Overview":
    c = st.columns(4)
    cards = [
        ("Total Reviews", f"{total:,}", "after de-duplication", "#0f172a"),
        ("Net Sentiment", f"{net:+.0f}", "% positive − % negative", COL["pos"] if net >= 0 else COL["neg"]),
        ("Avg Polarity", f"{avg_pol:.2f}", "−1 to +1", COL["pos"] if avg_pol > POS_T else COL["neg"] if avg_pol < NEG_T else COL["neu"]),
        ("Avg Subjectivity", f"{avg_subj:.2f}", "0 factual · 1 opinion", "#0f172a"),
    ]
    for col_, (label, value, subtitle, color) in zip(c, cards):
        col_.markdown(
            metric_card(label, value, subtitle, color),
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1, 1])
    with left:
        panel_open("Sentiment Breakdown", "Share of reviews by predicted sentiment.")
        fig = go.Figure(go.Pie(
            labels=["Positive", "Neutral", "Negative"],
            values=[counts["Positive"], counts["Neutral"], counts["Negative"]],
            hole=.66, sort=False, direction="clockwise", rotation=-40,
            marker=dict(colors=[COL["pos"], COL["neu"], COL["neg"]],
                        line=dict(color="#ffffff", width=3)),
            pull=[0.06, 0, 0],
            textinfo="percent", textfont=dict(size=14, color="#ffffff", family="Inter"),
            insidetextorientation="horizontal",
            hovertemplate="%{label}: %{value} reviews (%{percent})<extra></extra>"))
        net_col = COL["pos"] if net >= 0 else COL["neg"]
        fig.add_annotation(
            text=(f"<b style='font-size:28px;color:{net_col}'>{net:+.0f}</b><br>"
                  f"<span style='font-size:10px;letter-spacing:.12em;color:#94a3b8'>NET SENTIMENT</span>"),
            showarrow=False, font=dict(family="Inter"))
        st.plotly_chart(style_fig(fig, 320, legend=True), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        panel_open("Opinion Strength",
                   "How positive and how opinionated the reviews are, on average.")
        mood = "Positive" if avg_pol > POS_T else "Negative" if avg_pol < NEG_T else "Neutral"
        mood_col = COL["pos"] if avg_pol > POS_T else COL["neg"] if avg_pol < NEG_T else COL["neu"]
        g = go.Figure()
        g.add_trace(go.Indicator(
            mode="gauge+number+delta", value=avg_pol,
            number={"font": {"size": 36, "color": mood_col}, "valueformat": ".2f"},
            delta={"reference": 0, "increasing": {"color": COL["pos"]},
                   "decreasing": {"color": COL["neg"]}},
            title={"text": "Polarity", "font": {"size": 13, "color": "#64748b"}},
            domain={"x": [0, .46], "y": [0, 1]},
            gauge={"axis": {"range": [-1, 1], "tickwidth": 1, "tickcolor": "#cbd5e1"},
                   "bar": {"color": "rgba(15,23,42,.82)", "thickness": .26},
                   "borderwidth": 0,
                   "steps": [{"range": [-1, NEG_T], "color": "#fecaca"},
                             {"range": [NEG_T, POS_T], "color": "#e2e8f0"},
                             {"range": [POS_T, 1], "color": "#bbf7d0"}],
                   "threshold": {"line": {"color": mood_col, "width": 4},
                                 "thickness": .9, "value": avg_pol}}))
        g.add_trace(go.Indicator(
            mode="gauge+number", value=avg_subj,
            number={"font": {"size": 36, "color": COL["red"]}, "valueformat": ".2f"},
            title={"text": "Subjectivity", "font": {"size": 13, "color": "#64748b"}},
            domain={"x": [.54, 1], "y": [0, 1]},
            gauge={"axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "#cbd5e1"},
                   "bar": {"color": "rgba(15,23,42,.82)", "thickness": .26},
                   "borderwidth": 0,
                   "steps": [{"range": [0, .5], "color": "#e2e8f0"},
                             {"range": [.5, 1], "color": "#fecdd3"}],
                   "threshold": {"line": {"color": COL["red"], "width": 4},
                                 "thickness": .9, "value": avg_subj}}))
        st.plotly_chart(style_fig(g, 300), use_container_width=True)
        st.markdown(
            f"""<div style="font-size:.8rem;color:#64748b;line-height:1.55;margin-top:-6px">
            <b style="color:{mood_col}">Overall mood: {mood}.</b><br>
            <b>Polarity</b> runs −1 (very negative) → +1 (very positive); ~0 is neutral.<br>
            <b>Subjectivity</b> runs 0 (factual / objective) → 1 (opinionated / personal).
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ===================== SENTIMENT =====================
elif nav == "Sentiment":
    bins = [("Very Neg", -1, -.5, "#e11d48"), ("Negative", -.5, NEG_T, "#fb7185"),
            ("Neutral", NEG_T, POS_T, "#94a3b8"), ("Positive", POS_T, .5, "#34d399"),
            ("Very Pos", .5, 1.01, "#059669")]
    bdata = [(lab, int(((df.polarity >= lo) & (df.polarity < hi)).sum()), c) for lab, lo, hi, c in bins]
    lcols = st.columns([1, 1])
    with lcols[0]:
        panel_open("Polarity Distribution", "Is sentiment polarized or moderate?")
        fig = go.Figure(go.Bar(x=[b[0] for b in bdata], y=[b[1] for b in bdata],
                               marker_color=[b[2] for b in bdata]))
        fig.update_traces(marker_line_width=0, width=.6)
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with lcols[1]:
        lb = [("1–10", 0, 10), ("11–25", 11, 25), ("26–50", 26, 50), ("51–100", 51, 100), ("100+", 101, 10**9)]
        ld = [(lab, int(((df.length >= lo) & (df.length <= hi)).sum())) for lab, lo, hi in lb]
        panel_open("Review Length", "How many reviews fall into each word-count range.")
        fig = go.Figure(go.Bar(x=[d[0] for d in ld], y=[d[1] for d in ld], marker_color=COL["indigo"]))
        fig.update_traces(width=.6)
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    c = st.columns(4)
    for col_, (label, value, subtitle, color) in zip(c, [
        ("Positive", f"{pct['Positive']:.1f}%", f"{counts['Positive']} reviews", COL["pos"]),
        ("Neutral", f"{pct['Neutral']:.1f}%", f"{counts['Neutral']} reviews", COL["neu"]),
        ("Negative", f"{pct['Negative']:.1f}%", f"{counts['Negative']} reviews", COL["neg"]),
        ("Avg Length", f"{df.length.mean():.0f}", "words / review", "#0f172a")]):
        col_.markdown(
            metric_card(label, value, subtitle, color),
            unsafe_allow_html=True,
        )

# ===================== WORDS =====================
elif nav == "Words":
    freq = word_freq(df)
    panel_open("Word Cloud", "Size = frequency · colour = sentiment (green positive · grey neutral · red negative)")
    if freq:
        ctx = context_polarity(df, list(freq))
        def cfunc(word, **kw):
            sp = TextBlob(word).sentiment.polarity        # standalone lexicon score
            if word in ADJECTIVES:                        # curated adjective → trust its sign
                return COL["pos"] if sp > .05 else COL["neg"] if sp < -.05 else "#64748b"
            cp = ctx.get(word, 0.0)                       # noun/other → require both to agree
            if sp > .1 and cp > .02:
                return COL["pos"]
            if sp < -.1 and cp < -.02:
                return COL["neg"]
            return "#64748b"
        wc = WordCloud(width=1200, height=460, background_color=None, mode="RGBA",
                       prefer_horizontal=.92, max_words=120, relative_scaling=.5,
                       color_func=cfunc).generate_from_frequencies(freq)
        st.image(wc.to_array(), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    cc = st.columns(2)
    with cc[0]:
        panel_open("Top Keywords", "Click a bar to read & download its reviews.")
        top = sorted(freq.items(), key=lambda x: -x[1])[:12][::-1]
        fig = go.Figure(go.Bar(x=[v for _, v in top], y=[k for k, _ in top],
                               orientation="h", marker_color=COL["indigo"]))
        clickable_bar(fig, "kw_bar", df, height=360)
        st.markdown("</div>", unsafe_allow_html=True)
    with cc[1]:
        panel_open("Top Phrases (Bigrams)", "Brand / place names filtered out · click a bar for its reviews.")
        bg = bigrams(df["toks"], extra_stop=ignore).most_common(12)[::-1]
        fig = go.Figure(go.Bar(x=[v for _, v in bg], y=[k for k, _ in bg],
                               orientation="h", marker_color=COL["purple"]))
        clickable_bar(fig, "bg_bar", df, height=360)
        st.markdown("</div>", unsafe_allow_html=True)

# ===================== EMOTIONS =====================
elif nav == "Emotions":
    panel_open(
        "Emotion Breakdown",
        "Each review's words are matched against seven core-emotion lexicons "
        "(Joy, Trust, Anger, Sadness, Fear, Surprise, Disgust). The chart shows the "
        "share of all emotion words that fall into each emotion — so a longer spoke means "
        "that feeling appears more often. It tells you <i>which</i> emotions drive the "
        "reviews, not just whether they're positive or negative.")
    emo = emotion_breakdown(df)
    rmax = max((e["pct"] for e in emo), default=1) or 1
    fig = go.Figure(go.Scatterpolar(
        r=[e["pct"] for e in emo] + [emo[0]["pct"]],
        theta=[e["emotion"] for e in emo] + [emo[0]["emotion"]],
        fill="toself", mode="lines+markers",
        line=dict(color=COL["red"], width=2.5),
        marker=dict(size=8, color=COL["red"]),
        fillcolor="rgba(220,38,38,.16)",
        hovertemplate="%{theta}: %{r:.1f}% of emotion words<extra></extra>"))
    fig.update_layout(
        height=460, margin=dict(l=80, r=80, t=30, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#475569", size=13),
        polar=dict(
            bgcolor="rgba(248,250,252,.6)",
            radialaxis=dict(visible=True, range=[0, rmax * 1.12], ticksuffix="%",
                            tickfont=dict(size=10, color="#94a3b8"),
                            gridcolor="rgba(148,163,184,.28)"),
            angularaxis=dict(rotation=90, direction="clockwise",
                             tickfont=dict(size=14, color="#334155", family="Inter"),
                             gridcolor="rgba(148,163,184,.28)")))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # clickable read-out: one button per emotion -> popup of its reviews + download
    panel_open("Explore by emotion", "Click an emotion to read and download every review expressing it.")
    cols = st.columns(len(emo))
    for col_, e in zip(cols, emo):
        if col_.button(f'{e["emotion"]}  ·  {e["pct"]:.0f}%', key=f'emo_{e["emotion"]}',
                       use_container_width=True):
            show_reviews(f'“{e["emotion"]}” reviews', match_emotion(df, e["emotion"]))
        col_.caption(f'{e["count"]} words')
    st.markdown("</div>", unsafe_allow_html=True)

# ===================== REVIEWS (top reviews + praise/complaints) =====================
elif nav == "Reviews":
    pos_df = df[df.sentiment == "Positive"]
    neu_df = df[df.sentiment == "Neutral"]
    neg_df = df[df.sentiment == "Negative"]

    def render_reviews(title, rows, bg, border):
        panel_open(title)
        if rows.empty:
            st.caption("None in this category.")
        for _, r in rows.iterrows():
            txt = (r["text"][:300] + "…") if len(r["text"]) > 300 else r["text"]
            st.markdown(
                f'<div class="review-card" style="background:{bg};border-color:{border}">'
                f'<div class="review-meta"><span>{r.sentiment} · polarity {r.polarity:.2f}</span>'
                f'<span>subjectivity {r.subjectivity:.2f}</span></div>'
                f'<div class="review-text">{txt}</div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    s = df.sort_values("polarity", ascending=False)
    rc = st.columns(3)
    with rc[0]:
        render_reviews("🟢 Top Positive Reviews", s.head(3), "#ecfdf5", "#bbf7d0")
    with rc[1]:
        # neutral = classified Neutral, most opinionated first so they read as real reviews
        render_reviews("⚪ Top Neutral Reviews",
                       neu_df.sort_values("subjectivity", ascending=False).head(3),
                       "#f8fafc", "#e2e8f0")
    with rc[2]:
        render_reviews("🔴 Top Negative Reviews", s.tail(3).iloc[::-1], "#fff1f2", "#fecdd3")

    st.markdown("<br>", unsafe_allow_html=True)
    cc2 = st.columns(2)
    with cc2[0]:
        panel_open("What People Praise (Phrases)", "Sentiment-bearing phrases · click a bar for its reviews.")
        pb = bigrams(pos_df["toks"], extra_stop=ignore, sentiment_only=True).most_common(8)[::-1]
        if pb:
            fig = go.Figure(go.Bar(x=[v for _, v in pb], y=[k for k, _ in pb],
                                   orientation="h", marker_color=COL["pos"]))
            clickable_bar(fig, "praise_bar", df, height=300)
        else:
            st.caption("Not enough positive reviews.")
        st.markdown("</div>", unsafe_allow_html=True)
    with cc2[1]:
        panel_open("What People Complain About (Phrases)", "Sentiment-bearing phrases · click a bar for its reviews.")
        nb = bigrams(neg_df["toks"], extra_stop=ignore, sentiment_only=True).most_common(8)[::-1]
        if nb:
            fig = go.Figure(go.Bar(x=[v for _, v in nb], y=[k for k, _ in nb],
                                   orientation="h", marker_color=COL["neg"]))
            clickable_bar(fig, "complaint_bar", df, height=300)
        else:
            st.caption("Not enough negative reviews.")
        st.markdown("</div>", unsafe_allow_html=True)

# ===================== ADD REVIEW =====================
elif nav == "Add Review":
    panel_open("Add a Review to the Database",
               "Type a comment, score it instantly, and append it to the analyzed dataset.")
    txt = st.text_area("Your review", height=130, placeholder="e.g. The delivery was fast and the support team was incredibly helpful…")
    if st.button("Analyze & add to database"):
        if not txt.strip():
            st.warning("Please enter a review first.")
        elif is_duplicate(txt):
            st.info("This review already exists in the database — skipped to avoid duplicates.")
        else:
            set_reviews(st.session_state.reviews + [txt.strip()])
            pred = get_model().predict(txt)
            sent = classify(pred["polarity"])
            emos = [e for w in tokenize(txt) for e in EMO_LOOKUP.get(w, [])]
            emo_str = ", ".join(sorted(set(emos))) or "—"
            color = COL["pos"] if sent == "Positive" else COL["neg"] if sent == "Negative" else COL["neu"]
            cc = st.columns(3)
            cc[0].markdown(metric_card("Sentiment", sent, "predicted", color), unsafe_allow_html=True)
            cc[1].markdown(metric_card("Polarity", f"{pred['polarity']:.2f}", "−1 to +1", color), unsafe_allow_html=True)
            cc[2].markdown(metric_card("Subjectivity", f"{pred['subjectivity']:.2f}", "0 to 1", "#0f172a"), unsafe_allow_html=True)
            st.markdown(f"<br>**Detected emotions:** {emo_str}", unsafe_allow_html=True)
            st.success(f"✓ Added. Database now holds {len(st.session_state.reviews):,} reviews. "
                       "Switch tabs to see the dashboard update.")
    st.markdown("</div>", unsafe_allow_html=True)

# ===================== ARCHITECTURE =====================
elif nav == "Architecture":
    panel_open("MLOps Architecture & Methodology")
    st.markdown(f"""
**Pipeline (data flow)**

`CSV / typed review` → **Data Store** (load + de-duplicate) → **Preprocessing**
(clean, tokenize, **proper-noun detection**) → **Sentiment Model** (noun-corrected
inference, cached) → **Analytics Engine** (keywords, phrases, emotions) → **Serving**
(this Streamlit UI).

**Layered design** — this single file is organized into 7 swappable layers
(Config, Styling, Preprocessing, Sentiment Model, Analytics Engine, Data Store, Serving).
For a production repo, split them into `src/` modules:

```
review-analytics/
├── app.py                 # serving layer (this UI)
├── requirements.txt
├── Dockerfile
├── .streamlit/config.toml
├── data/reviews.csv       # the persisted "database"
└── src/
    ├── config.py          # lexicons, thresholds, paths
    ├── preprocessing.py   # cleaning, tokenization, n-grams
    ├── sentiment.py       # SentimentModel ABC + TextBlobModel (swap for VADER / transformer)
    ├── analytics.py       # keywords, phrases, emotions, drill-down
    └── data_store.py      # load / save / dedup (swap CSV → SQLite / warehouse)
```

**Why it's MLOps-friendly:** the model sits behind an abstract `SentimentModel`
interface, so you can replace TextBlob with VADER or a fine-tuned transformer
without touching analytics or UI. Inference and feature computation are cached
(`@st.cache_resource` / `@st.cache_data`) for reproducible, fast reruns.

**How each metric is computed**

- **Sentiment** — TextBlob's PatternAnalyzer returns **polarity** (−1…+1) and
  **subjectivity** (0…1) per review, then a **noun-correction layer** recomputes
  polarity from the per-word assessments while dropping bare-noun misfires (e.g.
  TextBlob scoring *chicken* = −0.6), so genuine praise isn't dragged negative.
  Class thresholds: > {POS_T} positive, < {NEG_T} negative, else neutral.
- **Net Sentiment** = % positive − % negative.
- **Proper-noun filtering** — brand / restaurant / place names (e.g. the venue
  name) are auto-detected by mid-sentence capitalisation across the corpus and
  **excluded** from keywords and phrases, since they carry no sentiment signal.
- **Word cloud / keywords** — stopwords and short tokens removed; sized by
  frequency. Colour uses a **hybrid** rule: curated adjectives are coloured by
  their lexicon polarity (so *disappointed* is red), while other words are only
  coloured when their standalone score **and** their in-context usage agree —
  so a noun the lexicon mis-scores (e.g. *chicken* = "cowardly", −0.6) stays
  neutral grey instead of wrongly red.
- **Bigrams** — adjacent word pairs, dropping any pair touching a stopword or
  proper noun. For praise/complaints they're further restricted to
  **sentiment-bearing** phrases (at least one polar/adjective word).
- **Emotions** — seven emotion lexicons (Joy, Trust, Anger, Sadness, Fear,
  Surprise, Disgust) counted across the corpus; the radar shows each emotion's
  share of all emotion words.
- **Top reviews** — the Reviews tab surfaces the most positive, most neutral
  (highest-subjectivity neutral) and most negative reviews, alongside the
  praise/complaint phrase bars.
- **De-duplication** — reviews are normalized (lowercased, punctuation stripped)
  and exact duplicates removed **before** any analysis runs.
- **Interactive drill-down** — keyword / phrase bars and emotion buttons are
  clickable: each opens a popup of the exact matching reviews with a one-click
  CSV **download** of just that subset.

**Limitations to disclose:** a lexicon model doesn't fully capture sarcasm or
niche slang. It's fast, private and fully transparent — treat scores as
directional indicators.
""")
    st.markdown("</div>", unsafe_allow_html=True)
