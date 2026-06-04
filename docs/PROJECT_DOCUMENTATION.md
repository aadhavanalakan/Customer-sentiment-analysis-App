# Customer Sentiment Analysis — Project Documentation

**Author:** Aadhavan Alakan
**Course:** Agentic AI Cohort — Week 1 Homework
**Live app:** https://aadhavanalakan-customer-sentiment-analysis-app-app-fyfpik.streamlit.app
**Source code:** https://github.com/aadhavanalakan/Customer-sentiment-analysis-App
**Data source:** [Google Sheet — Founding Farmers reviews](https://docs.google.com/spreadsheets/d/1Jpe5kltkMCp56wR7R_ZbSNdfIv95pwkGXyrQf1CadZY/edit?usp=sharing)

---

## 1. Project Overview

**Customer Sentiment Analysis** is an interactive web dashboard that turns a raw CSV of
customer reviews into business-ready insight — sentiment, emotion, keyword/phrase analytics,
and drill-down exploration — without sending any data to an external API.

**What it does**

- Upload a CSV of reviews → the app auto-detects the text column and scores every review.
- Classifies each review as **Positive / Neutral / Negative** and measures **polarity**
  (−1 to +1) and **subjectivity** (0 to 1).
- Visualizes results across seven tabs: **Overview, Sentiment, Words, Emotions, Reviews,
  Add Review, Architecture.**
- Lets the user **click any chart bar or emotion** to open a pop-up of the exact matching
  reviews and **download that subset as CSV.**
- Supports adding new reviews live and exporting the full database.

**Tech stack:** Python · Streamlit · TextBlob (sentiment) · pandas/NumPy · Plotly ·
WordCloud/matplotlib · streamlit-option-menu. Data collected with **requests + BeautifulSoup**.
Environment managed with **uv**; deployed on **Streamlit Community Cloud**; version-controlled
on **GitHub** (MIT licensed).

**Architecture (intentionally layered / "MLOps-friendly"):** the single `app.py` is organized
into 7 swappable layers — Config, Styling, Preprocessing, **Sentiment Model (behind an abstract
interface)**, Analytics Engine, Data Store, and Serving. The model sits behind a `SentimentModel`
abstract class, so TextBlob can be swapped for VADER or a transformer without touching the
analytics or UI.

---

## 2. Datasets Used

**Primary dataset — Founding Farmers (Washington, DC) customer reviews**

I web-scraped real customer reviews of **Founding Farmers**, a popular restaurant in Washington,
DC, from its **OpenTable** page. The result is a single text column of reviews stored as a CSV
(`Webscraped_Data.csv`, ~6.2 MB).

- **Data source (Google Sheet):** https://docs.google.com/spreadsheets/d/1Jpe5kltkMCp56wR7R_ZbSNdfIv95pwkGXyrQf1CadZY/edit?usp=sharing
- **Volume:** ~23,458 raw reviews → after automatic de-duplication, **23,059 reviews analyzed
  (399 duplicates removed).**
- **Schema:** one text column (`Reviews`); the app auto-detects it as the longest-average-text column.

### How the data was collected (web scraping)

The reviews were scraped from OpenTable using **`requests`** (to fetch the HTML),
**`BeautifulSoup`** (to parse it), and **`pandas`** (to store it). Reviews live in `<p>` tags
inside a `div` with class `oc-reviews-ee80f19c`, so the scraper loops through paginated review
pages and extracts each paragraph.

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Confirm a single page is reachable (200 = success)
r = requests.get('https://www.opentable.com/r/founding-farmers-dc-washington')
print(r.status_code)
soup = BeautifulSoup(r.text, 'html.parser')

# A browser-like User-Agent avoids being blocked
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                         'AppleWebKit/537.36 (KHTML, like Gecko) '
                         'Chrome/100.0.4896.75 Safari/537.36'}

questionlist = []
def getQuestions(page):
    url = f'https://www.opentable.com/r/founding-farmers-dc-washington?page={page}'
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, 'html.parser')
    reviews = soup.find_all('div', {'class': 'oc-reviews-ee80f19c'})
    for item in reviews:
        questionlist.append(item.find('p').text)

# Loop through review pages 1–250
for x in range(1, 251):
    getQuestions(x)

df = pd.DataFrame(questionlist)
print('Finished')
```

The resulting DataFrame was exported to CSV and is what the dashboard consumes.

**Data handling inside the app:**

- Reviews are normalized (lowercased, punctuation stripped) and **exact duplicates removed
  before any analysis.**
- The database is persisted to `data/reviews.csv`; the live file is **git-ignored** so user
  data never ships with the repo.

**Secondary (dev-only) dataset:** a hard-coded list of 35 synthetic reviews used to prototype
the dashboard before the real data was wired in (later removed from the UI).

---

## 3. Prompts Used During Vibe Coding

The project was built entirely through conversational "vibe coding" with Claude (Claude Code).
Representative prompts, in order:

1. **Analysis first:** "I built this code on Claude's website. Analyze it and give me the
   architecture of what it's doing. Don't build anything until I say so."
2. **Build:** "Use Streamlit and a uv project."
3. **Sidebar + branding overhaul:** "Remove the text-column selector; rename the uploader to
   'Upload reviews'; fix the invisible white uploader background; use a red-and-white theme with
   a nice red sidebar; remove 'Load sample data'; add a 'Download database' button; remove the
   model caption; rename the title to 'Sentiment Analysis' with a professional logo; rename the
   landing page and drop the taglines; make the donut fancier and add help text to the gauges."
4. **Accuracy + cleanup pass:** "The uploader background is still white — fix it. Remove the
   confusing 'r = −0.32' text. 'Founding Farmers' is the restaurant name — remove it from the
   phrases. Why is the word 'chicken' red in the word cloud? Remove the aspect chart and improve
   the emotion radar with better help text. Why is 'chicken' showing as a negative complaint when
   the review is positive?"
5. **Interactivity:** "Make the chart bars and emotions clickable — open a pop-up of all related
   reviews with a download button. Also change the polarity thresholds."
6. **Final tuning + bug:** "Make polarity: negative < 0, neutral 0–0.4, positive > 0.4. Merge the
   Themes and Reviews pages. And these clearly positive 'fried chicken' reviews show −0.75
   polarity — please investigate."
7. **Ship it:** "Structure this nicely and publish it on GitHub" → "Deploy it on Streamlit" →
   "Rename and move the folder into Week 1 Homework."

**Key observation:** most prompts were **screenshot + plain-English feedback** ("why is this
white?", "why is chicken red?") rather than technical specs — the agent diagnosed the root cause
each time and proposed the fix.

---

## 4. Iterations I Tried

The most interesting part of the project was the *debugging loop* — several features took
multiple attempts:

**a) Sentiment thresholds** — `±0.1` → `±0.3` → final logic (**negative < 0, neutral 0–0.4,
positive > 0.4**). Tuned by looking at the real distribution of the Founding Farmers data.

**b) Word-cloud coloring (3 attempts)**

- v1: color by TextBlob's standalone word score → **bug:** "chicken" turned red (TextBlob reads
  "chicken" as "cowardly," −0.6).
- v2: color by *contextual* polarity (avg sentiment of reviews containing the word) → **bug:**
  because the corpus is ~74% positive, *everything* turned green/grey.
- v3 (final): a **hybrid** — trust the curated adjective lexicon for real sentiment words (so
  "disappointed" = red), and require standalone + contextual agreement for everything else (so
  "chicken" = grey).

**c) The "fried chicken = −0.75" bug** — Diagnosed via TextBlob's per-word assessments: a praise
sentence ("…must try") scored negative *only* because of the word "chicken." Fix: a
**noun-correction layer** that recomputes polarity from TextBlob's assessments but **drops
bare-noun misfires**, keeping real sentiment words. Misclassified praise flipped back to Positive
while genuine complaints stayed Negative.

**d) Brand-name pollution** — "Founding Farmers" dominated keywords/phrases. Added an automatic
**proper-noun detector** (words capitalized mid-sentence across the corpus) that filters
brand/place names out of all keyword and phrase analytics.

**e) Themes feature → removed** — The original "Top 5 Themes / Complaints" put "chicken" under
*complaints*. I first fixed it by routing aspects by their true corpus-wide sentiment leaning;
ultimately we **merged Themes into the Reviews page** (top positive/neutral/negative reviews +
praise/complaint phrase bars), reducing 8 tabs to 7.

**f) The stubborn white uploader** — Two CSS attempts failed silently; investigation revealed
**Streamlit 1.58 gives the uploaded-file chip no test-id**, so the selectors never matched. Final
fix overrode the whole uploader subtree and added an explicit "✓ file ready" banner.

**g) Interactivity** — Added `st.dialog` pop-ups triggered by Plotly bar clicks (`on_select`) and
per-emotion buttons, each with CSV export — plus a per-chart guard to prevent the dialog from
re-opening in a loop.

---

## 5. Learnings & Observations from the Workflow

- **Web scraping is brittle but powerful.** The whole project depended on a CSS-class selector
  (`oc-reviews-ee80f19c`) and a browser-like User-Agent header; OpenTable could change either at
  any time. Pagination + a simple append-to-list pattern was enough to gather ~23K real reviews.
- **Lexicon models have domain blind spots.** TextBlob scoring "chicken" as −0.6 broke a
  *restaurant*-review app in a non-obvious way. Lesson: validate a generic NLP model against
  *your* domain and add a thin correction layer rather than trusting scores blindly.
- **Good architecture pays off.** Because the sentiment model sat behind an abstract interface,
  adding the noun-correction was a localized change — no analytics or UI code had to move.
- **Debugging beats describing.** The fastest progress came from *diagnosing root causes*
  (printing TextBlob's per-word assessments, grepping the installed Streamlit package for real
  DOM test-ids) rather than guessing at fixes.
- **Test what actually renders.** A subtle gotcha: in Streamlit's test harness the custom nav menu
  always returns the default tab, so an early "all tabs pass" check was misleading until each tab
  was forced.
- **Vibe coding is iterative and visual.** Screenshots + plain-language feedback drove almost every
  improvement.
- **Shipping has its own gotchas.** GitHub device-flow auth, Streamlit Cloud's **ephemeral storage**
  (added reviews don't persist across reboots), and keeping the live data file out of version
  control were all real-world deployment lessons.

**Future work:** swap TextBlob for VADER or a fine-tuned transformer via the existing interface;
move persistence from CSV to SQLite; make the scraper resilient to layout changes; cache analysis
per-dataset for faster cold starts.
