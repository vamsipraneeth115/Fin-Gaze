import requests
import streamlit as st
import xml.etree.ElementTree as ET
import yfinance as yf

from theme import apply_shared_theme, render_page_hero

st.set_page_config(page_title="Market News Analysis", layout="wide")
apply_shared_theme()

st.markdown(
    """
<style>
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 12px;
    color: var(--text-main);
    letter-spacing: 0.01em;
}
.section-title::before {
    content: "";
    display: inline-block;
    width: 12px;
    height: 12px;
    margin-right: 10px;
    border-radius: 999px;
    background: linear-gradient(180deg, #67e8f9, #60a5fa);
    box-shadow: 0 0 14px rgba(103, 232, 249, 0.55);
}
.news-layout {
    display: grid;
    grid-template-columns: minmax(0, 1.1fr) minmax(0, 0.9fr);
    gap: 18px;
    align-items: start;
}
.news-card {
    background: linear-gradient(180deg, rgba(9, 18, 36, 0.72), rgba(4, 10, 25, 0.88));
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 18px;
    padding: 16px 18px;
    margin-bottom: 12px;
    box-shadow: 0 16px 36px rgba(2, 6, 23, 0.2);
}
.news-card a {
    color: #e2e8f0;
    text-decoration: none;
    font-weight: 700;
}
.news-card a:hover {
    color: #7dd3fc;
}
.news-meta {
    margin-top: 8px;
    color: #93c5fd;
    font-size: 0.9rem;
}
.sentiment-card {
    background: linear-gradient(180deg, rgba(7, 16, 31, 0.9), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(125, 211, 252, 0.18);
    border-radius: 20px;
    padding: 20px 22px;
    margin: 10px 0 18px 0;
    box-shadow: 0 18px 42px rgba(2, 6, 23, 0.24);
}
.stats-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
    margin-top: 16px;
}
.stat-tile {
    border-radius: 16px;
    padding: 14px;
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.76), rgba(2, 6, 23, 0.6));
    border: 1px solid rgba(148, 163, 184, 0.14);
}
.stat-label {
    font-size: 0.85rem;
    color: #93c5fd;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.stat-value {
    margin-top: 8px;
    font-size: 1.7rem;
    font-weight: 700;
    color: #f8fafc;
}
.sentiment-label {
    display: inline-block;
    margin-top: 10px;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 0.86rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.sentiment-buy {
    background: rgba(34, 197, 94, 0.16);
    color: #86efac;
}
.sentiment-hold {
    background: rgba(250, 204, 21, 0.14);
    color: #fde68a;
}
.sentiment-sell {
    background: rgba(248, 113, 113, 0.16);
    color: #fca5a5;
}
.headline-tag {
    display: inline-block;
    margin-top: 8px;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
}
.headline-positive {
    background: rgba(34, 197, 94, 0.16);
    color: #86efac;
}
.headline-neutral {
    background: rgba(148, 163, 184, 0.14);
    color: #cbd5e1;
}
.headline-negative {
    background: rgba(248, 113, 113, 0.16);
    color: #fca5a5;
}
@media (max-width: 900px) {
    .news-layout {
        grid-template-columns: 1fr;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

render_page_hero(
    "Market News Analysis",
    "Read live market headlines, measure positive versus negative pressure for a stock, and turn that into a simple Buy, Hold, or Sell suggestion.",
)

NEWS_FEED_URLS = [
    {"name": "Google News", "url": "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en"},
    {"name": "Google News Markets", "url": "https://news.google.com/rss/search?q=stock%20market&hl=en-US&gl=US&ceid=US:en"},
    {"name": "WSJ Markets", "url": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"},
]

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

POSITIVE_NEWS_TERMS = {
    "beat", "beats", "growth", "gains", "gain", "surge", "surges", "strong",
    "record", "upgrade", "upgrades", "bullish", "profit", "profits", "rally",
    "expands", "expansion", "partnership", "innovation", "outperform", "buyback",
    "rebound", "optimistic", "higher", "rise", "rises", "jump", "jumps",
}

NEGATIVE_NEWS_TERMS = {
    "miss", "misses", "drop", "drops", "fall", "falls", "warning", "warns",
    "downgrade", "downgrades", "lawsuit", "probe", "investigation", "weak",
    "decline", "declines", "cut", "cuts", "lower", "loss", "losses", "selloff",
    "plunge", "plunges", "risk", "risks", "tariff", "tariffs", "fraud",
    "recall", "bankruptcy", "bearish", "slump",
}

GENERIC_COMPANY_WORDS = {
    "inc", "corp", "corporation", "company", "co", "limited", "ltd", "group", "holdings"
}


def clean_text(value) -> str:
    return " ".join((value or "").split()).strip()


def tag_name(element) -> str:
    return element.tag.split("}", 1)[-1].lower()


def extract_feed_items(root, fallback_source: str):
    parsed_items = []
    for node in root.iter():
        if tag_name(node) not in {"item", "entry"}:
            continue

        title = ""
        link = ""
        pub_date = ""
        source = fallback_source

        for child in list(node):
            child_tag = tag_name(child)
            child_text = clean_text(child.text)
            if child_tag == "title" and not title:
                title = child_text
            elif child_tag == "link" and not link:
                link = clean_text(child.get("href") or child_text)
            elif child_tag in {"pubdate", "published", "updated"} and not pub_date:
                pub_date = child_text
            elif child_tag == "source" and child_text:
                source = child_text

        if title:
            parsed_items.append(
                {
                    "title": title,
                    "link": link,
                    "pub_date": pub_date,
                    "source": source,
                }
            )

    return parsed_items


@st.cache_data(ttl=600)
def fetch_market_news(limit: int = 12):
    items = []
    for feed in NEWS_FEED_URLS:
        try:
            response = requests.get(feed["url"], headers=REQUEST_HEADERS, timeout=8)
            response.raise_for_status()
        except Exception:
            continue

        try:
            root = ET.fromstring(response.content)
        except Exception:
            continue

        items.extend(extract_feed_items(root, fallback_source=feed["name"]))
        if len(items) >= limit:
            break

    seen = set()
    unique_items = []
    for item in items:
        title_key = item["title"].lower()
        if title_key in seen:
            continue
        seen.add(title_key)
        unique_items.append(item)
        if len(unique_items) >= limit:
            break

    return unique_items


@st.cache_data(ttl=3600)
def get_company_aliases(ticker_symbol: str):
    aliases = {ticker_symbol.upper()}
    if not ticker_symbol:
        return aliases

    try:
        info = yf.Ticker(ticker_symbol).info
    except Exception:
        return aliases

    for key in ("shortName", "longName", "displayName"):
        raw_name = info.get(key)
        if not raw_name:
            continue
        aliases.add(raw_name.lower())
        for token in raw_name.replace(",", " ").replace(".", " ").split():
            token = token.strip().lower()
            if len(token) > 2 and token not in GENERIC_COMPANY_WORDS:
                aliases.add(token)

    return aliases


def analyze_news_sentiment(news_items, ticker_symbol: str):
    aliases = get_company_aliases((ticker_symbol or "").strip().upper())
    positive_count = 0
    negative_count = 0
    relevant_count = 0
    scored_items = []

    for item in news_items:
        normalized_title = clean_text(item.get("title", "")).lower()
        words = set(normalized_title.replace("-", " ").replace("/", " ").split())
        positive_hits = sum(1 for term in POSITIVE_NEWS_TERMS if term in words or term in normalized_title)
        negative_hits = sum(1 for term in NEGATIVE_NEWS_TERMS if term in words or term in normalized_title)
        relevance = any(alias and alias.lower() in normalized_title for alias in aliases)

        if relevance:
            positive_hits += 1
            relevant_count += 1

        if positive_hits > negative_hits:
            sentiment = "Positive"
            score = positive_hits - negative_hits
            positive_count += 1
        elif negative_hits > positive_hits:
            sentiment = "Negative"
            score = positive_hits - negative_hits
            negative_count += 1
        else:
            sentiment = "Neutral"
            score = 0

        scored_items.append({**item, "sentiment": sentiment, "score": score, "relevant": relevance})

    if not scored_items:
        return {
            "recommendation": "Hold",
            "rationale": "No live headlines were available, so a news-based trade call would be unreliable.",
            "aggregate_score": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "relevant_count": 0,
            "items": [],
        }

    aggregate_score = sum(item["score"] for item in scored_items)
    if relevant_count == 0:
        aggregate_score *= 0.5

    if aggregate_score >= 2 or (positive_count >= negative_count + 2 and relevant_count > 0):
        recommendation = "Buy"
        rationale = "Headline tone is mostly favorable, suggesting positive short-term sentiment."
    elif aggregate_score <= -2 or (negative_count >= positive_count + 2 and relevant_count > 0):
        recommendation = "Sell"
        rationale = "Headline tone is mostly negative, which raises short-term downside risk."
    else:
        recommendation = "Hold"
        rationale = "Headline tone is mixed or limited, so waiting for stronger confirmation is safer."

    return {
        "recommendation": recommendation,
        "rationale": rationale,
        "aggregate_score": float(aggregate_score),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": max(0, len(scored_items) - positive_count - negative_count),
        "relevant_count": relevant_count,
        "items": scored_items,
    }


def render_sentiment_panel(summary, ticker_symbol: str):
    recommendation = summary["recommendation"]
    css_class = {
        "Buy": "sentiment-buy",
        "Hold": "sentiment-hold",
        "Sell": "sentiment-sell",
    }[recommendation]

    st.markdown(
        f"""
        <div class="sentiment-card">
            <div class="news-meta">Ticker under review</div>
            <h2 style="margin: 8px 0 4px 0;">{ticker_symbol}</h2>
            <div style="font-size: 1.1rem; color: #e2e8f0;">{summary['rationale']}</div>
            <div class="sentiment-label {css_class}">{recommendation} Signal</div>
            <div class="stats-grid">
                <div class="stat-tile">
                    <div class="stat-label">Positive</div>
                    <div class="stat-value">{summary['positive_count']}</div>
                </div>
                <div class="stat-tile">
                    <div class="stat-label">Negative</div>
                    <div class="stat-value">{summary['negative_count']}</div>
                </div>
                <div class="stat-tile">
                    <div class="stat-label">Neutral</div>
                    <div class="stat-value">{summary['neutral_count']}</div>
                </div>
                <div class="stat-tile">
                    <div class="stat-label">Score</div>
                    <div class="stat-value">{summary['aggregate_score']:.1f}</div>
                </div>
            </div>
            <div class="news-meta" style="margin-top: 14px;">
                Stock-related headlines found: {summary['relevant_count']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_headline_card(item):
    tone_class = {
        "Positive": "headline-positive",
        "Neutral": "headline-neutral",
        "Negative": "headline-negative",
    }[item["sentiment"]]
    meta = item.get("source") or "Market Feed"
    if item.get("pub_date"):
        meta = f"{meta} | {item['pub_date']}"
    relevance_note = "Stock-related" if item["relevant"] else "Broad market"
    title_html = f'<a href="{item["link"]}" target="_blank">{item["title"]}</a>' if item.get("link") else item["title"]
    st.markdown(
        f"""
        <div class="news-card">
            {title_html}
            <div class="headline-tag {tone_class}">{item['sentiment']}</div>
            <div class="news-meta">{meta} | {relevance_note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


default_ticker = st.session_state.get("app_selected_ticker") or "AAPL"
st.sidebar.header("News Controls")
ticker = st.sidebar.text_input("Stock Ticker", default_ticker).strip().upper() or "AAPL"
headline_limit = st.sidebar.slider("Headline Count", 6, 20, 10, 1)
st.session_state["app_selected_ticker"] = ticker

news_items = fetch_market_news(limit=headline_limit)
sentiment_summary = analyze_news_sentiment(news_items, ticker)

st.markdown("<div class='section-title'>News-Based Recommendation</div>", unsafe_allow_html=True)
left_col, right_col = st.columns([1, 1.2], gap="large")

with left_col:
    render_sentiment_panel(sentiment_summary, ticker)
    st.caption("This is a headline-based heuristic. Use it as a supporting signal, not standalone financial advice.")

with right_col:
    st.markdown("<div class='section-title'>Latest Headlines</div>", unsafe_allow_html=True)
    if not sentiment_summary["items"]:
        st.caption("No market headlines are available right now.")
    else:
        for item in sentiment_summary["items"][:5]:
            render_headline_card(item)

st.markdown("<div class='section-title'>Detailed Sentiment Breakdown</div>", unsafe_allow_html=True)
if sentiment_summary["items"]:
    for item in sentiment_summary["items"]:
        render_headline_card(item)
else:
    st.info("Try again in a few minutes to refresh live headlines.")
