"""Hybrid, layout-aware retrieval: lexical + dense first stage, cross-encoder rerank, section
smoothing, and query-intent handling for article citations, whole-issue queries and abstention."""

import json
import math
import re
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from functools import cache
from pathlib import Path

import numpy as np

from .corpus import load
from .text import MONTHS, article_number, char_terms, date_span, expand_months, normalize, stem, tokens, word_terms

EMBEDDER, EMBEDDER_REVISION = "BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181"
RERANKER, RERANKER_REVISION = "BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"

RERANK_DEPTH = 30
RERANK_WEIGHT = 0.5
SECTION_WEIGHT = 0.5
MIN_EVIDENCE = 0.1  # reranker probability a passage needs to count as an answer
CONFIDENT_EVIDENCE = 0.5  # with a verifier: below this, the LLM decides whether the top passages answer
RELATIVE_PAGE_SCORE = 0.5  # without a reranker: page kept if within this fraction of the best page
OLLAMA_CHAT = "http://localhost:11434/api/chat"

_ISSUE_DATE = re.compile(r"\b(?:ال)?(?:عدد|اعداد)\s+(?:يوم\s+)?\d")
_EXHAUSTIVE = re.compile(r"\b(?:كل|جميع)\s+ما\b")
_PUBLISHED = frozenset(stem(t) for t in tokens("نشر نشرت ينشر تنشر نشره نشرها منشور المنشورة المنشورات"))
_CUE_WORDS = _PUBLISHED | frozenset(stem(t) for t in tokens(
    "عدد اعداد تم يوم لغاية مقال مقالات موضوع مواضيع صفحة صفحات السبت الأحد الاثنين الثلاثاء الأربعاء الخميس الجمعة"
))


@dataclass(frozen=True)
class Config:
    name: str
    channels: tuple[str, ...] = ("word", "char", "dense")
    layout: bool = True
    rerank: bool = True
    verifier: str | None = None  # Ollama model that checks answerability when the reranker is unsure


CONFIGS = (
    Config("bm25", channels=("word",), layout=False, rerank=False),
    Config("hybrid", layout=False, rerank=False),
    Config("hybrid+layout", rerank=False),
    Config("hybrid+layout+rerank"),
)


@dataclass
class Result:
    mode: str  # "passages", "pages" or "abstain"
    passages: list[str]
    pages: list[str]
    evidence: float | None = None  # best answer evidence found; None when decided by structure alone


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.tfs = [Counter(doc) for doc in docs]
        lengths = np.array([len(doc) for doc in docs], dtype=float)
        self.norm = k1 * (1 - b + b * lengths / lengths.mean())
        self.k1 = k1
        df = Counter(term for tf in self.tfs for term in tf)
        self.idf = {t: math.log(1 + (len(docs) - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, terms: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tfs))
        for term in set(terms) & self.idf.keys():
            tf = np.array([doc[term] for doc in self.tfs], dtype=float)
            out += self.idf[term] * tf * (self.k1 + 1) / (tf + self.norm)
        return out


def _load(cls, name: str, **kwargs):
    """Load from the local cache without touching the network; download only on first use."""
    try:
        return cls(name, local_files_only=True, **kwargs)
    except OSError:
        return cls(name, **kwargs)


@cache
def _embedder():
    from sentence_transformers import SentenceTransformer

    return _load(SentenceTransformer, EMBEDDER, revision=EMBEDDER_REVISION)


@cache
def _reranker():
    from sentence_transformers import CrossEncoder

    return _load(CrossEncoder, RERANKER, revision=RERANKER_REVISION, max_length=512)


def _verifies(model: str, question: str, passage: str) -> bool:
    """Ask a local Ollama model whether the passage contains the answer to the question."""
    prompt = (
        f"Question: {question}\n\nPassage:\n{passage}\n\n"
        "Does the passage contain the answer to the question? Reply with only yes or no."
    )
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 4},
    }
    request = urllib.request.Request(OLLAMA_CHAT, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        reply = json.loads(response.read())["message"]["content"]
    return reply.strip().lower().startswith(("yes", "نعم"))


@cache
def _embed(texts: tuple[str, ...]) -> np.ndarray:
    return _embedder().encode(list(texts), normalize_embeddings=True, batch_size=16)


class _Index:
    """Lexical and dense scorers over a list of documents."""

    def __init__(self, docs: list[str], channels: tuple[str, ...]):
        self.word = BM25([word_terms(d) for d in docs]) if "word" in channels else None
        self.char = BM25([char_terms(d) for d in docs]) if "char" in channels else None
        self.dense = _embed(tuple(docs)) if "dense" in channels else None

    def scores(self, query: str) -> list[np.ndarray]:
        out = []
        if self.word:
            out.append(self.word.scores(word_terms(expand_months(query))))
        if self.char:
            out.append(self.char.scores(char_terms(query)))
        if self.dense is not None:
            out.append(self.dense @ _embed((query,))[0])
        return out


def _is_topic_term(token: str) -> bool:
    return stem(token) not in _CUE_WORDS and token not in MONTHS and not token.isdigit()


def _minmax(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


class Retriever:
    def __init__(self, data_dir: Path, config: Config = CONFIGS[-1]):
        self.config = config
        self.pages, self.passages = load(data_dir)
        self.page_dates = {p.id: date.fromisoformat(p.date) for p in self.pages}
        self.latest_year = max(self.page_dates.values()).year
        self.normalized_text = "\n".join(normalize(p.text) for p in self.passages)
        units = [(i, text) for i, p in enumerate(self.passages) for text in (p.units if config.layout else [p.text])]
        self.unit_texts = [text for _, text in units]
        self.unit_owner = np.array([i for i, _ in units])
        self.index = _Index(self.unit_texts, config.channels)
        self.sections = np.array([p.section for p in self.passages])
        if config.layout:
            members = defaultdict(list)
            for p in self.passages:
                members[p.section].append(p.body)
            self.section_ids = sorted(members)
            self.section_index = _Index(["\n".join(members[s]) for s in self.section_ids], config.channels)

    def search(self, query: str) -> Result:
        norm = normalize(query)
        published = self.config.layout and (
            bool(_ISSUE_DATE.search(norm)) or any(stem(t) in _PUBLISHED for t in tokens(query))
        )
        span = date_span(query, self.latest_year) if published else None
        if span or (self.config.layout and _EXHAUSTIVE.search(norm)):
            return self._search_pages(query, span)

        scores, evidence = self._score(query)
        order = list(np.argsort(-scores, kind="stable"))
        cited = []
        if self.config.layout and (article := article_number(query)) is not None:
            cited = self._cited_article(article, scores)
            if not cited and not re.search(rf"ماده\s+{article}\b", self.normalized_text):
                return Result("abstain", [], [])
            order = cited + [i for i in order if i not in cited]
        best = None if evidence is None else round(float(evidence.max()), 3)
        if not cited and evidence is not None and not self._answerable(query, evidence, order):
            return Result("abstain", [], [], best)
        ids = [self.passages[i].id for i in order]
        return Result("passages", ids, list(dict.fromkeys(self.passages[i].page for i in order)), best)

    def _answerable(self, query: str, evidence: np.ndarray, order: list[int]) -> bool:
        """Enough reranker or lexical evidence; with a verifier, an uncertain case is decided by the LLM
        reading the top three passages in their context."""
        if not self.config.verifier or evidence.max() >= CONFIDENT_EVIDENCE:
            return evidence.max() >= MIN_EVIDENCE
        return any(_verifies(self.config.verifier, query, "\n".join(self.passages[i].units)) for i in order[:3])

    def _covered(self, query: str) -> np.ndarray:
        """Passages with a unit containing every content term of the query."""
        terms = set(word_terms(query))
        covered = np.zeros(len(self.passages))
        for unit, tf in enumerate(self.index.word.tfs):
            if terms and terms <= tf.keys():
                covered[self.unit_owner[unit]] = 1.0
        return covered

    def _score(self, query: str, within: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """Passage scores in [0, 1] and, when reranking, per-passage answer evidence: the reranker
        probability, or 1 when the passage contains every query term. `within` limits which
        passages are reranked."""
        n = len(self.passages)
        channels = []
        for unit_scores in self.index.scores(query):
            passage_scores = np.full(n, -np.inf)
            np.maximum.at(passage_scores, self.unit_owner, unit_scores)
            channels.append(_minmax(passage_scores))
        scores = np.mean(channels, axis=0)

        evidence = None
        if self.config.rerank:
            candidates = scores if within is None else np.where(within, scores, -np.inf)
            top = set(np.argsort(-candidates)[:RERANK_DEPTH])
            units = [u for u in range(len(self.unit_texts)) if self.unit_owner[u] in top]
            probs = _reranker().predict([(query, self.unit_texts[u]) for u in units], batch_size=16)
            relevance = np.zeros(n)
            np.maximum.at(relevance, self.unit_owner[units], probs)
            scores = (1 - RERANK_WEIGHT) * scores + RERANK_WEIGHT * relevance
            evidence = np.maximum(relevance, self._covered(query))

        if self.config.layout:
            section_scores = np.mean([_minmax(s) for s in self.section_index.scores(query)], axis=0)
            by_section = dict(zip(self.section_ids, section_scores))
            scores = (1 - SECTION_WEIGHT) * scores + SECTION_WEIGHT * np.array([by_section[s] for s in self.sections])
        return scores, evidence

    def _cited_article(self, article: int, scores: np.ndarray) -> list[int]:
        """Passages of the cited article, in page order; if several pages carry that article
        number, the page whose passages best match the rest of the query."""
        by_page = defaultdict(list)
        for i, p in enumerate(self.passages):
            if article in p.articles:
                by_page[p.page].append(i)
        if not by_page:
            return []
        return max(by_page.values(), key=lambda idx: scores[idx].max())

    def _search_pages(self, query: str, span: tuple[date, date] | None) -> Result:
        """Page-level answers: every page in a publication-date span, and/or every page about a topic."""
        pages = [p.id for p in self.pages if span is None or span[0] <= self.page_dates[p.id] <= span[1]]
        topic = " ".join(w for w in query.split() if any(_is_topic_term(t) for t in tokens(w)))
        if not pages:
            return Result("abstain", [], [])
        if not topic:
            pages.sort(key=lambda p: (self.page_dates[p], p))
            return Result("pages", [q.id for q in self.passages if q.page in pages], pages)

        scores, evidence = self._score(topic, within=np.array([p.page in pages for p in self.passages]))
        page_evidence = scores if evidence is None else evidence
        order = [i for i in np.argsort(-scores, kind="stable") if self.passages[i].page in pages]
        best = defaultdict(float)
        for i in order:
            best[self.passages[i].page] = max(best[self.passages[i].page], page_evidence[i])
        threshold = MIN_EVIDENCE if evidence is not None else RELATIVE_PAGE_SCORE * max(best.values())
        selected = [p for p in dict.fromkeys(self.passages[i].page for i in order) if best[p] >= threshold]
        strongest = None if evidence is None else round(float(max(best.values())), 3)
        if not selected:
            return Result("abstain", [], [], strongest)
        passages = [self.passages[i].id for i in order if self.passages[i].page in selected]
        return Result("pages", passages, selected, strongest)
