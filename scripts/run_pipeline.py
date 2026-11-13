"""Rebuild a run file from the saved generations: BM25 with the expanded query, then reranking.

No LLM call is made; the passages come from data/generations/ (GitHub Release ZIP).

  # one strategy (five passages)
  python scripts/run_pipeline.py --dataset scifact --passages strict --out strict.run
  # all-strategy integration (40 passages)
  python scripts/run_pipeline.py --dataset scifact --passages all8 --out all8.run
  # no expansion
  python scripts/run_pipeline.py --dataset scifact --passages none --out n0.run
  # stage-wise analysis: strict passages in BM25, all-strategy passages in reranking
  python scripts/run_pipeline.py --dataset scifact --passages strict --rerank-passages all8 --out a.run
  # passage-count control
  python scripts/run_pipeline.py --dataset scifact --passages strict:40 --out s40.run
  python scripts/run_pipeline.py --dataset scifact --passages pool:20:0 --out p20.run

Passage specifications (--passages, --rerank-passages):
  none              no expansion
  <strategy>        the five passages of one strategy (legacy, strict, aspect, hybrid, broaden,
                    corpus_strict, corpus_aspect, corpus_broaden)
  a+b+c             the passages of several strategies together (integration)
  all8              all eight strategies (40 passages); all6 = all but corpus_aspect/corpus_broaden
  strict:N          the first N strict passages of the passage-count control (N up to 40)
  pool:K:SEED       K passages drawn uniformly from the 40-passage pool of the eight strategies
  mix:S:K:SEED      K passages each from S strategies drawn at random

Pipeline (Section III-C of the paper):
  1. BM25 (Pyserini prebuilt index, k1 = 0.9, b = 0.4, top 100) with the query
     (q + " ") * t + passages,  t = max(1, floor(floor(|G| / |q|) / 6)),
     where |q| and |G| are the character lengths of the query and the concatenated passages.
  2. Reranking of the 100 candidates by cosine similarity to the re-normalized mean of the
     L2-normalized embeddings of "q + passage_i" (all-mpnet-base-v2, mean pooling).
     Documents are "Title: <title> Content: <text>" when the collection has titles.

Requires pyserini, torch and transformers. Evaluate the result with evaluate_runs.py.
"""
import argparse
import json
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import common as C

TOP = 100
REPEAT_DIVISOR = 6
ALL6 = [s for s in C.STRATEGIES if s not in ("corpus_aspect", "corpus_broaden")]


# ---------------------------------------------------------------- passages
class Passages:
    """Returns the passages of one query for a passage specification."""

    def __init__(self, dataset, spec):
        self.dataset, self.spec = dataset, spec
        p = spec.split(":")
        self.kind = "set"
        if spec == "none":
            self.kind, needed = "none", []
        elif p[0] == "pool":
            self.kind, self.k, self.seed, needed = "pool", int(p[1]), int(p[2]), C.STRATEGIES
        elif p[0] == "mix":
            self.kind, self.s, self.k, self.seed = "mix", int(p[1]), int(p[2]), int(p[3])
            needed = C.STRATEGIES
        elif len(p) == 2:
            if p[0] != "strict":
                raise SystemExit("only strict was extended beyond five passages (strict:N)")
            self.kind, self.n, needed = "count", int(p[1]), []
            self.count = C.load_count_control_generations(dataset)
        else:
            needed = {"all8": C.STRATEGIES, "all6": ALL6}.get(spec) or spec.split("+")
            unknown = [s for s in needed if s not in C.STRATEGIES]
            if unknown:
                raise SystemExit(f"unknown strategy: {unknown}")
            needed = [s for s in C.STRATEGIES if s in needed]      # order of Table I
        self.strategies = needed
        self.gens = {s: C.load_generations(dataset, s) for s in needed}

    def _rng(self, qid):
        # reproducible per query: seed x dataset x query id
        return np.random.default_rng(zlib.crc32(f"{self.seed}|{self.dataset}|{qid}".encode()))

    def get(self, qid):
        if self.kind == "none":
            return []
        if self.kind == "count":
            refs = self.count.get(qid, [])
            if len(refs) < self.n:
                raise SystemExit(f"query {qid} has only {len(refs)} strict passages")
            return refs[:self.n]
        if self.kind == "set":
            return [r for s in self.strategies for r in self.gens[s].get(qid, [])[:5]]
        pool = [r for s in C.STRATEGIES for r in self.gens[s].get(qid, [])]
        rng = self._rng(qid)
        if self.kind == "pool":
            idx = rng.choice(len(pool), size=min(self.k, len(pool)), replace=False)
            return [pool[i] for i in sorted(idx)]
        chosen = rng.choice(len(C.STRATEGIES), size=min(self.s, len(C.STRATEGIES)), replace=False)
        refs = []
        for si in sorted(chosen):
            avail = self.gens[C.STRATEGIES[si]].get(qid, [])
            j = rng.choice(len(avail), size=min(self.k, len(avail)), replace=False)
            refs.extend(avail[i] for i in sorted(j))
        return refs


# ---------------------------------------------------------------- first stage
def expanded_query(query, refs):
    if not refs:
        return query
    text = " ".join(refs)
    times = max(1, (len(text) // max(1, len(query))) // REPEAT_DIVISOR)
    return (query + " ") * times + text


def document_text(searcher, docid):
    doc = searcher.doc(docid)
    raw = doc.raw() if doc is not None else ""
    try:
        content = json.loads(raw)
        text = content.get("text", content.get("contents", ""))
        if "title" in content:
            text = f"Title: {content['title']} Content: {text}"
    except (json.JSONDecodeError, TypeError):
        text = ""
    return " ".join(text.split())


# ---------------------------------------------------------------- second stage
class Reranker:
    def __init__(self, model_name):
        self.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
        self.cls_pooling = "bge" in model_name.lower()      # bge uses the [CLS] vector
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def embed(self, texts):
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**tokens)
            if self.cls_pooling:
                emb = out[0][:, 0]
            else:
                mask = tokens["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
                emb = torch.sum(out.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        return F.normalize(emb, p=2, dim=-1)

    def rerank(self, query, refs, docids, docs):
        queries = [query + " " + r for r in refs] or [query]
        q = F.normalize(torch.mean(self.embed(queries), dim=0, keepdim=True), p=2, dim=1)
        scores = torch.matmul(q, self.embed(docs).T)
        top_scores, idx = scores.topk(min(TOP, len(docs)), dim=1)
        return [(docids[i], top_scores[0, k].item()) for k, i in enumerate(idx.reshape(-1).tolist())]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=C.MAIN_DATASETS + C.HELDOUT_DATASETS)
    ap.add_argument("--passages", required=True, help="passages used in the BM25 query (see above)")
    ap.add_argument("--rerank-passages", help="passages used in reranking (default: same as --passages)")
    ap.add_argument("--reranker", default="sentence-transformers/all-mpnet-base-v2",
                    help="e.g. BAAI/bge-base-en-v1.5 for the robustness check")
    ap.add_argument("--out", required=True, help="run file to write")
    ap.add_argument("--limit", type=int, help="only the first N queries (quick test)")
    args = ap.parse_args()

    from pyserini.search import get_topics
    from pyserini.search.lucene import LuceneSearcher

    qids = C.qids_of(args.dataset)[:args.limit]
    topics = {str(k): v["title"] for k, v in get_topics(C.PYSERINI_TOPICS[args.dataset]).items()}
    bm25_passages = Passages(args.dataset, args.passages)
    rerank_passages = (Passages(args.dataset, args.rerank_passages)
                       if args.rerank_passages else bm25_passages)
    searcher = LuceneSearcher.from_prebuilt_index(C.PYSERINI_INDEX[args.dataset])   # k1 0.9, b 0.4
    reranker = Reranker(args.reranker)

    ranking = {}
    for qid in tqdm(qids, desc=f"{args.dataset} {args.passages}"):
        query = topics[qid]
        hits = searcher.search(expanded_query(query, bm25_passages.get(qid)), k=TOP)
        if not hits:
            continue
        docids = [h.docid for h in hits]
        docs = [document_text(searcher, d) for d in docids]
        ranking[qid] = reranker.rerank(query, rerank_passages.get(qid), docids, docs)
    tag = args.passages if not args.rerank_passages else f"bm25-{args.passages}_rerank-{args.rerank_passages}"
    C.write_run(args.out, ranking, tag.replace(":", "_"))
    print(f"wrote {args.out} ({len(ranking)} queries)")


if __name__ == "__main__":
    main()
