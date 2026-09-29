"""RAG-style class-level OD line retrieval for EXP-OD2 (selective injection).

Embeds class names with sentence-transformers/all-MiniLM-L6-v2 (loaded from
the repo-local copy at playground/models/all-MiniLM-L6-v2 -- pure local
folder, no HF hub calls, no sentence-transformers dependency; mean pooling
done here exactly as the ST framework would) and, given a query (question /
referring description), selects the top-C classes by cosine similarity.  The
OD text keeps only the bbox lines of those classes, in original order and
numbering, with coordinates rounded to 2 decimals (same rounding as
scripts/generate_with_od_data.py, so the injected block format is
byte-compatible with EXP-OD1).

Shared by training-data generation (generate_with_od_topk_data.py) and the
eval-side selective injection (llava/eval/model_*.py --od-topk).

Usage (self-test):
    python scripts/od_retrieval.py
"""

import os
import re

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO, "playground", "models", "all-MiniLM-L6-v2")

BBOX_RE = re.compile(r"^bbox_(\d+)=Bbox\((\w+),")
NUM_RE = re.compile(r"-?\d+\.\d+")
DEFAULT_TOP_C = 3


def round_od(text: str) -> str:
    return NUM_RE.sub(lambda m: f"{float(m.group()):.2f}", text)


def parse_od_lines(od_text):
    """-> list of (bbox_id, class_name, raw_line); non-bbox lines are dropped."""
    out = []
    for line in od_text.strip().splitlines():
        m = BBOX_RE.match(line)
        if m:
            out.append((m.group(1), m.group(2), line))
    return out


class OdRetriever:
    """Cosine-similarity class retrieval over OD bbox lines via MiniLM."""

    def __init__(self, model_dir=MODEL_DIR, device=None, max_length=256):
        import torch  # deferred: only needed when retrieval is actually used
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir)
        self.model.eval().to(self.device)
        self._cls_cache = {}  # class name -> L2-normalized embedding (np.float32)

    @property
    def dim(self):
        return self.model.config.hidden_size

    def embed(self, texts):
        """-> (n, d) L2-normalized float32 embeddings (mean pooling)."""
        emb = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i in range(0, len(texts), 256):
            batch = [t if t.strip() else " " for t in texts[i : i + 256]]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self.torch.no_grad():
                out = self.model(**inputs).last_hidden_state  # (b, l, d)
            mask = inputs["attention_mask"].unsqueeze(-1).to(out.dtype)
            pooled = (out * mask).sum(1) / mask.sum(1)  # masked mean pooling
            pooled = pooled / pooled.norm(dim=-1, keepdim=True)
            emb[i : i + len(batch)] = pooled.float().cpu().numpy()
        return emb

    def class_scores(self, query, classes):
        """-> {class: cosine(query, class name)} for the given class names."""
        missing = [c for c in classes if c not in self._cls_cache]
        if missing:
            for c, e in zip(missing, self.embed(missing)):
                self._cls_cache[c] = e
        q = self.embed([query])[0]
        return {c: float(np.dot(q, self._cls_cache[c])) for c in classes}

    def select_lines(self, od_text, query, top_c=DEFAULT_TOP_C):
        """-> (selected [(bbox_id, cls, line)], {class: score}) in original order."""
        lines = parse_od_lines(od_text)
        if not lines:
            return [], {}
        classes = sorted({c for _, c, _ in lines})
        scores = self.class_scores(query, classes)
        keep = set(sorted(classes, key=lambda c: -scores[c])[:top_c])
        return [(i, c, l) for i, c, l in lines if c in keep], scores

    def select_od_text(self, od_text, query, top_c=DEFAULT_TOP_C):
        """Selected bbox lines, original order, coords rounded to 2 decimals."""
        sel, _ = self.select_lines(od_text, query, top_c)
        return "\n".join(round_od(l) for _, _, l in sel)


if __name__ == "__main__":
    sample = (
        "bbox_0=Bbox(window,2.92834906912758,0.3213458479071657,1.5539944245169561,-3.121965,0.96875,0.234375,1.8125)\n"
        "bbox_1=Bbox(chair,1.2345678,2.3456789,0.87654321,-0.523598,0.987654,0.123456,0.654321)\n"
        "bbox_2=Bbox(door,4.123456,0.654321,1.234567,1.570796,0.111111,0.222222,0.333333)\n"
        "bbox_3=Bbox(chair,5.678901,2.345678,0.987654,-1.047197,0.555555,0.666666,0.777777)\n"
        "bbox_4=Bbox(sink,3.141592,1.111111,0.888888,0.785398,0.444444,0.999999,1.000000)"
    )
    r = OdRetriever()
    print(f"device: {r.device}, dim: {r.dim}")
    for query in [
        "What color is the chair next to the desk?",
        "Please close the bathroom door.",
        "Where can I wash my hands?",
    ]:
        sel, scores = r.select_lines(sample, query, top_c=2)
        print(f"\nquery: {query!r}")
        print("  scores:", {k: round(v, 4) for k, v in sorted(scores.items(), key=lambda kv: -kv[1])})
        print("  kept:", [f"bbox_{i}({c})" for i, c, _ in sel])
        print("  text:\n    " + r.select_od_text(sample, query, top_c=2).replace("\n", "\n    "))
