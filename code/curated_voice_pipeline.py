"""Curated Voice: reproduction pipeline.

Reproduces Tables 2, 3 and S5 to S8 of the manuscript from the derived data in ../data.
Usage: python curated_voice_pipeline.py --data ../data --out ../results
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

# matplotlib is optional — the notebook still runs without figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:  # pragma: no cover
    HAS_MPL = False

B_DEFAULT = 1000
SEED_DEFAULT = 20260821


# Built-in stop-word list (subset of NLTK English). Used so the notebook does
# not depend on nltk.download(), which often fails in closed/offline
# environments.
_STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can't cannot could couldn't did
didn't do does doesn't doing don't down during each few for from further had
hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it
it's its itself let's me more most mustn't my myself no nor not of off on once
only or other ought our ours ourselves out over own same shan't she she'd
she'll she's should shouldn't so some such than that that's the their theirs
them themselves then there there's these they they'd they'll they're they've
this those through to too under until up very was wasn't we we'd we'll we're
we've were weren't what what's when when's where where's which while who who's
whom why why's with won't would wouldn't you you'd you'll you're you've your
yours yourself yourselves rt amp via just get like new one will now
""".split())

# Domain terms removed (S2). "nft" and "defi" are INTENTIONALLY kept.
DOMAIN_STOPWORDS = {"crypto", "blockchain", "web3"}

# Data-collection keywords — MUST match Table S1.1 / S1.2 of the manuscript,
# to avoid circular frequency inflation.
COLLECTION_KEYWORDS = {
    # NFT (S1.1)
    "nfts", "nonfungible", "marketplace", "opensea", "blur", "project",
    "community", "mint", "minting", "collection", "pfp", "airdrop",
    # DeFi (S1.2)
    "decentralized", "finance", "web3finance", "yieldfarming", "staking",
    "liquidity", "pool", "lending", "protocol", "swap", "dex", "tvl",
}

_URL = re.compile(r"https?://\S+|www\.\S+")
_HANDLE = re.compile(r"@\w+")
_RTMARK = re.compile(r"^\s*rt\b[:\s]*", re.I)
_NUM = re.compile(r"\d+")
_PUNCT = re.compile(r"[^\w\s]")


def preprocess(text: str, extra_stop: set[str] | None = None) -> list[str]:
    """S2 pipeline: lowercase; strip URLs/handles/numbers/punctuation; keep
    hashtag text (only the # symbol is removed); then filter stop words."""
    if not isinstance(text, str):
        return []
    t = _RTMARK.sub(" ", text)
    t = _URL.sub(" ", t)
    t = _HANDLE.sub(" ", t)
    t = t.lower()
    t = t.replace("#", " ")          # symbol removed, hashtag text retained
    t = _NUM.sub(" ", t)
    t = _PUNCT.sub(" ", t)
    stop = _STOPWORDS | DOMAIN_STOPWORDS | COLLECTION_KEYWORDS | (extra_stop or set())
    return [w for w in t.split() if len(w) > 1 and w not in stop]


def tokenise_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add `tokens` and `n_tok` columns."""
    out = df.copy()
    out["tokens"] = out["text"].map(preprocess)
    out["n_tok"] = out["tokens"].map(len)
    return out[out["n_tok"] > 0].reset_index(drop=True)


def counts_from_tokens(token_lists) -> dict[str, int]:
    c: dict[str, int] = {}
    for toks in token_lists:
        for w in toks:
            c[w] = c.get(w, 0) + 1
    return c


def entropy_bits(counts: dict[str, int]) -> float:
    """Plug-in (maximum likelihood) Shannon entropy, in bits.
    THIS is what the manuscript uses — and it is the version biased by
    sample size."""
    n = sum(counts.values())
    if n == 0:
        return 0.0
    p = np.fromiter(counts.values(), dtype=float) / n
    return float(-(p * np.log2(p)).sum())


def entropy_miller_madow(counts: dict[str, int]) -> float:
    """First-order bias correction: H_MM = H_plugin + (V-1) / (2 N ln2).
    Reduces (does not eliminate) downward bias on small samples."""
    n = sum(counts.values())
    v = len(counts)
    if n == 0:
        return 0.0
    return entropy_bits(counts) + (v - 1) / (2 * n * np.log(2))


def entropy_normalised(counts: dict[str, int]) -> float:
    """H / log2(V) — "evenness". This is the column reported in Table S6."""
    v = len(counts)
    if v <= 1:
        return 0.0
    return entropy_bits(counts) / np.log2(v)


def hill1(counts: dict[str, int]) -> float:
    """Hill number q=1 = 2^H = "effective vocabulary size".
    Easier to interpret than bits: the number of equally common words this
    is equivalent to."""
    return float(2 ** entropy_bits(counts))


def jsd_bits(c1: dict[str, int], c2: dict[str, int]) -> float:
    """Base-2 Jensen-Shannon Divergence, bounded in [0, 1]."""
    vocab = set(c1) | set(c2)
    n1, n2 = sum(c1.values()), sum(c2.values())
    if n1 == 0 or n2 == 0:
        return float("nan")
    p = np.array([c1.get(w, 0) for w in vocab], dtype=float) / n1
    q = np.array([c2.get(w, 0) for w in vocab], dtype=float) / n2
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def gini(x: np.ndarray) -> float:
    """Gini coefficient over non-negative values."""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def subsample_to_token_budget(token_lists, budget: int, rng) -> tuple[list, list]:
    """Randomly draw tweets (WITHOUT replacement) until the token budget is
    reached. Sampling is done at the tweet level, not the token level, so
    document structure (retweets, repeated phrases) is preserved — this is
    more conservative and more honest than shuffling individual tokens."""
    order = rng.permutation(len(token_lists))
    picked, rest, total = [], [], 0
    for i in order:
        if total < budget:
            picked.append(token_lists[i])
            total += len(token_lists[i])
        else:
            rest.append(token_lists[i])
    return picked, rest


@dataclass
class EntropyRow:
    community: str
    group: str
    n_tweets: int
    n_tokens: int
    vocab_V: int
    H_plugin: float
    H_miller_madow: float
    H_normalised: float
    hill1_effective_vocab: float


def run_entropy(df: pd.DataFrame, B: int, seed: int, outdir: str = "."):
    rng = np.random.default_rng(seed)
    rows, nulls = [], []

    for comm, sub in df.groupby("community", sort=False):
        cen = sub[sub["group"] == "central"]["tokens"].tolist()
        reg = sub[sub["group"] == "regular"]["tokens"].tolist()
        if not cen or not reg:
            print(f"  [skip] {comm}: needs both central & regular groups")
            continue

        c_cen, c_reg = counts_from_tokens(cen), counts_from_tokens(reg)
        for g, toks, cnt in (("central", cen, c_cen), ("regular", reg, c_reg)):
            rows.append(EntropyRow(comm, g, len(toks), sum(cnt.values()), len(cnt),
                                   entropy_bits(cnt), entropy_miller_madow(cnt),
                                   entropy_normalised(cnt), hill1(cnt)))

        # ---- rarefaction null: the regular corpus downsampled to the central corpus's size ----
        budget = sum(c_cen.values())
        h_null = np.empty(B)
        for b in range(B):
            picked, _ = subsample_to_token_budget(reg, budget, rng)
            h_null[b] = entropy_bits(counts_from_tokens(picked))

        h_cen = entropy_bits(c_cen)
        h_reg = entropy_bits(c_reg)
        mu, sd = h_null.mean(), h_null.std(ddof=1)
        z = (h_cen - mu) / sd if sd > 0 else float("nan")
        pct = float((h_null < h_cen).mean() * 100)
        p_emp = (1 + int((h_null <= h_cen).sum())) / (B + 1)

        naive_drop = (h_reg - h_cen) / h_reg * 100
        artefact_drop = (h_reg - mu) / h_reg * 100
        residual = naive_drop - artefact_drop

        nulls.append(dict(
            community=comm, H_central=h_cen, H_regular_full=h_reg,
            H_regular_rarefied_mean=mu, H_regular_rarefied_sd=sd,
            ci_lo=float(np.percentile(h_null, 2.5)),
            ci_hi=float(np.percentile(h_null, 97.5)),
            z_score=z, percentile_of_central=pct, p_one_sided=p_emp,
            drop_naive_pct=naive_drop, drop_explained_by_size_pct=artefact_drop,
            drop_residual_pct=residual,
            verdict=("narrowing SURVIVES" if p_emp < 0.05 else "does NOT survive"),
            _null=h_null,
        ))

    _print_entropy(rows, nulls)
    pd.DataFrame([asdict(r) for r in rows]).to_csv(f"{outdir}/results_entropy_descriptives.csv", index=False)
    pd.DataFrame([{k: v for k, v in n.items() if k != "_null"} for n in nulls]).to_csv(
        f"{outdir}/results_entropy_rarefaction.csv", index=False)
    if HAS_MPL and nulls:
        _plot_null(nulls, "H_central", "Shannon entropy (bits)",
                   "Rarefaction null: entropy of the regular corpus at the central corpus's size",
                   f"{outdir}/fig_entropy_rarefaction.png")
    return rows, nulls


def _print_entropy(rows, nulls):
    print("\n" + "=" * 96)
    print("A. ENTROPY — descriptives")
    print("=" * 96)
    print(f"{'Community':<11}{'Group':<10}{'Tweets':>9}{'Tokens':>11}{'V':>9}"
          f"{'H':>9}{'H_MM':>9}{'H/log2V':>10}{'eff.vocab':>12}")
    print("-" * 96)
    for r in rows:
        print(f"{r.community:<11}{r.group:<10}{r.n_tweets:>9,}{r.n_tokens:>11,}{r.vocab_V:>9,}"
              f"{r.H_plugin:>9.3f}{r.H_miller_madow:>9.3f}{r.H_normalised:>10.4f}"
              f"{r.hill1_effective_vocab:>12,.0f}")

    print("\n" + "=" * 96)
    print("A. ENTROPY — rarefaction test  (question: does the narrowing survive?)")
    print("=" * 96)
    for n in nulls:
        print(f"\n  {n['community']}")
        print(f"    H central                                 : {n['H_central']:.3f}")
        print(f"    H regular (full corpus)                   : {n['H_regular_full']:.3f}")
        print(f"    H regular RAREFIED to central's size      : {n['H_regular_rarefied_mean']:.3f} "
              f"[95% CI {n['ci_lo']:.3f}–{n['ci_hi']:.3f}]")
        print(f"    ------------------------------------------")
        print(f"    Reduction reported in the manuscript      : {n['drop_naive_pct']:.1f}%")
        print(f"    Portion explained purely by corpus size   : {n['drop_explained_by_size_pct']:.1f}%")
        print(f"    REMAINING substantive reduction           : {n['drop_residual_pct']:.1f}%")
        print(f"    z = {n['z_score']:.2f}   percentile = {n['percentile_of_central']:.1f}   "
              f"p (one-sided) = {n['p_one_sided']:.4f}")
        print(f"    >> {n['verdict']}")


def run_jsd(df: pd.DataFrame, B: int, seed: int, outdir: str = "."):
    rng = np.random.default_rng(seed + 1)
    out, nulls = [], []

    for comm, sub in df.groupby("community", sort=False):
        cen = sub[sub["group"] == "central"]["tokens"].tolist()
        reg = sub[sub["group"] == "regular"]["tokens"].tolist()
        if not cen or not reg:
            continue

        c_cen, c_reg = counts_from_tokens(cen), counts_from_tokens(reg)
        observed = jsd_bits(c_cen, c_reg)
        budget = sum(c_cen.values())

        # Null: "if central-actor discourse were simply a random sample of
        # community discourse of the same size, what JSD would we observe?"
        j_null = np.empty(B)
        for b in range(B):
            picked, rest = subsample_to_token_budget(reg, budget, rng)
            j_null[b] = jsd_bits(counts_from_tokens(picked), counts_from_tokens(rest))

        mu, sd = j_null.mean(), j_null.std(ddof=1)
        p_emp = (1 + int((j_null >= observed).sum())) / (B + 1)
        out.append(dict(
            community=comm, jsd_observed=observed,
            jsd_null_mean=mu, jsd_null_sd=sd,
            null_ci_lo=float(np.percentile(j_null, 2.5)),
            null_ci_hi=float(np.percentile(j_null, 97.5)),
            jsd_excess=observed - mu,
            z_score=(observed - mu) / sd if sd > 0 else float("nan"),
            p_one_sided=p_emp,
            verdict=("genuine divergence" if p_emp < 0.05 else "NOT above noise"),
        ))
        nulls.append(dict(community=comm, H_central=observed, _null=j_null))

    print("\n" + "=" * 96)
    print("B. JSD — noise floor from size asymmetry")
    print("=" * 96)
    for r in out:
        print(f"\n  {r['community']}")
        print(f"    Observed JSD                : {r['jsd_observed']:.3f}")
        print(f"    Null JSD (matched size)     : {r['jsd_null_mean']:.3f} "
              f"[95% CI {r['null_ci_lo']:.3f}–{r['null_ci_hi']:.3f}]")
        print(f"    EXCESS above null           : {r['jsd_excess']:+.3f}")
        print(f"    z = {r['z_score']:.2f}   p (one-sided) = {r['p_one_sided']:.4f}")
        print(f"    >> {r['verdict']}")
    print("\n  NOTE: what should be reported in the manuscript as evidence is the")
    print("  EXCESS above null, not the raw JSD. Compare the excess for NFT vs DeFi \u2014")
    print("  that is where a claim like 'DeFi is nearly double NFT' needs to be re-tested.")

    pd.DataFrame(out).to_csv(f"{outdir}/results_jsd_permutation.csv", index=False)
    if HAS_MPL and nulls:
        _plot_null(nulls, "H_central", "Jensen-Shannon Divergence (bits)",
                   "Permutation null: JSD between two random samples of regular discourse",
                   f"{outdir}/fig_jsd_permutation.png")
    return out


def run_gini(edges: pd.DataFrame, B: int, seed: int, outdir: str = "."):
    rng = np.random.default_rng(seed + 2)
    out = []

    for comm, sub in edges.groupby("community", sort=False):
        nodes = pd.unique(pd.concat([sub["source"], sub["target"]]).values)
        idx = {u: i for i, u in enumerate(nodes)}
        N, E = len(nodes), len(sub)
        w = sub["weight"].to_numpy(dtype=float) if "weight" in sub else np.ones(E)

        wdeg = np.zeros(N)
        np.add.at(wdeg, sub["source"].map(idx).to_numpy(), w)
        np.add.at(wdeg, sub["target"].map(idx).to_numpy(), w)
        observed = gini(wdeg)

        # Erdos-Renyi null: N nodes, E edges, endpoints drawn uniformly at
        # random, weights shuffled.
        # IMPORTANT NOTE: a configuration model is NOT useful here — it
        # preserves the degree sequence, so its Gini is identical by
        # construction.
        g_null = np.empty(B)
        for b in range(B):
            s = rng.integers(0, N, size=E)
            t = rng.integers(0, N, size=E)
            ww = rng.permutation(w)
            d = np.zeros(N)
            np.add.at(d, s, ww)
            np.add.at(d, t, ww)
            g_null[b] = gini(d)

        mu = g_null.mean()
        out.append(dict(community=comm, n_nodes=N, n_edges=E,
                        edges_per_node=E / N, gini_observed=observed,
                        gini_random_null=mu,
                        null_ci_lo=float(np.percentile(g_null, 2.5)),
                        null_ci_hi=float(np.percentile(g_null, 97.5)),
                        gini_excess=observed - mu))

    print("\n" + "=" * 96)
    print("C. GINI — compared against a random network with the same N and E")
    print("=" * 96)
    print(f"{'Community':<11}{'N':>10}{'E':>10}{'E/N':>7}{'Gini':>9}{'Gini null':>11}{'Excess':>11}")
    print("-" * 96)
    for r in out:
        print(f"{r['community']:<11}{r['n_nodes']:>10,}{r['n_edges']:>10,}"
              f"{r['edges_per_node']:>7.2f}{r['gini_observed']:>9.3f}"
              f"{r['gini_random_null']:>11.3f}{r['gini_excess']:>+11.3f}")
    print("\n  NOTE: this 'excess' is the answer to the reviewer's question")
    print("  'compared to what?'. A Gini of 0.9 that sits only slightly above null means")
    print("  your concentration is mostly a consequence of edge sparsity, not a")
    print("  genuine finding about Web3. Ideally, add ONE non-Web3 community collected")
    print("  with an identical procedure as an empirical comparison.")

    pd.DataFrame(out).to_csv(f"{outdir}/results_gini_null.csv", index=False)
    return out


def run_halo(df: pd.DataFrame, outdir: str = "."):
    if "halo_role" not in df.columns:
        print("\n[D. HALO] skipped: `halo_role` column not present.")
        print("  Add a column with 'authored' / 'received' values for rows where group=central.")
        print("  This matters: your Supplementary S3 states out-degree = 0 for")
        print("  all central accounts \u2014 if that is correct, the 'authored' sub-corpus")
        print("  will be EMPTY, meaning the halo corpus consists entirely of other")
        print("  people's voices.")
        return []

    out = []
    for comm, sub in df.groupby("community", sort=False):
        reg = sub[sub["group"] == "regular"]["tokens"].tolist()
        c_reg = counts_from_tokens(reg)
        for role in ("authored", "received"):
            part = sub[(sub["group"] == "central") & (sub["halo_role"] == role)]["tokens"].tolist()
            if not part:
                out.append(dict(community=comm, halo_role=role, n_tweets=0,
                                n_tokens=0, vocab_V=0, H_plugin=float("nan"),
                                H_normalised=float("nan"), jsd_vs_regular=float("nan")))
                continue
            c = counts_from_tokens(part)
            out.append(dict(community=comm, halo_role=role, n_tweets=len(part),
                            n_tokens=sum(c.values()), vocab_V=len(c),
                            H_plugin=entropy_bits(c),
                            H_normalised=entropy_normalised(c),
                            jsd_vs_regular=jsd_bits(c, c_reg)))

    print("\n" + "=" * 96)
    print("D. HALO DECOMPOSITION \u2014 authored by central actors vs addressed to them")
    print("=" * 96)
    print(f"{'Community':<11}{'Role':<11}{'Tweets':>9}{'Tokens':>11}{'V':>9}{'H':>9}"
          f"{'H/log2V':>10}{'JSD vs reg':>12}")
    print("-" * 96)
    for r in out:
        print(f"{r['community']:<11}{r['halo_role']:<11}{r['n_tweets']:>9,}{r['n_tokens']:>11,}"
              f"{r['vocab_V']:>9,}{r['H_plugin']:>9.3f}{r['H_normalised']:>10.4f}"
              f"{r['jsd_vs_regular']:>12.3f}")
    print("\n  NOTE: if 'authored' is empty or very small, the term 'curated")
    print("  voice' cannot refer to the central actors' own voice. Reframe it")
    print("  as an addressivity effect, or explicitly scale back the claim.")

    pd.DataFrame(out).to_csv(f"{outdir}/results_halo_decomposition.csv", index=False)
    return out


def _plot_null(nulls, obs_key, xlabel, title, path):
    n = len(nulls)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 3.6), squeeze=False)
    for ax, d in zip(axes[0], nulls):
        ax.hist(d["_null"], bins=40, color="#c9d4ea", edgecolor="#8fa4c8", linewidth=0.5)
        ax.axvline(d[obs_key], color="#a8202b", linewidth=2,
                   label=f"observed = {d[obs_key]:.3f}")
        ax.axvline(d["_null"].mean(), color="#23408e", linestyle="--", linewidth=1.4,
                   label=f"null mean = {d['_null'].mean():.3f}")
        ax.set_title(d["community"], fontsize=11)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("frequency (bootstrap)", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=11.5, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    [figure] {path}")


def make_demo(seed: int = SEED_DEFAULT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic data where central and regular discourse are drawn
    from the SAME distribution, with size ratios similar to your data. Any
    'narrowing' that appears here is a pure artefact."""
    rng = np.random.default_rng(seed)
    letters = np.array(list("abcdefghijklmnopqrstuvwxyz"))
    vocab = ["".join(rng.choice(letters, size=6)) for _ in range(40_000)]
    p = np.arange(1, 40_001, dtype=float) ** -1.0
    p /= p.sum()

    def corpus(n_tweets, comm, grp, role=None):
        rows = []
        for _ in range(n_tweets):
            k = max(1, int(rng.normal(14, 4)))
            rows.append(dict(community=comm, group=grp,
                             text=" ".join(rng.choice(vocab, size=k, p=p)),
                             halo_role=role))
        return rows

    tw = (corpus(1200, "NFT", "central", "received") + corpus(200, "NFT", "central", "authored")
          + corpus(8000, "NFT", "regular")
          + corpus(120, "DeFi", "central", "received") + corpus(10, "DeFi", "central", "authored")
          + corpus(9000, "DeFi", "regular"))

    ed = []
    for comm, N, E in (("NFT", 4000, 5400), ("DeFi", 4500, 6300)):
        s = rng.integers(0, N, E)
        t = rng.integers(0, int(N * 0.08), E)   # concentrated target
        for i in range(E):
            ed.append(dict(community=comm, source=f"u{s[i]}", target=f"h{t[i]}",
                           weight=int(rng.integers(1, 4))))
    return pd.DataFrame(tw), pd.DataFrame(ed)


def load_tweets(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"community", "group", "text"} - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: missing columns in {path}: {sorted(missing)}")
    df["group"] = df["group"].str.strip().str.lower()
    print(f"[read] {path}: {len(df):,} rows")
    return tokenise_frame(df)




# =====================================================================
# Reproduction of the results reported in the manuscript
# =====================================================================
import argparse, os, json
CENTRAL = {
    "main":         {"NFT": {"opensea", "giverep", "hubdotxyz"},        "DeFi": {"lombard_finance", "infinit_labs", "giverep"}},
    "nft_top2":     {"NFT": {"opensea", "giverep"},                     "DeFi": {"lombard_finance", "infinit_labs", "giverep"}},
    "all_accounts": {"NFT": {"opensea", "freeexyz", "giverep"},         "DeFi": {"lombard_finance", "infinit_labs", "giverep"}},
}
NAME_TOKENS = {"opensea", "openseanfts", "sea", "giverep", "freeexyz", "hubdotxyz", "hub",
               "lombard", "lombard_finance", "infinit", "infinit_labs", "labs"}
SPECS = {
    # name: (central set, drop automated accounts, drop account-name tokens, drop duplicate records)
    "main":                    ("main",         True,  True,  False),
    "names_retained":          ("main",         True,  False, False),
    "nft_top2_names_retained": ("nft_top2",     True,  False, False),
    "all_accounts":            ("all_accounts", False, True,  False),
    "main_dedup":              ("main",         True,  True,  True),
}

def build_groups(tw, central):
    tw = tw.copy()
    tw["group"] = "regular"; tw["halo_role"] = None
    for comm, handles in central.items():
        m = tw["community"] == comm
        mention = np.zeros(m.sum(), dtype=bool)
        for h in handles:
            mention |= tw.loc[m, "mentions_" + h].to_numpy()
        authored = tw.loc[m, "authored_by_central"].fillna("").isin(handles).to_numpy()
        idx = tw.index[m]
        tw.loc[idx[mention | authored], "group"] = "central"
        tw.loc[idx[authored], "halo_role"] = "authored"
        tw.loc[idx[mention & ~authored], "halo_role"] = "received"
    return tw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--B", type=int, default=B_DEFAULT)
    ap.add_argument("--specs", default=",".join(SPECS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    raw = pd.concat([pd.read_csv(os.path.join(a.data, f"tweets_{c}.csv.gz")) for c in ("NFT", "DeFi")], ignore_index=True)
    edges = pd.read_csv(os.path.join(a.data, "edges.csv.gz"))
    for name in a.specs.split(","):
        cset, drop_auto, drop_names, dedup = SPECS[name]
        print("\n" + "#" * 96 + f"\nSPECIFICATION: {name}\n" + "#" * 96)
        tw = raw
        if drop_auto: tw = tw[~tw["automated_account"]]
        if dedup: tw = tw[~tw["duplicate_record"]]
        tw = build_groups(tw, CENTRAL[cset])
        DOMAIN_STOPWORDS.difference_update(NAME_TOKENS)
        if drop_names: DOMAIN_STOPWORDS.update(NAME_TOKENS)
        tw = tokenise_frame(tw)
        d = os.path.join(a.out, name); os.makedirs(d, exist_ok=True)
        run_entropy(tw, a.B, SEED_DEFAULT, d)
        run_jsd(tw, a.B, SEED_DEFAULT, d)
        run_halo(tw, d)
    for label, ed in (("gini_automated_removed", edges[~edges["source_automated"]]), ("gini_all_accounts", edges)):
        ed = ed.groupby(["community", "source", "target"], as_index=False)["weight"].sum()
        d = os.path.join(a.out, label); os.makedirs(d, exist_ok=True)
        print("\n" + "#" * 96 + f"\n{label}\n" + "#" * 96)
        run_gini(ed, a.B, SEED_DEFAULT, d)

if __name__ == "__main__":
    main()
