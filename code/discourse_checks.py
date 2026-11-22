"""Is discourse addressed to top accounts specific to them? (Supplementary Material S4, Table S4)

Compares tweets addressing the three most-addressed accounts, accounts ranked 4 to 10, and random sets of three
accounts (ranks 11 to 100, or individual accounts) with all other tweets, using rarefied entropy and
permutation-based Jensen-Shannon divergence. Automated accounts are excluded.

Usage: python discourse_checks.py --data ../data --out ../results --spec main|strict
"""
import argparse, collections, glob, os
import numpy as np, pandas as pd
import curated_voice_pipeline as cvp

SEED = cvp.SEED_DEFAULT; B = 1000; BP = 200; NDRAW = 100; CAP = 3
TOP3 = {"NFT": ["opensea", "giverep", "hubdotxyz"], "DeFi": ["lombard_finance", "infinit_labs", "giverep"]}
R410 = {"NFT": ["spaace_io", "blur_io", "abstractchain", "base", "kaitoai", "magiceden", "anoma"],
        "DeFi": ["hubdotxyz", "kaitoai", "rhea_finance", "hananetwork", "attntoken", "0xsoulprotocol", "nearprotocol"]}
cvp.DOMAIN_STOPWORDS.update(cvp.NAME_TOKENS)

def name_tokens(hs):
    s = set()
    for h in hs:
        if h.startswith('u_'): continue
        s.add(h); s.update(p for p in h.replace('xyz', ' ').replace('_', ' ').split() if len(p) > 2)
    return s
def H(c):
    c = c[c > 0]; p = c / c.sum(); return float(-(p * np.log2(p)).sum())
def JSD(a, b):
    p = a / a.sum(); q = b / b.sum(); m = .5 * (p + q)
    kl = lambda x, y: float((x[x > 0] * np.log2(x[x > 0] / y[x > 0])).sum())
    return .5 * kl(p, m) + .5 * kl(q, m)
def cap(df, rng):
    df = df.iloc[rng.permutation(len(df))]
    return df[df.groupby('user').cumcount() < CAP]

def test(sub, hs, rng, B, strict):
    if strict: sub = sub[~sub.duplicate_record & ~sub.near_dup]
    m = sub.ms.map(lambda s: bool(s & set(hs))); t_df, r_df = sub[m], sub[~m]
    if strict: t_df, r_df = cap(t_df, rng), cap(r_df, rng)
    ex = name_tokens(hs)
    tier = [x for x in (cvp.preprocess(s, ex) for s in t_df.text) if x]
    reg = [x for x in (cvp.preprocess(s, ex) for s in r_df.text) if x]
    vocab = {}; f = lambda L: [np.array([vocab.setdefault(w, len(vocab)) for w in l]) for l in L]
    ti, ri = f(tier), f(reg); V = len(vocab)
    ct = np.bincount(np.concatenate(ti), minlength=V).astype(float); cr = np.bincount(np.concatenate(ri), minlength=V).astype(float)
    bud = ct.sum(); lens = np.array([len(x) for x in ri]); hn = np.empty(B); jn = np.empty(B)
    for b in range(B):
        o = rng.permutation(len(ri)); k = int(np.searchsorted(np.cumsum(lens[o]), bud) + 1)
        cp = np.bincount(np.concatenate([ri[i] for i in o[:k]]), minlength=V).astype(float)
        hn[b] = H(cp); jn[b] = JSD(cp, cr - cp)
    ht, hr, j = H(ct), H(cr), JSD(ct, cr)
    return dict(n_tweets=len(tier), n_tokens=int(bud), residual_reduction_pct=(hn.mean() - ht) / hr * 100,
                p_entropy=(1 + (hn <= ht).sum()) / (B + 1), jsd_excess=j - jn.mean(), p_jsd=(1 + (jn >= j).sum()) / (B + 1))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default='../data/v2'); ap.add_argument('--out', default='../results')
    ap.add_argument('--spec', default='main', choices=['main', 'strict']); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    strict = a.spec == 'strict'
    t = pd.concat([pd.read_csv(p) for c in ('NFT', 'DeFi') for p in sorted(glob.glob(f'{a.data}/tweets_{c}_part*.csv.gz'))], ignore_index=True)
    t = t[~t.automated_account].reset_index(drop=True)
    t['ms'] = t.mentions.fillna('').map(lambda s: set(x for x in s.split(';') if x))
    norm = t.text.fillna('').str.lower().str.replace(r'https?://\S+|@\w+|[^a-z\s]', ' ', regex=True).str.split().str.join(' ')
    t['near_dup'] = norm.duplicated(keep='first') & (norm.str.len() > 0)
    indiv = pd.read_csv(f'{a.data}/individual_accounts.csv')
    rows, plac = [], []
    for comm in ('NFT', 'DeFi'):
        rng = np.random.default_rng(SEED); sub = t[t.community == comm]
        cnt = collections.Counter(h for s in sub.ms for h in s)
        excl = set(TOP3[comm]) | set(R410[comm]) | {'freeexyz', 'grok'}
        pool = [h for h, _ in cnt.most_common(130) if h not in excl][:90]
        for lab, hs in (('Top 3', TOP3[comm]), ('Ranks 4-10', R410[comm])):
            d = test(sub, hs, rng, B, strict); d.update(spec=a.spec, community=comm, group=lab); rows.append(d); print(d, flush=True)
        for pname, pl in (('ranks 11-100', pool), ('individual accounts', indiv[indiv.community == comm].account.tolist())):
            for k in range(NDRAW):
                hs = list(rng.choice(pl, 3, replace=False))
                d = test(sub, hs, rng, BP, strict); d.update(spec=a.spec, community=comm, pool=pname, draw=k, accounts=', '.join(hs)); plac.append(d)
            print(a.spec, comm, pname, 'done', flush=True)
    pd.DataFrame(rows).to_csv(f'{a.out}/discourse_groups_{a.spec}.csv', index=False)
    pd.DataFrame(plac).to_csv(f'{a.out}/discourse_random_sets_{a.spec}.csv', index=False)

if __name__ == '__main__':
    main()
