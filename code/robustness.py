"""Regression of non-organic attention on rank (Supplementary Table S6) and robustness of
the main findings to misclassification of automated accounts (Supplementary Table S7).

Usage: python robustness.py --data ../data/v2 --out ../results
Requires: pandas, numpy, statsmodels. Runtime: a few minutes.
"""
import argparse, collections, os
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from attention_sources import load, INFOFI

BINS = [0, 10, 50, 200, 1000, np.inf]
LABELS = ['top10', 'r11_50', 'r51_200', 'r201_1000', 'rest']
SCENARIOS = (('observed', 0.05, 0.03), ('pessimistic', 0.18, 0.08))  # from the manual check, Supplementary S2


def mentions(t):
    rows = [(c, u, bool(au), h, (h in INFOFI) or bool((ms - {h}) & INFOFI))
            for c, u, au, ms in zip(t.community, t.user, t.automated_account, t.ms) for h in ms]
    m = pd.DataFrame(rows, columns=['community', 'user', 'auto', 'target', 'inc_ctx'])
    n = m.groupby(['community', 'target']).size().rename('n').reset_index()
    n['rank'] = n.groupby('community').n.rank(ascending=False, method='first')
    m = m.merge(n, on=['community', 'target'])
    m['bucket'] = pd.Categorical(pd.cut(m['rank'], BINS, labels=LABELS), categories=LABELS[::-1])  # reference: rest
    m['nonorg'] = (m.auto | m.inc_ctx).astype(int)
    return m


def model(m):
    out = []
    for spec, d in (('all', m), ('excl_reward_platforms', m[~m.target.isin(INFOFI)])):
        for c in ('NFT', 'DeFi'):
            s = d[d.community == c]
            r = smf.logit('nonorg ~ C(bucket)', data=s).fit(disp=0, cov_type='cluster',
                                                          cov_kwds={'groups': pd.factorize(s.user)[0]})
            ci = r.conf_int()
            for k in r.params.index.drop('Intercept'):
                out.append(dict(spec=spec, community=c, term=k.split('T.')[-1].rstrip(']'), OR=np.exp(r.params[k]),
                                lo=np.exp(ci.loc[k, 0]), hi=np.exp(ci.loc[k, 1]), p=r.pvalues[k], n=len(s)))
    return pd.DataFrame(out)


def simulate(t, m, B, seed):
    users = t[['community', 'user', 'automated_account']].drop_duplicates(['community', 'user'])
    idx = {k: i for i, k in enumerate(zip(users.community, users.user))}
    lab0 = users.automated_account.to_numpy().astype(bool)
    mi = np.array([idx[k] for k in zip(m.community, m.user)])
    inc = m.inc_ctx.to_numpy(); tgt = m.target.to_numpy(); com = m.community.to_numpy()
    rng = np.random.default_rng(seed); sims = []
    for scen, p_fp, p_fn in SCENARIOS:
        for it in range(B):
            lab = lab0 ^ np.where(lab0, rng.random(len(lab0)) < p_fp, rng.random(len(lab0)) < p_fn)
            org = ~lab[mi] & ~inc
            for c in ('NFT', 'DeFi'):
                sel = com == c; tg = tgt[sel]
                top_raw = [h for h, _ in collections.Counter(tg).most_common(10)]
                oc = pd.Series(org[sel], index=tg).groupby(level=0).sum().sort_values(ascending=False, kind='stable')
                top_org = set(oc.index[:10]); intop = np.isin(tg, top_raw)
                sims.append(dict(scenario=scen, it=it, community=c, nonorg_all=(~org[sel]).mean() * 100,
                                 nonorg_top10=(~org[sel][intop]).mean() * 100,
                                 dropped=sum(h not in top_org for h in top_raw),
                                 freeexyz_out='freeexyz' not in top_org, giverep_out='giverep' not in top_org))
    return pd.DataFrame(sims)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default='../data/v2'); ap.add_argument('--out', default='../results')
    ap.add_argument('--B', type=int, default=500); ap.add_argument('--seed', type=int, default=20260821)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    t = load(a.data); m = mentions(t)
    res = model(m); res.to_csv(f'{a.out}/model_rank.csv', index=False); print(res.round(3).to_string(index=False))
    S = simulate(t, m, a.B, a.seed); S.to_csv(f'{a.out}/misclassification_sims.csv', index=False)
    summ = S.groupby(['scenario', 'community']).agg(
        nonorg_all=('nonorg_all', 'mean'), top10_lo=('nonorg_top10', lambda x: x.quantile(.025)),
        top10_hi=('nonorg_top10', lambda x: x.quantile(.975)), dropped_mode=('dropped', lambda x: x.mode()[0]),
        all_four_defi_out=('dropped', lambda x: (x >= 4).mean()), giverep_out=('giverep_out', 'mean'),
        freeexyz_out=('freeexyz_out', 'mean'))
    summ.to_csv(f'{a.out}/misclassification_summary.csv'); print(summ.round(3).to_string())


if __name__ == '__main__':
    main()
