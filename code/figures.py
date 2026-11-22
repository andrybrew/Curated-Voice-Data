"""Additional analyses and Figures 2 to 9 of the manuscript.

Writes rank_buckets.csv (Figure 3), monthly.csv (Figure 7) and words_incentivized_vs_organic.csv (Figure 8)
to the results folder, and all figures to results/figures. Run discourse_checks.py --spec main first (Figure 9).

Usage: python figures.py --data ../data/v2 --edges ../data/edges.csv.gz --out ../results
"""
import argparse, collections, glob, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import curated_voice_pipeline as cvp
from attention_sources import INFOFI

AUT, INC, ORG, RED, GREY = '#1f3b73', '#d98a1f', '#bdbdbd', '#a8202b', '#555555'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

def source(m, h, au):
    return 'auto' if au else ('inc' if (h in INFOFI or (m - {h}) & INFOFI) else 'org')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default='../data/v2'); ap.add_argument('--edges', default='../data/edges.csv.gz')
    ap.add_argument('--out', default='../results'); a = ap.parse_args(); F = os.path.join(a.out, 'figures'); os.makedirs(F, exist_ok=True)
    save = lambda fig, n: (fig.savefig(f'{F}/{n}.png', dpi=200, bbox_inches='tight'), plt.close(fig))
    t = pd.concat([pd.read_csv(p) for c in ('NFT', 'DeFi') for p in sorted(glob.glob(f'{a.data}/tweets_{c}_part*.csv.gz'))], ignore_index=True)
    t['ms'] = t.mentions.fillna('').map(lambda s: set(x for x in s.split(';') if x))

    # ---- Figure 2: Lorenz curves (automated accounts removed)
    e = pd.read_csv(a.edges); e = e[~e.source_automated]; rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for c, ls in (('NFT', '-'), ('DeFi', '--')):
        s = e[e.community == c].groupby(['source', 'target'], as_index=False).weight.sum()
        nodes = pd.unique(pd.concat([s.source, s.target])); idx = {u: i for i, u in enumerate(nodes)}; N = len(nodes)
        dg = np.zeros(N); np.add.at(dg, s.source.map(idx), s.weight); np.add.at(dg, s.target.map(idx), s.weight)
        xs = np.arange(1, N + 1) / N; v = np.sort(dg); ax.plot(xs, np.cumsum(v) / v.sum(), ls, color=AUT, lw=1.8, label=f'{c} (Gini {cvp.gini(dg):.3f})')
        d2 = np.zeros(N); ww = rng.permutation(s.weight.values); np.add.at(d2, rng.integers(0, N, len(s)), ww); np.add.at(d2, rng.integers(0, N, len(s)), ww)
        v2 = np.sort(d2); ax.plot(xs, np.cumsum(v2) / v2.sum(), ls, color=ORG, lw=1.5, label=f'{c} random benchmark (Gini {cvp.gini(d2):.3f})')
    ax.plot([0, 1], [0, 1], color='black', lw=0.6); ax.set_xlabel('Cumulative share of accounts'); ax.set_ylabel('Cumulative share of weighted degree'); ax.legend(frameon=False, fontsize=7)
    save(fig, 'Figure2_Lorenz_curves')

    # ---- per-account source counts
    raw, src = {}, {}
    for c in ('NFT', 'DeFi'):
        s = t[t.community == c]; r = collections.Counter(); sc = collections.defaultdict(collections.Counter)
        for m, au in zip(s.ms, s.automated_account):
            for h in m: r[h] += 1; sc[h][source(m, h, au)] += 1
        raw[c], src[c] = r, sc

    # ---- Figure 3: non-organic share by rank group
    rows = []
    for c in ('NFT', 'DeFi'):
        order = [h for h, _ in raw[c].most_common()]
        for lo, hi, lab in ((1, 10, 'Top 10'), (11, 50, '11-50'), (51, 200, '51-200'), (201, 1000, '201-1,000'), (1001, len(order), 'Rest')):
            hs = order[lo - 1:hi]; tot = sum(raw[c][h] for h in hs); au = sum(src[c][h]['auto'] for h in hs); inc = sum(src[c][h]['inc'] for h in hs)
            rows.append(dict(community=c, group=lab, mentions=tot, automated_pct=au / tot * 100, incentivized_pct=inc / tot * 100, nonorganic_pct=(au + inc) / tot * 100))
    b = pd.DataFrame(rows); b.to_csv(f'{a.out}/rank_buckets.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, c in zip(axes, ('NFT', 'DeFi')):
        d = b[b.community == c].reset_index(drop=True); x = np.arange(len(d))
        ax.bar(x, d.automated_pct, color=AUT, width=0.65, label='Automated'); ax.bar(x, d.incentivized_pct, bottom=d.automated_pct, color=INC, width=0.65, label='Incentivized')
        for i, v in enumerate(d.nonorganic_pct): ax.text(i, v + 0.8, f'{v:.0f}%', ha='center', fontsize=8.5)
        ax.set_xticks(x, d.group); ax.set_title(c, loc='left', weight='bold'); ax.set_xlabel('Accounts grouped by rank of attention received')
    axes[0].set_ylabel('Non-organic share of attention (%)'); axes[0].legend(frameon=False); axes[0].set_ylim(0, 42)
    save(fig, 'Figure3_Nonorganic_by_rank')

    # ---- Figure 4: sources of attention to the ten most-addressed accounts
    r = pd.read_csv(f'{a.out}/attention_sources_top10.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, c in zip(axes, ('NFT', 'DeFi')):
        d = r[r.community == c].iloc[::-1]; y = [f'@{h}' for h in d.account]; left = np.zeros(len(d))
        for k, col, lab in (('automated_pct', AUT, 'Automated'), ('incentivized_pct', INC, 'Incentivized'), ('organic_pct', ORG, 'Organic')):
            ax.barh(y, d[k], left=left, color=col, label=lab); left += d[k].values
        ax.set_xlim(0, 100); ax.set_title(f'{c}: ten most-addressed accounts', loc='left'); ax.set_xlabel('Share of tweets addressing the account (%)')
    axes[0].legend(frameon=False, loc='lower center', bbox_to_anchor=(1.05, -0.3), ncol=3)
    save(fig, 'Figure4_Attention_sources_top10')

    # ---- Figure 5: who addresses @freeexyz and @blur_io
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.7))
    for ax, h, title in ((axes[0], 'freeexyz', '@freeexyz (NFT rank 3)'), (axes[1], 'blur_io', '@blur_io (NFT rank 7)')):
        s = t[(t.community == 'NFT') & t.ms.map(lambda m: h in m)]
        g = s.groupby('user').agg(n=('text', 'size'), auto=('automated_account', 'max'), inc=('ms', lambda x: any(source(m, h, False) == 'inc' for m in x)))
        g['col'] = np.where(g.auto, AUT, np.where(g.inc, INC, ORG)); rr = np.random.default_rng(3); n = len(g)
        ang = rr.uniform(0, 2 * np.pi, n); rad = 1 + rr.uniform(0, 0.35, n)
        for an, ra, cl, w in zip(ang, rad, g.col, g.n): ax.plot([0, ra * np.cos(an)], [0, ra * np.sin(an)], color=cl, lw=0.3 + 0.15 * min(w, 20), alpha=0.35 if cl == ORG else 0.6)
        ax.scatter(rad * np.cos(ang), rad * np.sin(ang), s=4 + 3 * np.minimum(g.n, 40), c=g.col, lw=0, zorder=3); ax.scatter([0], [0], s=300, c='black', zorder=4)
        ax.set_aspect('equal'); ax.axis('off'); ax.set_title(f'{title}\n{len(s):,} tweets from {n:,} accounts; {s.automated_account.mean() * 100:.1f}% automated')
    axes[0].legend(handles=[Line2D([0], [0], color=k, lw=3, label=l) for k, l in ((AUT, 'Automated'), (INC, 'Incentivized'), (ORG, 'Organic'))], frameon=False, loc='lower center', bbox_to_anchor=(1.05, -0.12), ncol=3)
    save(fig, 'Figure5_Freeexyz_vs_blur')

    # ---- Figure 6: ranking by all attention vs organic attention only
    org_rank = pd.read_csv(f'{a.out}/organic_top10.csv')
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))
    for ax, c in zip(axes, ('NFT', 'DeFi')):
        d = r[r.community == c]; o = org_rank[org_rank.community == c]; ax.set_xlim(-0.6, 1.75); ax.set_ylim(11.9, 0.3); ax.axis('off'); ax.set_title(c, loc='left', weight='bold')
        for _, x in d.iterrows():
            y1 = x.organic_rank if (x.organic_rank == x.organic_rank and x.organic_rank <= 10) else 11.4; col = RED if y1 == 11.4 else GREY
            ax.plot([0.05, 0.95], [x['rank'], y1], color=col); ax.text(-0.02, x['rank'], f"{x['rank']}. @{x.account}", ha='right', va='center', fontsize=7.6, color=col if y1 == 11.4 else 'black')
        for _, x in o.iterrows():
            new = not x.in_raw_top10; lab = f'@{x.account}' if not x.account.startswith('u_') else 'individual account'
            ax.text(1.02, x.organic_rank, f'{x.organic_rank}. {lab}', ha='left', va='center', fontsize=7.6, color='#2a7a3b' if new else 'black')
    save(fig, 'Figure6_Ranking_change')

    # ---- Figure 7: monthly shares
    t['month'] = pd.to_datetime(t.created_at, errors='coerce', format='mixed', utc=True).dt.month
    hmn = t[~t.automated_account].assign(inc=lambda x: x.ms.map(lambda m: bool(m & INFOFI)))
    mon = pd.concat({'incentivized_pct_of_human': hmn.groupby(['month', 'community']).inc.mean().unstack() * 100,
                     'automated_pct': t.groupby(['month', 'community']).automated_account.mean().unstack() * 100}, axis=1)
    mon.to_csv(f'{a.out}/monthly.csv')
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for c, ls in (('NFT', '-'), ('DeFi', '--')):
        axes[0].plot(mon.index, mon[('incentivized_pct_of_human', c)], ls, color=INC, marker='o', ms=3, label=c)
        axes[1].plot(mon.index, mon[('automated_pct', c)], ls, color=AUT, marker='o', ms=3, label=c)
    for ax, tt, yl in ((axes[0], 'Human tweets tagging a reward platform, 2025', '% of human tweets'), (axes[1], 'Tweets from automated accounts, 2025', '% of all tweets')):
        ax.set_xticks(range(1, 13), MONTHS); ax.set_title(tt, loc='left'); ax.set_ylabel(yl); ax.legend(frameon=False)
    save(fig, 'Figure7_Monthly_trend')

    # ---- Figure 8: words in incentivized vs organic human tweets addressing the same non-reward accounts
    cvp.DOMAIN_STOPWORDS.update(cvp.NAME_TOKENS | {'giverep', 'kaito', 'kaitoai', 'cookie', 'cookiedotfun', 'wallchain', 'rep'})
    hh = hmn[hmn.ms.map(lambda m: bool(m - INFOFI))].copy(); hh['tok'] = hh.text.map(cvp.preprocess)
    A = collections.Counter(w for l in hh[hh.inc].tok for w in l); B = collections.Counter(w for l in hh[~hh.inc].tok for w in l)
    na, nb = sum(A.values()), sum(B.values()); rows = []
    for w in set(A) | set(B):
        p, q = A[w] / na, B[w] / nb; m = (p + q) / 2
        rows.append(dict(word=w, per1000_incentivized=p * 1000, per1000_organic=q * 1000, direction=np.sign(p - q),
                         jsd_contribution=0.5 * (p * np.log2(p / m) if p else 0) + 0.5 * (q * np.log2(q / m) if q else 0)))
    wd = pd.DataFrame(rows).sort_values('jsd_contribution', ascending=False); wd.to_csv(f'{a.out}/words_incentivized_vs_organic.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, dd, tt in ((axes[0], wd[wd.direction > 0].head(14).iloc[::-1], 'More frequent in incentivized tweets'), (axes[1], wd[wd.direction < 0].head(10).iloc[::-1], 'More frequent in organic tweets')):
        y = np.arange(len(dd)); ax.barh(y - 0.2, dd.per1000_incentivized, 0.4, color=INC, label='Incentivized'); ax.barh(y + 0.2, dd.per1000_organic, 0.4, color='#8c8c8c', label='Organic')
        ax.set_yticks(y, dd.word); ax.set_title(tt, loc='left'); ax.set_xlabel('Occurrences per 1,000 words')
    axes[0].legend(frameon=False, loc='lower right'); save(fig, 'Figure8_Incentivized_vs_organic_words')

    # ---- Figure 9: content versus volume
    pth = f'{a.out}/discourse_random_sets_main.csv'
    if os.path.exists(pth):
        p = pd.read_csv(pth); g = pd.read_csv(f'{a.out}/discourse_groups_main.csv')
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), gridspec_kw={'width_ratios': [1, 1, 0.8]}); vol = {}
        for ax, c in zip(axes[:2], ('NFT', 'DeFi')):
            d = p[(p.community == c) & (p.pool == 'ranks 11-100')]; top = g[(g.community == c) & (g.group == 'Top 3')].iloc[0]
            ax.hist(d.residual_reduction_pct, bins=18, color='#c9d4ea', edgecolor='#8fa4c8'); ax.axvline(top.residual_reduction_pct, color=RED, lw=2)
            ax.set_title(f'{c}: narrowing (content)', loc='left'); ax.set_xlabel('Reduction in lexical diversity (%)')
            n = (~t.automated_account & (t.community == c)).sum(); vol[c] = (top.n_tweets / n * 100, d.n_tweets.median() / n * 100)
        x = np.arange(2); axes[2].bar(x - 0.18, [vol[c][0] for c in ('NFT', 'DeFi')], 0.36, color=RED, label='Three most-addressed')
        axes[2].bar(x + 0.18, [vol[c][1] for c in ('NFT', 'DeFi')], 0.36, color='#8fa4c8', label='Three random (median)')
        axes[2].set_xticks(x, ['NFT', 'DeFi']); axes[2].set_title('Volume', loc='left'); axes[2].set_ylabel('Share of community tweets (%)'); axes[2].legend(frameon=False, fontsize=7.5)
        save(fig, 'Figure9_Content_vs_volume')
    print(b.round(1).to_string(index=False)); print(mon.round(2).to_string())

if __name__ == '__main__':
    main()
