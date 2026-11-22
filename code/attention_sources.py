"""Sources of attention (Tables 1 to 3, Figure 1, Supplementary Table S3).

Usage: python attention_sources.py --data ../data --out ../results
"""
import argparse, collections, glob, os, re
import numpy as np, pandas as pd
from curated_voice_pipeline import gini

# Mention-reward (InfoFi) platforms, Supplementary Material S3
INFOFI = {'giverep', 'kaitoai', 'cookiedotfun', 'wallchain', 'wallchain_xyz', 'xeetdotai', 'galxe', 'layer3xyz'}
KW = re.compile(r'\b(?:yap|yaps|yapping|yapper|rep points|snaps|quacks|mindshare|leaderboard|points|airdrop|farming)\b')

def load(d):
    t = pd.concat([pd.read_csv(p) for c in ('NFT', 'DeFi') for p in sorted(glob.glob(f'{d}/tweets_{c}_part*.csv.gz'))], ignore_index=True)
    t['ms'] = t.mentions.fillna('').map(lambda s: set(x for x in s.split(';') if x))
    t['kw'] = t.text.fillna('').str.lower().str.contains(KW)
    return t

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default='../data/v2'); ap.add_argument('--out', default='../results')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t = load(a.data)
    acc, comp, rank = [], [], []
    for c in ('NFT', 'DeFi'):
        s = t[t.community == c]
        raw, org, aut, inc = (collections.Counter() for _ in range(4))
        for m, au in zip(s.ms, s.automated_account):
            for h in m:
                raw[h] += 1
                if au: aut[h] += 1
                elif h in INFOFI or (m - {h}) & INFOFI: inc[h] += 1
                else: org[h] += 1
        T = sum(raw.values()); top = [h for h, _ in raw.most_common(10)]
        org_order = [h for h, _ in org.most_common()]
        for i, h in enumerate(top, 1):
            a_ = s[s.ms.map(lambda m: h in m)]; au = a_.automated_account
            oth = a_.ms.map(lambda m: bool((m - {h}) & INFOFI))
            incm = (~au) & ((h in INFOFI) | oth); incb = (~au) & ((h in INFOFI) | oth | a_.kw)
            acc.append(dict(community=c, rank=i, account=h, tweets=len(a_), automated_pct=au.mean() * 100,
                            incentivized_pct=incm.mean() * 100, organic_pct=((~au) & ~incm).mean() * 100,
                            incentivized_broad_pct=incb.mean() * 100, organic_broad_pct=((~au) & ~incb).mean() * 100,
                            organic_mentions=org[h], organic_rank=(org_order.index(h) + 1) if org[h] else None))
        tt = sum(raw[h] for h in top)
        human = s[~s.automated_account]
        comp.append(dict(community=c, mentions=T, accounts_addressed=len(raw),
                         automated_pct=sum(aut.values()) / T * 100, incentivized_pct=sum(inc.values()) / T * 100,
                         organic_pct=sum(org.values()) / T * 100,
                         top10_automated_pct=sum(aut[h] for h in top) / tt * 100, top10_incentivized_pct=sum(inc[h] for h in top) / tt * 100,
                         top10_organic_pct=sum(org[h] for h in top) / tt * 100, top10_share_of_mentions=tt / T * 100,
                         human_tweets_tagging_reward_platform_pct=human.ms.map(lambda m: bool(m & INFOFI)).mean() * 100,
                         top10_leaving_when_organic=len([h for h in top if h not in org_order[:10]]),
                         gini_mentions_all=gini(np.array(list(raw.values()))), gini_mentions_organic=gini(np.array([org[h] for h in raw]))))
        for h in org_order[:10]: rank.append(dict(community=c, organic_rank=org_order.index(h) + 1, account=h, organic_mentions=org[h], in_raw_top10=h in top))
    pd.DataFrame(acc).to_csv(f'{a.out}/attention_sources_top10.csv', index=False)
    pd.DataFrame(comp).to_csv(f'{a.out}/attention_composition.csv', index=False)
    pd.DataFrame(rank).to_csv(f'{a.out}/organic_top10.csv', index=False)
    print(pd.DataFrame(acc).round(1).to_string(index=False)); print(pd.DataFrame(comp).round(1).T.to_string())

if __name__ == '__main__':
    main()
