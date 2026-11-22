# Bought, automated or earned? Separating the sources of attention in Web3 communities on X

Data and code accompanying the manuscript submitted to *Online Information Review*.

Authors: Andry Alamsyah, Wynona Jocelin Juono, and Dian Puteri Ramadhani, School of Economics and Business, Telkom University, Bandung, Indonesia.

## Contents

```
data/
  v2/tweets_NFT_part*.csv.gz   one row per tweet record, NFT community (2025), split into parts
  v2/tweets_DeFi_part*.csv.gz  one row per tweet record, DeFi community (2025), split into parts
  v2/individual_accounts.csv   hashed IDs of the individual accounts used for random comparison sets (Supplementary S4)
  edges.csv.gz                 weighted directed interaction edges (replies, mentions, retweets)
  tweets_NFT.csv.gz            earlier release of the same tweets, in the format used by curated_voice_pipeline.py
  tweets_DeFi.csv.gz           earlier release of the same tweets, in the format used by curated_voice_pipeline.py
code/
  attention_sources.py       Tables 1 to 3, Figure 1 and Supplementary Table S3
  discourse_checks.py        Supplementary Table S4 (discourse addressed to top accounts vs comparison groups)
  figures.py                 Figures 2 to 9, plus the rank-group, monthly and wording analyses behind Figures 3, 7 and 8
  curated_voice_pipeline.py  Gini benchmarks (Table 1) and Supplementary Table S5; also provides shared functions
results/                     outputs of the scripts above, as reported in the manuscript (figures in results/figures)
```

## Data

The raw data were obtained from X through a third-party data service using the platform's official API and cannot be redistributed in full under the platform's terms of service. The files here are derived from them.

* Usernames and @mentions are replaced by salted one-way hashes (`u_` followed by 12 hexadecimal characters). The salt is not stored, so hashes cannot be reversed. Organizational, platform and automated accounts that are named in the article keep their handles, because their position in the network is itself analyzed. Individual accounts are always hashed.
* `v2/tweets_<community>_part*.csv.gz` columns (the parts are read together by the scripts): `community`, `user`, `created_at`, `text` (tweet text with mentions hashed), `mentions` (semicolon-separated accounts addressed by the tweet), `automated_account` (account classified as automated, see Supplementary S2), `duplicate_record` (same account, text and timestamp as an earlier record).
* `edges.csv.gz` columns: `community`, `source`, `target`, `weight`, `source_automated`. Hashes in this file and in the earlier-release tweet files come from a different salt and do not match the hashes in `v2/`; named accounts match across all files. The earlier-release tweet files also flag mentions of the central accounts (`mentions_<handle>`, `authored_by_central`).

## Reproducing the results

```
pip install pandas numpy matplotlib
cd code
python attention_sources.py --data ../data/v2 --out ../results
python discourse_checks.py --data ../data/v2 --out ../results --spec main
python discourse_checks.py --data ../data/v2 --out ../results --spec strict
python figures.py --data ../data/v2 --edges ../data/edges.csv.gz --out ../results
python curated_voice_pipeline.py --data ../data --out ../results/pipeline
```

Run the scripts in this order, because `figures.py` reads the outputs of the first two. `attention_sources.py` and `figures.py` each run in under a few minutes. Each run of `discourse_checks.py` takes roughly 20 to 30 minutes; `curated_voice_pipeline.py` takes 30 to 40 minutes (use `--B 200` for a quick check). All resampling uses a fixed random seed (20260821).

## Definitions

* An account is **addressed** by a tweet when its handle appears in the tweet (replies, mentions and retweets).
* **Automated attention**: tweets by accounts classified as automated.
* **Incentivized attention**: human tweets addressed to a mention-reward platform (Kaito, Cookie, GiveRep, Wallchain, Xeet, Galxe, Layer3), or tagging one.
* **Organic attention**: all other human tweets.

## Citation

Please cite the article once published. Until then, cite this repository (see `CITATION.cff`).

## License

Data and code are released under CC BY 4.0.
