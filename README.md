# Curated Voice: Communicative Inequality, Platform Power, and Automation in NFT and DeFi Communities on X

Data and code accompanying the manuscript submitted to *The Information Society*.

Authors: Andry Alamsyah, Wynona Jocelin Juono, and Dian Puteri Ramadhani, School of Economics and Business, Telkom University, Bandung, Indonesia.

## Contents

```
data/
  tweets_NFT.csv.gz    one row per tweet record, NFT community (January to December 2025)
  tweets_DeFi.csv.gz   one row per tweet record, DeFi community (January to December 2025)
  edges.csv.gz         weighted directed interaction edges (retweets, mentions, replies)
code/
  curated_voice_pipeline.py   reproduces Tables 2, 3 and S5 to S8
```

## Data

The raw data were obtained from X through a third-party provider and cannot be redistributed in full under the platform's terms of service. The files here are derived from them:

* Usernames and @mentions are replaced by salted one-way hashes (`u_` followed by 12 hexadecimal characters). The salt is not stored, so hashes cannot be reversed. The accounts listed in Supplementary Table S4, together with @Freeexyz and @grok, keep their handles because they are organizational or automated accounts whose structural position is itself analyzed.
* `tweets_<community>.csv.gz` columns: `community`, `user`, `created_at`, `text` (tweet text with mentions hashed), `automated_account` (account classified as automated, see Supplementary S2), `duplicate_record` (same account, text, and timestamp as an earlier record), `mentions_<handle>` (tweet mentions that central account), `authored_by_central` (handle if the tweet was posted by a central account).
* `edges.csv.gz` columns: `community`, `source`, `target`, `weight` (number of interactions), `source_automated`.

## Reproducing the results

```
pip install pandas numpy matplotlib
cd code
python curated_voice_pipeline.py --data ../data --out ../results
```

The script runs five specifications (main, names retained, NFT top two, automated accounts retained, duplicates removed) and the Gini analysis with and without automated accounts, using 1,000 resamples and a fixed random seed (20260821). Results are written as CSV files per specification. A full run takes roughly 30 to 40 minutes on a laptop; use `--B 200` for a quick check or `--specs main` for the main specification only.

## Citation

Please cite the article once published. Until then, cite this repository.

## License

Data and code are released under CC BY 4.0.
