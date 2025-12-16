## Data

The `labels.py` script lists all datasets used in this work and their features as per their documentation. Note that size may change from what is indicated there in the processed files we end up using due to the processing (e.g, if a blacklist is URL-based, we average the scoring for all URLs that resolve to the same domain).

Below is some more information about the processing of the data we use as labels.

## `DQR`

We use this data as it is, as our ground truth supervised labels.

## Weak labels

### LegitPhish

[LegitPhish](https://data.mendeley.com/datasets/hx4m73v2sf/2) is a URL-based datasets with binary labels (0 = Phishing, 1 = Legitimate). The data is processed by converting URLs to their domain name, averaging the scores of all URLs that resolve to the same domain, and standardizing the header (we always use `domain` and `label` in the processed csvs.)

```bash
0: 26957
1: 37113
```

### Wikipedia

[Wikipedia](https://github.com/kynoptic/wikipedia-reliable-sources) is an aggregate set of a reliability ratings from multiple Wikipedia sources. The data is processed to keep only 'boosted' and 'discarded' domains (corresponding to 1 and 0 respectively -- they also have a neutral category we discard).

```bash
0: 1029
1: 2906
```
