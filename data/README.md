## Data

The `labels.py` script lists all datasets used in this work and their features as per their documentation. Note that size may change from what is indicated there in the processed files we end up using due to the processing (e.g, if a blacklist is URL-based, we average the scoring for all URLs that resolve to the same domain).

Below is some more information about the processing of the data we use as labels.

## `DQR`

We use this data as it is, as our ground truth supervised labels.

Other regression labels we don't use yet:

- Zoznam from [Konspiratori](https://konspiratori.sk/zoznam-stranok)
- [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) which gives language scores.
- The [Iffy Index](https://iffy.news/index/)
- [Consensus Credibility](https://science.feedback.org/consensus-credibility-scores-comprehensive-dataset-web-domains-credibility) pending access
- [Open Feedback](https://open.feedback.org/) pending request

## Weak labels

Standardized into: 0 = Phishing, 1 = Legitimate, with headers `domain` and `label`. The label counts shown below follow this convention.

### LegitPhish

[LegitPhish](https://data.mendeley.com/datasets/hx4m73v2sf/2) is a URL-based dataset with binary labels (0 = Phishing, 1 = Legitimate). The data is processed by converting URLs to their domain name, averaging the scores of all URLs that resolve to the same domain, and standardizing the header (we always use `domain` and `label` in the processed csvs.)

```bash
0: 26957
1: 37113
```

### PhishDataset

[PhishDataset](https://github.com/ESDAUNG/PhishDataset/blob/main/data_imbal%20-%2055000.xlsx) is a URL-based dataset with binary labels (0 = Legitimate, 1 = Phishing)

```bash
0: 3730
1: 40535
```

### Nelez

[Nelež](https://www.nelez.cz) is a blacklist of misinformation websites from a Czech organisation of the same name.

```bash
0: 51
1: 0
```

### Wikipedia

[Wikipedia](https://github.com/kynoptic/wikipedia-reliable-sources) is an aggregate set of a reliability ratings from multiple Wikipedia sources. The data is processed to keep only 'boosted' and 'discarded' domains (corresponding to 1 and 0 respectively -- they also have a neutral category we discard).

```bash
0: 1029
1: 2906
```

### URL-Phish

[URL-Phish](https://data.mendeley.com/datasets/65z9twcx3r/1) is a feature-engineered dataset for phishing detection. URLs have label 0 if they are considered benign, 1 for phishing.

```bash
0: 10551
1: 92460
```

### Phish & Legit

[Phish & Legit](https://www.kaggle.com/datasets/harisudhan411/phishing-and-legitimate-urls?resource=download) is a URL Classification dataset of suspicous (0) and genuine (1) web addresses.

```bash
0: 292163
1: 146316
```

### Misinformation domains

[Misinfo-domains](https://github.com/JanaLasser/misinformation_domains/tree/main) is a collection of domains labelled unreliable if they are assessed as spreaders of unreliable information.

```bash
0: 2170
1: 2597
```
