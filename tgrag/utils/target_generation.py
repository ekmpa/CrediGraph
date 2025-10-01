import csv
import gzip
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import tldextract
from matching import flip_if_needed, lookup_exact
from tqdm import tqdm

from tgrag.utils.load_labels import get_full_dict

_extract = tldextract.TLDExtract(include_psl_private_domains=True)
MAX_DOMAINS_TO_SHOW = 50
SAMPLES_PER_DOMAIN = 5


def strict_exact_etld1_match(
    raw_domain: str, rated_domains: Dict[str, List[float]]
) -> Optional[str]:
    """Accept only if rotating labels yields EXACTLY eTLD+1 (no subdomain) that exists in rated_domains.
    Examples matching: 'news.cn' -> 'news.cn'; 'co.uk.theregister' -> 'theregister.co.uk'
    Examples rejected: 'cn.360.news' (subdomain present), 'com.10...go' (subdomain present).
    """
    labels = [p for p in raw_domain.strip('.').lower().split('.') if p]

    n = len(labels)
    for r in range(n):
        rotated = labels[-r:] + labels[:-r] if r else labels
        rotated_str = '.'.join(rotated)
        ext = _extract(rotated_str)
        if not ext.suffix or not ext.domain:
            continue
        if ext.subdomain:  # exact only
            continue
        etld1 = f'{ext.domain}.{ext.suffix}'
        if (
            rotated_str == etld1 and etld1 in rated_domains
        ):  # rotated string itself should exactly equal eTLD+1
            return etld1
    return None


def generate_exact_targets(
    vertices_gz: str, targets_csv_out: str, dqr_domains: Dict[str, List[float]]
) -> None:
    """Generate targets.csv with exact domain matches."""
    chosen: Dict[str, Tuple[int, List[float]]] = {}  # domain -> (nid, metrics)
    total_lines = 0
    rejected = 0

    with gzip.open(vertices_gz, 'rt', encoding='utf-8') as f:
        _ = f.readline()
        for line in f:
            total_lines += 1
            parts = line.rstrip('\n').split(',')
            if len(parts) < 2:
                continue
            try:
                nid = int(parts[0].strip())
            except Exception:
                continue
            raw_domain = parts[1].strip()

            etld1 = strict_exact_etld1_match(raw_domain, dqr_domains)
            if etld1 is None:
                rejected += 1
                continue

            metrics = dqr_domains[etld1]
            if etld1 not in chosen:
                chosen[etld1] = (nid, metrics)

    with open(targets_csv_out, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                'nid',
                'pc1',
                'afm',
                'afm_bias',
                'afm_min',
                'afm_rely',
                'fc',
                'mbfc',
                'mbfc_bias',
                'mbfc_fact',
                'mbfc_min',
                'lewandowsky_acc',
                'lewandowsky_trans',
                'lewandowsky_rely',
                'lewandowsky_mean',
                'lewandowsky_min',
                'misinfome_bin',
            ]
        )
        for dom, (nid, values) in chosen.items():
            writer.writerow([nid, *values])

    print(f'[INFO] Generation done. Processed {total_lines:,} vertex rows.')
    print(
        f'[INFO] Wrote {len(chosen):,} exact-domain targets to {targets_csv_out}, rejected {rejected:,} nodes.'
    )


def generate_exact_targets_csv(
    node_file: str, targets_csv_out: str, dqr_domains: Dict[str, List[float]]
) -> None:
    """Generate targets.csv with exact domain matches."""
    chosen: Dict[str, List[float]] = {}  # domain -> (domain, metrics)
    total_lines = 0
    rejected = 0

    with open(node_file, 'rt', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc='Reading domains'):
            total_lines += 1
            raw_domain = row.get('domain', '').strip()
            if not raw_domain:
                continue

            etld1 = strict_exact_etld1_match(raw_domain, dqr_domains)
            if etld1 is None:
                rejected += 1
                continue

            metrics = dqr_domains[etld1]
            if etld1 not in chosen:
                chosen[etld1] = metrics

    with open(targets_csv_out, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                'domain',
                'pc1',
                'afm',
                'afm_bias',
                'afm_min',
                'afm_rely',
                'fc',
                'mbfc',
                'mbfc_bias',
                'mbfc_fact',
                'mbfc_min',
                'lewandowsky_acc',
                'lewandowsky_trans',
                'lewandowsky_rely',
                'lewandowsky_mean',
                'lewandowsky_min',
                'misinfome_bin',
            ]
        )
        for domain, values in tqdm(chosen.items(), desc='Writing target features'):
            writer.writerow([domain, *values])

    print(f'[INFO] Generation done. Processed {total_lines:,} vertex rows.')
    print(
        f'[INFO] Wrote {len(chosen):,} exact-domain targets to {targets_csv_out}, rejected {rejected:,} nodes.'
    )


def load_target_nids(path: str) -> set[int]:
    nids = set()
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        if r.fieldnames:
            r.fieldnames = [
                fn.strip().lstrip('\ufeff') if fn else fn for fn in r.fieldnames
            ]
        nid_key = 'nid'

        for row in r:
            nid_str = row.get(nid_key, '').strip()
            nids.add(int(nid_str))

    return nids


def generate(vertices_gz: str, targets_csv: str) -> None:
    dqr = get_full_dict()

    # GENERATION
    print(f'[INFO] Loaded {len(dqr):,} rated domains (DQR).')
    print(f'[INFO] Generating targets at: {targets_csv}')
    generate_exact_targets(vertices_gz, targets_csv, dqr)

    # ANALYSIS
    target_nids = load_target_nids(targets_csv)
    nid_to_raw_domain: Dict[int, str] = {}  # map target nids -> raw vertex domain

    with gzip.open(vertices_gz, mode='rt', encoding='utf-8') as vf:
        reader = csv.DictReader(vf)
        for row in reader:
            nid = int(row['nid'])

            if nid in target_nids and nid not in nid_to_raw_domain:
                raw_domain = (row.get('domain') or '').strip()
                nid_to_raw_domain[nid] = raw_domain
                if len(nid_to_raw_domain) == len(target_nids):
                    break

    # normalize and exact-match to DQR
    domain_to_nids: Dict[str, List[int]] = defaultdict(list)
    domain_to_samples: Dict[str, List[tuple]] = defaultdict(
        list
    )  # domain -> list of (nid, raw_domain, normalized)
    parse_fail = 0
    unrated = 0

    for nid, raw_dom in nid_to_raw_domain.items():
        norm = flip_if_needed(raw_dom) if raw_dom else None
        if not norm or '.' not in norm:
            parse_fail += 1
            continue
        if lookup_exact(norm, dqr) is not None:
            domain_to_nids[norm].append(nid)
            if len(domain_to_samples[norm]) < SAMPLES_PER_DOMAIN:
                domain_to_samples[norm].append((nid, raw_dom, norm))
        else:
            unrated += 1

    matched_targets = sum(len(v) for v in domain_to_nids.values())
    distinct_rated_domains_hit = len(domain_to_nids)

    print(
        f'[ANALYSIS] Distinct rated domains hit by targets:           {distinct_rated_domains_hit:,} / {len(dqr):,}'
    )

    if domain_to_nids:
        counts = Counter({d: len(nids) for d, nids in domain_to_nids.items()})
        max_dom, max_cnt = counts.most_common(1)[0]
        avg = matched_targets / distinct_rated_domains_hit
        print('\nTargets/domain distribution:')
        print(f'- Max targets landing on one rated domain:       {max_cnt}')
        print(f'- Avg targets per rated domain (hit>=1):         {avg:.2f}')

        # multi-hit domains to sanity check
        multi = [(d, c) for d, c in counts.most_common() if c > 1]
        print(f'- Rated domains with >1 target nid:                {len(multi):,}')
        for idx, (d, c) in enumerate(multi[:MAX_DOMAINS_TO_SHOW], start=1):
            print(f'  {idx:>3}. {d}: {c} targets')
            for nid, raw_dom, norm in domain_to_samples[d]:
                print(
                    f"       - nid={nid}  raw_domain='{raw_dom}'  normalized='{norm}'"
                )

    else:
        print('\n[ERR] No targets matched any rated domain.')
