# Project python 
# A rendre le 2 octobre, 

#!/usr/bin/env python3
"""
Greedy ontology-based summary using GOATOOLS and your tabular GO files

This script uses your three files:
  - geneOntology-BP-label.tsv.bz2               (GOID \t Label)
  - geneOntology-BP-hierarchy-direct.tsv.bz2   (ParentGO \t ChildGO)
  - associations.tsv                            (Element \t GOID)  -- can be uncompressed
  - elements_of_interest.txt                    (one element id per line)

What it does:
  - builds a minimal OBO from the label + direct-hierarchy tsv (temporary file)
  - loads it with goatools.GODag
  - runs GO enrichment (GOATOOLS) to get over-represented GO terms among EOI
  - performs the greedy summarization you described (score = IC * #new_EOI_covered)
  - prunes redundant terms
  - writes summary_terms.txt

Dependencies:
  pip install goatools pandas

Usage:
  python go_summary_greedy_goatools.py \
    --labels geneOntology-BP-label.tsv.bz2 \
    --hierarchy geneOntology-BP-hierarchy-direct.tsv.bz2 \
    --assoc associations.tsv \
    --eoi elements_of_interest.txt \
    --out summary_terms.txt

Notes:
  - If you have the indirect hierarchy (ancestor->descendant) we can use it to speed propagation.
  - The temporary OBO is minimal (id, name, is_a). GODag can parse it.
"""

import argparse
import math
import tempfile
import os
import sys
from collections import defaultdict, deque

import pandas as pd
from goatools.obo_parser import GODag
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS


def read_labels(path):
    return pd.read_csv(path, sep='\t', header=None, names=['GO', 'Label'], compression='bz2')


def read_hierarchy_direct(path):
    return pd.read_csv(path, sep='\t', header=None, names=['Parent', 'Child'], compression='bz2')


def read_assoc(path):
    # associations can be large; use pandas for robustness; support .bz2/.gz
    comp = None
    if path.endswith('.bz2'):
        comp = 'bz2'
    elif path.endswith('.gz'):
        comp = 'gzip'
    df = pd.read_csv(path, sep='\t', header=None, names=['Element', 'GO'], compression=comp, dtype=str)
    assoc = defaultdict(set)
    for e, g in zip(df['Element'], df['GO']):
        if pd.isna(e) or pd.isna(g):
            continue
        assoc[e].add(g)
    return dict(assoc)


def read_eoi(path):
    with open(path) as f:
        return {l.strip() for l in f if l.strip()}


def build_minimal_obo(labels_df, hier_df, tmp_obo_path):
    # Build mapping parent->children, and ensure all GO ids are present
    parents_map = defaultdict(list)
    for p, c in zip(hier_df['Parent'], hier_df['Child']):
        parents_map[c].append(p)
    # gather all GO ids
    go_ids = set(labels_df['GO']) | set(hier_df['Parent']) | set(hier_df['Child'])

    with open(tmp_obo_path, 'w') as fo:
        fo.write('format-version: 1.2\n')
        fo.write('data-version: generated-from-tsv\n')
        fo.write('\n')
        for go in sorted(go_ids):
            fo.write('[Term]\n')
            fo.write(f'id: {go}\n')
            lab = labels_df.loc[labels_df['GO'] == go, 'Label']
            if not lab.empty:
                fo.write(f'name: {lab.values[0]}\n')
            # write direct is_a lines
            for p in parents_map.get(go, []):
                fo.write(f'is_a: {p} !\n')
            fo.write('\n')


def descendants_of_godag(goid, godag):
    """Return set of descendant GO IDs (recursively) for a term in GODag."""
    if goid not in godag:
        return set()
    visited = set()
    q = deque()
    # TermObj.children may contain TermObj instances
    children = getattr(godag[goid], 'children', None)
    if not children:
        return set()
    # normalize to ids
    def child_ids(children):
        ids = []
        for c in children:
            if isinstance(c, str):
                ids.append(c)
            else:
                # TermObj
                ids.append(getattr(c, 'id', str(c)))
        return ids

    for c in child_ids(children):
        q.append(c)

    while q:
        cur = q.popleft()
        if cur in visited:
            continue
        visited.add(cur)
        if cur in godag:
            ch = getattr(godag[cur], 'children', None)
            if ch:
                for cid in child_ids(ch):
                    if cid not in visited:
                        q.append(cid)
    return visited


def compute_ic(term_to_elts, term, population_size):
    n = len(term_to_elts.get(term, set()))
    if n == 0:
        return 0.0
    p = n / population_size
    return -math.log2(p)


def greedy_summary(candidates, term_to_elts, godag, population_size, eoi_set):
    summary = []
    covered = set()
    candidates = set(candidates)
    desc_cache = {}

    def get_descs(t):
        if t not in desc_cache:
            desc_cache[t] = descendants_of_godag(t, godag)
        return desc_cache[t]

    while True:
        useful = []
        for c in candidates:
            covered_by_c = term_to_elts.get(c, set()) & eoi_set
            new_covered = covered_by_c - covered
            if new_covered:
                ic = compute_ic(term_to_elts, c, population_size)
                score = ic * len(new_covered)
                useful.append((score, c, len(new_covered), ic))
        if not useful:
            break
        useful.sort(key=lambda x: (-x[0], -x[2]))
        max_score = useful[0][0]
        winners = [c for (s, c, cov, ic) in useful if abs(s - max_score) < 1e-12]

        # Tie-break: if one winner is ancestor of another, keep most specific (descendant)
        winners_set = set(winners)
        to_remove_from_winners = set()
        for a in winners:
            for b in winners:
                if a == b:
                    continue
                if b in get_descs(a):
                    to_remove_from_winners.add(a)
        winners_final = [w for w in winners if w not in to_remove_from_winners]

        for w in winners_final:
            summary.append(w)
            covered |= (term_to_elts.get(w, set()) & eoi_set)
            # remove descendants from candidates
            descs = get_descs(w)
            candidates -= descs
        candidates -= set(winners_final)

        # remove candidates that no longer cover any uncovered EOI
        to_remove = set()
        for c in list(candidates):
            if len((term_to_elts.get(c, set()) & eoi_set) - covered) == 0:
                to_remove.add(c)
        candidates -= to_remove

    return summary


def get_ancestors_godag(goid, godag):
    if goid not in godag:
        return set()
    visited = set()
    q = deque()
    parents = getattr(godag[goid], 'parents', None)
    if parents:
        for p in parents:
            pid = p if isinstance(p, str) else getattr(p, 'id', None)
            if pid:
                q.append(pid)
    while q:
        cur = q.popleft()
        if cur in visited:
            continue
        visited.add(cur)
        if cur in godag:
            pp = getattr(godag[cur], 'parents', None)
            if pp:
                for p in pp:
                    pid = p if isinstance(p, str) else getattr(p, 'id', None)
                    if pid and pid not in visited:
                        q.append(pid)
    return visited


def prune_summary(summary, term_to_elts, godag, eoi_set):
    summary_set = set(summary)
    changed = True
    while changed:
        changed = False
        # Rule 1: if descendant in summary and ancestor annotates >=1 EOI not annotated by others, discard descendant
        for s in list(summary_set):
            ancs = get_ancestors_godag(s, godag) & summary_set
            for a in ancs:
                a_covered = term_to_elts.get(a, set()) & eoi_set
                others = set()
                for t in summary_set:
                    if t == a:
                        continue
                    others |= (term_to_elts.get(t, set()) & eoi_set)
                unique = a_covered - others
                if len(unique) >= 1 and s in summary_set:
                    summary_set.remove(s)
                    changed = True
        # Rule 2: if ancestor in summary and all its EOI are annotated by >=1 other summary annotation, discard ancestor
        for a in list(summary_set):
            a_covered = term_to_elts.get(a, set()) & eoi_set
            others = set()
            for t in summary_set:
                if t == a:
                    continue
                others |= (term_to_elts.get(t, set()) & eoi_set)
            if a_covered and a_covered.issubset(others):
                summary_set.remove(a)
                changed = True
    final = [t for t in summary if t in summary_set]
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', required=True, help='geneOntology-BP-label.tsv.bz2')
    parser.add_argument('--hierarchy', required=True, help='geneOntology-BP-hierarchy-direct.tsv.bz2')
    parser.add_argument('--assoc', required=True, help='associations.tsv (Element\tGO)')
    parser.add_argument('--eoi', required=True, help='elements_of_interest.txt')
    parser.add_argument('--out', default='summary_terms.txt')
    parser.add_argument('--pval', type=float, default=0.05)
    args = parser.parse_args()

    print('Reading labels...')
    labels_df = read_labels(args.labels)
    print('Reading hierarchy (direct)...')
    hier_df = read_hierarchy_direct(args.hierarchy)
    print('Reading associations...')
    assoc = read_assoc(args.assoc)
    print('Reading EOI...')
    eoi_set = read_eoi(args.eoi)

    # Build minimal OBO for GODag
    tmp_obo = tempfile.NamedTemporaryFile(delete=False, suffix='.obo')
    tmp_obo_path = tmp_obo.name
    tmp_obo.close()
    try:
        print('Building temporary OBO...')
        build_minimal_obo(labels_df, hier_df, tmp_obo_path)
        print('Loading GODag via GOATOOLS...')
        godag = GODag(tmp_obo_path)

        # invert assoc
        print('Inverting associations (term -> elements)')
        term_to_elts = defaultdict(set)
        for elt, gos in assoc.items():
            for g in gos:
                term_to_elts[g].add(elt)

        # run enrichment using GOATOOLS to get candidates
        pop = set(assoc.keys())
        print(f'Population size: {len(pop)}')
        # Build assoc map expected by GOATOOLS: element -> set(goids)
        go_assoc = {e: set(gs) for e, gs in assoc.items()}
        print('Running GOEA (GOATOOLS)...')
        goeaobj = GOEnrichmentStudyNS(list(pop), go_assoc, godag, methods=['fdr_bh'])
        goea_results = goeaobj.run_study([e for e in eoi_set if e in pop])
        sig = [r for r in goea_results if getattr(r, 'p_fdr_bh', 1.0) <= args.pval]
        candidates = set([r.GO for r in sig])
        print(f'Found {len(candidates)} significant terms (p_fdr_bh <= {args.pval})')

        # Propagate annotations upward: ancestors annotate all elements annotated to descendants
        print('Propagating annotations upward (term -> elements includes descendant annotations)...')
        all_terms = set(list(term_to_elts.keys()) + list(godag.keys()))
        propagated = {t: set(term_to_elts.get(t, set())) for t in all_terms}
        for t in all_terms:
            descs = descendants_of_godag(t, godag)
            for d in descs:
                propagated[t] |= term_to_elts.get(d, set())
        term_to_elts = propagated

        population_size = len(pop)
        print('Running greedy summarization...')
        summary = greedy_summary(candidates, term_to_elts, godag, population_size, eoi_set & pop)
        print(f'Selected {len(summary)} terms before pruning')

        print('Pruning summary...')
        summary_pruned = prune_summary(summary, term_to_elts, godag, eoi_set & pop)
        print(f'{len(summary_pruned)} terms after pruning')

        # write output
        rows = []
        for go in summary_pruned:
            name = godag[go].name if go in godag else ''
            n_all = len(term_to_elts.get(go, set()))
            n_eoi = len(term_to_elts.get(go, set()) & (eoi_set & pop))
            ic = compute_ic(term_to_elts, go, population_size)
            rows.append({'GO': go, 'Name': name, 'Annotated_all': n_all, 'Annotated_EOI': n_eoi, 'IC': ic})
        df = pd.DataFrame(rows)
        df = df.sort_values(by=['Annotated_EOI', 'IC'], ascending=[False, False])
        df.to_csv(args.out, sep='\t', index=False)
        print('Wrote summary to', args.out)

    finally:
        try:
            os.remove(tmp_obo_path)
        except Exception:
            pass


if __name__ == '__main__':
    main()
