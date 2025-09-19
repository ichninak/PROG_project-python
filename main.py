# Project python 
# A rendre le 2 octobre, 

#!/usr/bin/env python3
"""
Greedy ontology-based summary of over-represented annotations (using GOATOOLS)

Requirements:
  pip install goatools pandas

Inputs (examples):
  - go-basic.obo (ontology)
  - associations.tsv: two columns (element\tGOID) possibly multiple lines per element
  - elements_of_interest.txt: one element id per line

Usage:
  python go_summary_greedy.py \
      --obo go-basic.obo \
      --assoc associations.tsv \
      --eoi elements_of_interest.txt \
      --out summary_terms.txt

This script:
  - loads ontology with GOATOOLS
  - loads a background population (all elements found in assoc file)
  - performs enrichment (GOEA) to find over-represented GO terms
  - runs a greedy summarization algorithm described in your notes:
      choose a term maximizing score = IC(term) * (# new EOI covered)
    where IC = -log2(p) with p = |annotated_by(term)| / |population|
  - removes descendants of chosen terms from candidates
  - iterates until no candidate covers new elements
  - performs a small pruning pass to remove redundant terms in the summary

Outputs:
  - summary_terms.txt : selected GO IDs and metadata

Notes:
  - The code aims to be clear and modifiable; it is written to work with general ontologies
    represented by GOATOOLS' GODag.
"""

import argparse
import math
import sys
from collections import defaultdict, deque

import pandas as pd
from goatools.obo_parser import GODag
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS


def read_assoc_tsv(path):
    """Read simple two-column file: element\tGOID per line."""
    assoc = defaultdict(set)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            elt, go = parts[0].strip(), parts[1].strip()
            assoc[elt].add(go)
    return dict(assoc)


def invert_assoc(assoc):
    """Return term -> set(elements) mapping."""
    inv = defaultdict(set)
    for elt, gos in assoc.items():
        for g in gos:
            inv[g].add(elt)
    return dict(inv)


def read_list(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def descendants_of(term, godag):
    """Return set of descendants (recursively). Works robustly even if attribute names vary."""
    # Term objects typically have .children as set of GO IDs or TermObjs
    if term not in godag:
        return set()
    visited = set()
    q = deque()
    t = godag[term]
    # children may be set of GO IDs or TermObjs
    children = getattr(t, 'children', None)
    if children is None:
        # try alternate attribute
        children = getattr(t, 'kids', None)
    if children is None:
        # no children recorded
        return set()
    # normalize children to GO IDs
    def child_ids(children):
        ids = []
        for c in children:
            if isinstance(c, str):
                ids.append(c)
            else:
                # TermObj
                ids.append(c.id)
        return ids

    for c in child_ids(children):
        q.append(c)

    while q:
        cur = q.popleft()
        if cur in visited:
            continue
        visited.add(cur)
        if cur in godag:
            cchildren = getattr(godag[cur], 'children', None)
            if cchildren is None:
                cchildren = getattr(godag[cur], 'kids', None)
            if cchildren:
                for cc in child_ids(cchildren):
                    if cc not in visited:
                        q.append(cc)
    return visited


def compute_ic(term_to_elements, term, population_size):
    n = len(term_to_elements.get(term, []))
    if n == 0:
        return 0.0
    p = n / population_size
    return -math.log2(p)


def greedy_summary(candidates, term_to_elements, godag, population_size, eoi_set):
    """Main greedy loop as described. Returns list of selected terms in order."""
    # candidates: set of GO IDs (over-represented)
    # term_to_elements: mapping GO -> set(elements in population annotated by GO)
    elts_annot_by_summary = set()
    summary = []
    candidates = set(candidates)

    # Precompute descendants for speed
    desc_cache = {}

    def get_descs(t):
        if t not in desc_cache:
            desc_cache[t] = descendants_of(t, godag)
        return desc_cache[t]

    while True:
        # Filter candidates to those that cover at least one EOI not yet covered
        useful = []
        for c in candidates:
            covered = term_to_elements.get(c, set()) & eoi_set
            new_covered = covered - elts_annot_by_summary
            if len(new_covered) > 0:
                ic = compute_ic(term_to_elements, c, population_size)
                score = ic * len(new_covered)
                useful.append((score, c, len(new_covered), ic))
        if not useful:
            break
        useful.sort(key=lambda x: (-x[0], -x[2]))  # highest score, then coverage
        max_score = useful[0][0]
        # select all with same max_score (ties)
        winners = [c for (s, c, cov, ic) in useful if abs(s - max_score) < 1e-12]

        # tie-break: if one winner is ancestor of another, keep the most specific (descendant)
        winners_set = set(winners)
        to_remove_from_winners = set()
        for a in winners:
            for b in winners:
                if a == b:
                    continue
                # if a is ancestor of b (b more specific), drop a
                # check by seeing if b in descendants(a)
                if b in get_descs(a):
                    to_remove_from_winners.add(a)
        winners_final = [w for w in winners if w not in to_remove_from_winners]

        # Add winners to summary
        for w in winners_final:
            summary.append(w)
            covered = term_to_elements.get(w, set()) & eoi_set
            elts_annot_by_summary |= covered
            # remove descendants of w from candidates
            descs = get_descs(w)
            candidates -= descs
        # remove winners themselves from candidates too
        candidates -= set(winners_final)

        # Also remove any candidate that no longer covers any uncovered EOI
        to_remove = set()
        for c in list(candidates):
            covered = term_to_elements.get(c, set()) & eoi_set
            if len(covered - elts_annot_by_summary) == 0:
                to_remove.add(c)
        candidates -= to_remove

    return summary


def prune_summary(summary, term_to_elements, godag, eoi_set):
    """Prune redundant terms inside the summary.
    Two rules described:
      1) If descendant term in summary and ancestor annotates >=1 element of interest not annotated by any other summary term, discard descendant
      2) If an ancestor is redundant because all its EOI are annotated by other summary terms, discard ancestor

    We attempt an iterative pruning until stable.
    """
    summary_set = set(summary)
    changed = True
    # Precompute descendants & ancestors
    desc_cache = {}
    anc_cache = {}
    def get_descs(t):
        if t not in desc_cache:
            desc_cache[t] = descendants_of(t, godag)
        return desc_cache[t]
    # ancestors by walking godag parents
    def get_ancs(t):
        if t in anc_cache:
            return anc_cache[t]
        if t not in godag:
            anc_cache[t] = set()
            return set()
        visited = set()
        q = deque()
        parents = getattr(godag[t], 'parents', None)
        if parents is None:
            parents = getattr(godag[t], 'parents_keys', None)
        if parents:
            # normalize
            for p in parents:
                if isinstance(p, str):
                    q.append(p)
                else:
                    q.append(p.id)
        while q:
            cur = q.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            if cur in godag:
                pp = getattr(godag[cur], 'parents', None)
                if pp:
                    for p in pp:
                        if isinstance(p, str):
                            nid = p
                        else:
                            nid = p.id
                        if nid not in visited:
                            q.append(nid)
        anc_cache[t] = visited
        return visited

    while changed:
        changed = False
        # Rule 1: if descendant in summary and ancestor annotates >=1 EOI not annotated by any other summary term => discard descendant
        for s in list(summary_set):
            ancs = get_ancs(s)
            inter = ancs & summary_set
            for a in inter:
                # check if ancestor a annotates >=1 EOI not annotated by any other summary term
                a_covered = term_to_elements.get(a, set()) & eoi_set
                # elements annotated by other summary terms excluding a
                others = set()
                for t in summary_set:
                    if t == a:
                        continue
                    others |= (term_to_elements.get(t, set()) & eoi_set)
                unique = a_covered - others
                if len(unique) >= 1 and s in summary_set:
                    # discard descendant s
                    summary_set.remove(s)
                    changed = True
        # Rule 2: if ancestor in summary and all its EOI are annotated by >=1 other summary annotation (excluding itself), discard ancestor
        for s in list(summary_set):
            ancs = get_ancs(s)
            for a in list(ancs & summary_set):
                a_covered = term_to_elements.get(a, set()) & eoi_set
                others = set()
                for t in summary_set:
                    if t == a:
                        continue
                    others |= (term_to_elements.get(t, set()) & eoi_set)
                if a_covered and a_covered.issubset(others):
                    summary_set.remove(a)
                    changed = True
    # preserve original order as much as possible
    final = [t for t in summary if t in summary_set]
    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--obo', required=True, help='go-basic.obo file')
    p.add_argument('--assoc', required=True, help='associations tsv: element\tGOID')
    p.add_argument('--eoi', required=True, help='elements of interest (one per line)')
    p.add_argument('--out', default='summary_terms.txt', help='output summary file')
    p.add_argument('--pval', type=float, default=0.05, help='p-value cutoff for enrichment')
    args = p.parse_args()

    print('Loading ontology...')
    godag = GODag(args.obo)
    print('Loading associations...')
    assoc = read_assoc_tsv(args.assoc)
    pop = set(assoc.keys())
    print(f'Population size (elements with any annotation): {len(pop)}')

    print('Loading elements of interest...')
    eoi = read_list(args.eoi)
    eoi_set = set([x for x in eoi if x in pop])
    print(f'Elements of interest contained in population: {len(eoi_set)} / {len(eoi)}')

    # invert assoc to term->elements
    term_to_elements = invert_assoc(assoc)

    # Prepare GOEA: GOATOOLS expects assoc as a mapping element->set(goids)
    print('Running enrichment (GOEA) to find over-represented terms...')
    # Using namespace None so all namespaces are considered. Adjust if you want only BP/CC/MF
    goeaobj = GOEnrichmentStudyNS(list(pop), assoc, godag, methods=['fdr_bh'])
    goea_results_all = goeaobj.run_study(list(eoi_set))
    # Filter significant
    sig = [r for r in goea_results_all if getattr(r, 'p_fdr_bh', 1.0) <= args.pval]
    print(f'Found {len(sig)} significant terms (p_fdr_bh <= {args.pval})')
    candidates = set([r.GO for r in sig])

    # Ensure that term_to_elements contains entries for ancestors as well (because if an element annotated to child it should be annotated by ancestors)
    # We will propagate annotations up the ontology so term_to_elements[anc] includes elements annotated by its descendants
    print('Propagating annotations upward to ensure ancestor coverage...')
    # Build children map to efficiently traverse
    # For each term in godag, gather all descendants and add their elements
    all_terms = set(term_to_elements.keys()) | set(godag.keys())
    propagated = {t: set(term_to_elements.get(t, set())) for t in all_terms}
    for t in list(all_terms):
        descs = descendants_of(t, godag)
        for d in descs:
            propagated[t] |= term_to_elements.get(d, set())
    term_to_elements = propagated

    population_size = len(pop)

    print('Running greedy summarization...')
    summary = greedy_summary(candidates, term_to_elements, godag, population_size, eoi_set)
    print(f'Selected {len(summary)} terms before pruning')

    print('Pruning summary...')
    summary_pruned = prune_summary(summary, term_to_elements, godag, eoi_set)
    print(f'{len(summary_pruned)} terms after pruning')

    # Write output with metadata
    rows = []
    for go in summary_pruned:
        name = godag[go].name if go in godag else ''
        n_all = len(term_to_elements.get(go, set()))
        n_eoi = len(term_to_elements.get(go, set()) & eoi_set)
        ic = compute_ic(term_to_elements, go, population_size)
        rows.append({'GO': go, 'Name': name, 'Annotated_all': n_all, 'Annotated_EOI': n_eoi, 'IC': ic})
    df = pd.DataFrame(rows)
    df = df.sort_values(by=['Annotated_EOI', 'IC'], ascending=[False, False])
    df.to_csv(args.out, sep='\t', index=False)
    print('Wrote summary to', args.out)


if __name__ == '__main__':
    main()
