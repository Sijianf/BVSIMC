#!/usr/bin/env python3
"""
Automated verification of predicted drug-disease pairs.

Input : CSV of top-N predictions with DrugBank IDs (DBxxxxx) and MeSH
        disease IDs (Dxxxxxx).
Output: the same rows annotated with independent evidence from
        (1) CTD curated 'therapeutic' direct evidence   <- DRIMC's criterion
        (2) ChEMBL drug_indication + max clinical phase <- stronger, adds phase
        (3) PubMed co-occurrence hits                   <- weak, for triage only

Usage
-----
  # one-time: download + cache CTD bulk files (~1-2 GB download, ~5 min)
  python verify_predictions.py cache

  # verify
  python verify_predictions.py verify predictions.csv -o verified.tsv

predictions.csv must have columns:
    drugbank_id, mesh_disease_id, score
optional but strongly recommended (needed for the CTD name mapping):
    drug_name, disease_name

Notes
-----
* CTD keys chemicals by MeSH chemical IDs, NOT DrugBank IDs. There is no
  free public DrugBank->MeSH table, so we map via drug name/synonym against
  CTD_chemicals.tsv.gz, falling back to PubChem if no name is supplied.
  Every mapping is reported so you can eyeball the failures.
* Exact MeSH disease matching is the primary criterion (this is what DRIMC
  did). A relaxed hierarchical match is reported in a SEPARATE column so you
  can state both numbers instead of silently inflating the hit rate.
"""

import argparse
import csv
import gzip
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict

CACHE = os.environ.get("VERIFY_CACHE", os.path.expanduser("~/.cache/drug_verify"))

CTD_FILES = {
    "chem_dis": "https://ctdbase.org/reports/CTD_chemicals_diseases.tsv.gz",
    "chemicals": "https://ctdbase.org/reports/CTD_chemicals.tsv.gz",
    "diseases": "https://ctdbase.org/reports/CTD_diseases.tsv.gz",
}

CHEM_DIS_COLS = ["ChemicalName", "ChemicalID", "CasRN", "DiseaseName", "DiseaseID",
                 "DirectEvidence", "InferenceGeneSymbol", "InferenceScore",
                 "OmimIDs", "PubMedIDs"]
CHEMICAL_COLS = ["ChemicalName", "ChemicalID", "CasRN", "Definition", "ParentIDs",
                 "TreeNumbers", "ParentTreeNumbers", "Synonyms"]
DISEASE_COLS = ["DiseaseName", "DiseaseID", "AltDiseaseIDs", "Definition",
                "ParentIDs", "TreeNumbers", "ParentTreeNumbers", "Synonyms",
                "SlimMappings"]

UA = {"User-Agent": "drug-repositioning-verifier/1.0 (research use)"}


# ---------------------------------------------------------------- utilities

def norm_id(x):
    """Strip MESH:/OMIM: prefixes, uppercase. 'MESH:D003866' -> 'D003866'."""
    if not x:
        return ""
    return x.split(":")[-1].strip().upper()


def norm_name(x):
    """Aggressive name normalisation for synonym matching."""
    x = x.lower().strip()
    x = re.sub(r"\s*\(.*?\)\s*", " ", x)          # drop parentheticals
    x = re.sub(r"[^a-z0-9]+", "", x)              # drop punctuation/space
    return x


def get_json(url, data=None, tries=3):
    for i in range(tries):
        try:
            body = json.dumps(data).encode() if data is not None else None
            hdr = dict(UA)
            if body:
                hdr["Content-Type"] = "application/json"
            req = urllib.request.Request(url, data=body, headers=hdr)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            if i == tries - 1:
                sys.stderr.write(f"  ! {url} -> {e}\n")
                return None
            time.sleep(2 ** i)


def find_all(obj, pattern):
    """Recursively pull every string in a nested JSON blob matching pattern.

    Used so the UniChem/PubChem parsing survives schema drift.
    """
    out = []
    rx = re.compile(pattern)
    def walk(o):
        if isinstance(o, str):
            if rx.fullmatch(o):
                out.append(o)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
    walk(obj)
    return out


def read_ctd(path, cols):
    """Stream a CTD tsv.gz, skipping the '#' comment block."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(cols):
                parts += [""] * (len(cols) - len(parts))
            yield dict(zip(cols, parts))


# ------------------------------------------------------------------- cache

def ctd_header(path, n=15):
    """First n comment lines of a CTD file -- carries the release date."""
    out = []
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                out.append(line.rstrip("\n"))
                if len(out) >= n:
                    break
    except Exception as e:
        out.append(f"# (could not read header: {e})")
    return out


def write_provenance():
    """Record exactly which CTD release is cached.

    Reviewers ask 'which version of CTD, accessed when'. Answer it once here
    instead of guessing six months later.
    """
    path = os.path.join(CACHE, "PROVENANCE.txt")
    with open(path, "w") as fh:
        fh.write(f"# cache built {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n\n")
        for url in CTD_FILES.values():
            f = os.path.join(CACHE, os.path.basename(url))
            if not os.path.exists(f):
                continue
            fh.write(f"source : {url}\n")
            fh.write(f"file   : {os.path.basename(f)}\n")
            fh.write(f"bytes  : {os.path.getsize(f)}\n")
            fh.write(f"mtime  : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(f)))}\n")
            for line in ctd_header(f):
                fh.write(f"  {line}\n")
            fh.write("\n")
    print(f"[cache] provenance -> {path}")
    return path


def cmd_cache(_args):
    os.makedirs(CACHE, exist_ok=True)
    for key, url in CTD_FILES.items():
        dest = os.path.join(CACHE, os.path.basename(url))
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f"[cache] {os.path.basename(dest)} already present")
            continue
        print(f"[cache] downloading {url}")
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as out:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        print(f"[cache]   -> {dest} ({os.path.getsize(dest)/1e6:.0f} MB)")
    write_provenance()
    print(f"[cache] done  (cache dir: {CACHE})")


# ------------------------------------------------------------ CTD indexes

def load_ctd_therapeutic():
    """(mesh_chem, mesh_disease) -> {'evidence': str, 'pmids': str}"""
    path = os.path.join(CACHE, "CTD_chemicals_diseases.tsv.gz")
    direct = {}
    n = 0
    for row in read_ctd(path, CHEM_DIS_COLS):
        ev = row["DirectEvidence"]
        if not ev:
            continue                     # ~99% of rows are inferred, skip
        n += 1
        key = (norm_id(row["ChemicalID"]), norm_id(row["DiseaseID"]))
        direct[key] = {"evidence": ev, "pmids": row["PubMedIDs"],
                       "chem_name": row["ChemicalName"],
                       "dis_name": row["DiseaseName"]}
    sys.stderr.write(f"[ctd] {n} curated direct-evidence chemical-disease rows\n")
    return direct


def load_ctd_chem_names(loose=False):
    """normalised name/synonym -> MeSH chemical ID.

    loose=False : CTD names exactly as filed.
    loose=True  : each CTD name additionally keyed with trailing salt/ester
                  words removed, so a DrugBank parent name ('Sildenafil')
                  reaches a CTD entry filed as 'Sildenafil Citrate'.

    Keep the two separate and consult loose only after exact fails -- a loose
    key is a weaker claim and should never outrank an exact one.
    """
    path = os.path.join(CACHE, "CTD_chemicals.tsv.gz")
    idx = {}
    for row in read_ctd(path, CHEMICAL_COLS):
        cid = norm_id(row["ChemicalID"])
        for nm in [row["ChemicalName"]] + row["Synonyms"].split("|"):
            if not nm:
                continue
            key = norm_name(strip_trailing_salt(nm)) if loose else norm_name(nm)
            if key and key not in idx:
                idx[key] = cid
    sys.stderr.write(f"[ctd] {len(idx)} chemical "
                     f"{'salt-stripped ' if loose else ''}name keys\n")
    return idx


def load_ctd_disease_names():
    """normalised disease name/synonym -> MeSH disease ID.

    Only needed if your disease axis is labelled with names rather than
    Dxxxxxx MeSH IDs.
    """
    path = os.path.join(CACHE, "CTD_diseases.tsv.gz")
    idx = {}
    for row in read_ctd(path, DISEASE_COLS):
        did = norm_id(row["DiseaseID"])
        if not did.startswith("D") and not did.startswith("C"):
            continue                       # skip OMIM-only rows
        for nm in [row["DiseaseName"]] + row["Synonyms"].split("|"):
            nm = norm_name(nm)
            if nm and nm not in idx:
                idx[nm] = did
    sys.stderr.write(f"[ctd] {len(idx)} disease name/synonym keys\n")
    return idx


DRUGBANK_VOCAB_URL = (
    "https://raw.githubusercontent.com/dhimmel/drugbank/gh-pages/data/drugbank.tsv")


def parse_drugbank_xml(xml_path, out_tsv="drugbank_vocab.tsv"):
    """Stream DrugBank's full database.xml -> a small TSV of id/name/cas/synonyms.

    The XML is ~2 GB, so this reads line by line and never builds a tree. It
    relies on DrugBank's stable indentation: drug-level fields sit at two
    spaces, synonyms at four. Nested <name> elements (targets, products, salts)
    are deeper and therefore ignored.

    Run once; afterwards load the TSV with load_drugbank_vocab(out_tsv).
    """
    strip = lambda s: html.unescape(re.sub(r"<[^>]+>", "", s)).strip()
    recs, cur = [], None
    with open(xml_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith(("  <", "    <")):
                continue
            s = line.strip()
            if s.startswith('<drugbank-id primary') and line.startswith("  <"):
                cur = {"id": strip(s), "name": "", "cas": "", "syn": []}
                recs.append(cur)
            elif cur is None:
                continue
            elif s.startswith("<name>") and line.startswith("  <"):
                if not cur["name"]:
                    cur["name"] = strip(s)
            elif s.startswith("<cas-number") and line.startswith("  <"):
                cur["cas"] = strip(s)
            elif s.startswith("<synonym") and line.startswith("    <"):
                v = strip(s)
                if v:
                    cur["syn"].append(v)

    with open(out_tsv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["drugbank_id", "name", "cas", "synonyms"])
        for r in recs:
            w.writerow([r["id"], r["name"], r["cas"], "|".join(r["syn"])])
    sys.stderr.write(f"[vocab] parsed {len(recs)} drugs -> {out_tsv}\n")
    return out_tsv


def load_drugbank_vocab(path=None, force=False):
    """DrugBank ID -> {'name', 'cas', 'syn'}.

    path : a TSV produced by parse_drugbank_xml (preferred -- it is your own
           DrugBank release, complete and current).
    None : falls back to downloading dhimmel/drugbank, a public snapshot of
           DrugBank 5.0.x with names only, no CAS or synonyms.
    """
    if path and os.path.exists(path):
        vocab = {}
        with open(path, encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                db = (row.get("drugbank_id") or "").strip()
                if not db:
                    continue
                vocab[db] = {
                    "name": (row.get("name") or "").strip(),
                    "cas":  (row.get("cas") or "").strip(),
                    "syn":  [s for s in (row.get("synonyms") or "").split("|") if s],
                }
        sys.stderr.write(f"[vocab] {len(vocab)} drugs from {path}\n")
        return vocab

    os.makedirs(CACHE, exist_ok=True)
    dest = os.path.join(CACHE, "drugbank_vocab_dhimmel.tsv")
    if force or not os.path.exists(dest) or os.path.getsize(dest) < 1000:
        sys.stderr.write(f"[vocab] no local XML given; downloading "
                         f"{DRUGBANK_VOCAB_URL}\n")
        req = urllib.request.Request(DRUGBANK_VOCAB_URL, headers=UA)
        with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as out:
            out.write(r.read())
    vocab = {}
    with open(dest, encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            db = (row.get("drugbank_id") or "").strip()
            if db:
                vocab[db] = {"name": (row.get("name") or "").strip(),
                             "cas": "", "syn": []}
    sys.stderr.write(f"[vocab] {len(vocab)} drugs (public snapshot)\n")
    return vocab


def load_ctd_chem_cas():
    """CAS registry number -> MeSH chemical ID. Unambiguous when present."""
    path = os.path.join(CACHE, "CTD_chemicals.tsv.gz")
    idx = {}
    for row in read_ctd(path, CHEMICAL_COLS):
        cas = row["CasRN"].strip()
        if cas:
            idx.setdefault(cas, norm_id(row["ChemicalID"]))
    sys.stderr.write(f"[ctd] {len(idx)} CAS numbers\n")
    return idx


def load_ctd_omim_map():
    """OMIM number -> CTD disease ID (MeSH where CTD has one).

    Needed because several benchmark sets (Cdataset, PREDICT) label diseases
    with OMIM MIM numbers, while CTD curates against MeSH. CTD's AltDiseaseIDs
    column carries the OMIM cross-references.
    """
    path = os.path.join(CACHE, "CTD_diseases.tsv.gz")
    omim = {}
    for row in read_ctd(path, DISEASE_COLS):
        canon = norm_id(row["DiseaseID"])
        ids = [row["DiseaseID"]] + row["AltDiseaseIDs"].split("|")
        for a in ids:
            a = a.strip()
            if not a.upper().startswith("OMIM:"):
                continue
            num = a.split(":")[-1].strip()
            # prefer a MeSH canonical id over an OMIM-only one
            if num not in omim or (not omim[num].startswith("D")
                                   and canon.startswith("D")):
                omim[num] = canon
    sys.stderr.write(f"[ctd] {len(omim)} OMIM cross-references\n")
    return omim


# salt / ester / hydrate words. Used in BOTH directions:
#  - stripped from a DrugBank name  ("Imatinib mesylate" -> "Imatinib")
#  - stripped from a CTD name       ("Sildenafil Citrate" -> "Sildenafil")
# The second direction matters more in practice: DrugBank names the parent
# compound while CTD frequently files the marketed salt.
_SALT_WORDS = (
    r"hydrochlorides?|dihydrochlorides?|hydrobromides?|hcl|sulfates?|sulphates?|"
    r"bisulfates?|mesylates?|maleates?|tartrates?|bitartrates?|citrates?|"
    r"acetates?|phosphates?|diphosphates?|succinates?|fumarates?|besylates?|"
    r"tosylates?|napsylates?|nitrates?|chlorides?|bromides?|iodides?|"
    r"oxalates?|lactates?|gluconates?|benzoates?|salicylates?|pamoates?|"
    r"embonates?|aspartates?|glutamates?|carbamates?|"
    r"propionates?|dipropionates?|valerates?|acetonides?|palmitates?|"
    r"stearates?|decanoates?|enanthates?|undecanoates?|caproates?|"
    r"furoates?|etabonates?|xinafoates?|dimesylates?|camsylates?|"
    r"edisylates?|isethionates?|hyclates?|estolates?|gluceptates?|"
    r"cypionates?|pivalates?|teoclates?|aceponates?|butyrates?|"
    r"laurates?|mucates?|hemisuccinates?|monohydrochlorides?|"
    r"trihydrochlorides?|sesquihydrates?|"
    r"sodium|disodium|monosodium|potassium|dipotassium|calcium|magnesium|"
    r"aluminum|aluminium|zinc|meglumine|"
    r"dihydrate|monohydrate|hemihydrate|trihydrate|hydrates?|anhydrous"
)
_SALT = re.compile(r"\b(" + _SALT_WORDS + r")\b", re.I)
_SALT_TOKEN = re.compile(r"^(" + _SALT_WORDS + r")$", re.I)


def strip_trailing_salt(name):
    """Drop trailing salt/ester words: 'Sildenafil Citrate' -> 'Sildenafil'.

    Trailing-only on purpose. Stripping anywhere would turn 'Calcium Carbonate'
    into 'Carbonate', inventing a bogus index key; the salt in a marketed name
    is essentially always a suffix.
    """
    toks = (name or "").split()
    while len(toks) > 1 and _SALT_TOKEN.match(toks[-1]):
        toks.pop()
    out = " ".join(toks)
    return out if len(out) >= 4 else (name or "")


def name_variants(name):
    """Progressively looser forms of a drug name, best match first."""
    name = (name or "").strip()
    if not name:
        return []
    out, seen = [], set()

    def add(x):
        k = norm_name(x)
        if k and k not in seen:
            seen.add(k)
            out.append(k)

    add(name)
    add(_SALT.sub(" ", name))                       # drop salt/hydrate words
    add(re.sub(r"^\(.*?\)-?", "", name))            # drop (R)-, (S)-, (2R,3S)-
    add(re.sub(r"^[dlrs]-", "", name, flags=re.I))  # drop d-/l-/r-/s- prefix
    add(_SALT.sub(" ", re.sub(r"^\(.*?\)-?", "", name)))
    first = name.split()[0] if name.split() else ""
    if len(first) > 5:
        add(first)                                  # last resort: first token
    return out


def match_chem_name(name, chem_index):
    """Map a drug name onto a MeSH chemical ID, trying looser variants."""
    for v in name_variants(name):
        hit = chem_index.get(v)
        if hit:
            return hit
    return None


def match_drug(rec, chem_index, cas_index=None, loose_index=None):
    """Map one DrugBank record onto a MeSH chemical ID.

    Cascade, strongest evidence first:
      1. CAS registry number         -- unambiguous when it hits
      2. preferred name (exact)
      3. synonyms (exact)
      4. preferred name vs salt-stripped CTD names
      5. synonyms vs salt-stripped CTD names

    Steps 4-5 exist because DrugBank names the parent compound while CTD often
    files the marketed salt ('Sildenafil' vs 'Sildenafil Citrate'). The CAS
    numbers differ too -- DrugBank gives the free base, CTD the salt -- so
    step 1 cannot rescue these.

    Returns (mesh_id, how) so you can audit which rule fired.
    """
    if isinstance(rec, str):
        rec = {"name": rec, "cas": "", "syn": []}

    cas = (rec.get("cas") or "").strip()
    if cas and cas_index:
        hit = cas_index.get(cas)
        if hit:
            return hit, "cas"

    name, syns = rec.get("name", ""), rec.get("syn", [])

    hit = match_chem_name(name, chem_index)
    if hit:
        return hit, "name"

    for s in syns:
        hit = match_chem_name(s, chem_index)
        if hit:
            return hit, "synonym"

    if loose_index:
        hit = match_chem_name(name, loose_index)
        if hit:
            return hit, "name-desalted"
        for s in syns:
            hit = match_chem_name(s, loose_index)
            if hit:
                return hit, "synonym-desalted"

    return None, "unmapped"


def build_omim_names(omim_ids, cache_dir=None, hpoa_path=None,
                     out="omim_names.tsv"):
    """OMIM number -> disease title, from an OMIM API cache and/or phenotype.hpoa.

    cache_dir : directory of <mim>.json files from the OMIM API; the title is
                entry.titles.preferredTitle. Highest coverage.
    hpoa_path : HPO's phenotype.hpoa; column 1 is 'OMIM:xxxxxx', column 2 the
                disease name. Free, offline, no API key, but covers only
                diseases with HPO annotations.
    """
    ids = [re.sub(r"^[A-Za-z]+", "", str(i).strip()) for i in omim_ids]
    pref, alts = {}, defaultdict(list)

    def split_titles(s):
        """OMIM packs alternatives as 'A; SYMBOL;;B; SYM2'."""
        out = []
        for part in (s or "").split(";;"):
            t = part.split(";")[0].strip()
            if t:
                out.append(t)
        return out

    if cache_dir and os.path.isdir(cache_dir):
        for i in ids:
            p = os.path.join(cache_dir, f"{i}.json")
            if not os.path.exists(p):
                continue
            try:
                with open(p, encoding="utf-8") as fh:
                    entry = json.load(fh)["omim"]["entryList"][0]["entry"]
                t = entry.get("titles", {})
            except Exception:
                continue
            for nm in split_titles(t.get("preferredTitle") or t.get("title")):
                pref.setdefault(i, nm)
            # alternativeTitles and includedTitles are extra names for the SAME
            # entry -- often the eponym MeSH actually files under.
            for key in ("alternativeTitles", "includedTitles"):
                for nm in split_titles(t.get(key)):
                    if nm not in alts[i]:
                        alts[i].append(nm)

    if hpoa_path and os.path.exists(hpoa_path):
        seen = set()
        with open(hpoa_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith(("#", "database_id")):
                    continue
                f = line.split("\t")
                if len(f) > 1 and f[0].startswith("OMIM:"):
                    k = f[0][5:].strip()
                    if k in seen:
                        continue
                    seen.add(k)
                    nm = f[1].strip()
                    if nm:
                        if k not in pref:
                            pref[k] = nm
                        elif nm not in alts[k]:
                            alts[k].append(nm)

    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["omim", "name", "alt_names", "source"])
        for i in ids:
            w.writerow([i, pref.get(i, ""), "|".join(alts.get(i, [])),
                        "omim_cache" if i in pref else ""])

    n = sum(1 for i in ids if i in pref)
    n_alt = sum(1 for i in ids if alts.get(i))
    sys.stderr.write(f"[omim] {n}/{len(ids)} titles, {n_alt} with alternatives "
                     f"-> {out}\n")
    return out


def load_omim_names(path="omim_names.tsv"):
    """OMIM number -> disease title, from a TSV of omim,name[,source].

    Build it from an OMIM API cache (entry.titles.preferredTitle) or from
    phenotype.hpoa column 2. Names are the second route into MeSH when CTD's
    OMIM cross-references come up short.
    """
    out = {}
    if not os.path.exists(path):
        sys.stderr.write(f"[omim] {path} not found; name route disabled\n")
        return out
    with open(path, encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            k = (row.get("omim") or "").strip()
            names = [(row.get("name") or "").strip()]
            names += [a for a in (row.get("alt_names") or "").split("|") if a]
            names = [n for n in names if n]
            if k and names:
                out[k] = names
    n_alt = sum(1 for v in out.values() if len(v) > 1)
    sys.stderr.write(f"[omim] {len(out)} OMIM entries "
                     f"({n_alt} with alternative titles)\n")
    return out


# OMIM titles carry qualifiers MeSH does not use:
#   'RESTLESS LEGS SYNDROME, SUSCEPTIBILITY TO, 1' vs MeSH 'Restless Legs Syndrome'
# OMIM bookkeeping markers. 'GLOSSITIS, BENIGN MIGRATORY, INCLUDED' names the
# same entity as 'GLOSSITIS, BENIGN MIGRATORY' -- dropping the marker is not a
# semantic widening, so it happens before variants are generated and never
# counts as peeling.
_OMIM_BOOKKEEPING = re.compile(
    r"(,\s*)?\b(included|formerly|obsolete)\b\.?\s*$", re.I)


def clean_omim_title(title):
    """Strip OMIM bookkeeping markers from a title."""
    t = (title or "").split(";")[0].strip()
    prev = None
    while t != prev:
        prev = t
        t = _OMIM_BOOKKEEPING.sub("", t).strip().strip(",")
    return t


_OMIM_QUAL = re.compile(
    r",\s*(susceptibility to|resistance to|modifier of|"
    r"type\s+[ivx0-9]+|autosomal (dominant|recessive)|x-linked( dominant|"
    r" recessive)?|mitochondrial|somatic|familial|congenital|juvenile|"
    r"infantile|early-onset|late-onset|\d+)\s*$", re.I)


def omim_title_variants(title):
    """Progressively trimmed forms of an OMIM title, most specific first."""
    title = clean_omim_title(title)
    if not title:
        return []
    out, seen = [], set()

    def add(x):
        x = x.strip().strip(",")
        if x and x.lower() not in seen:
            seen.add(x.lower())
            out.append(x)

    add(title)
    prev, cur = None, title
    while cur != prev:                       # peel qualifiers one at a time
        prev = cur
        cur = _OMIM_QUAL.sub("", cur).strip().strip(",")
        add(cur)
    if "," in title:                         # 'X SYNDROME, TYPE 2' -> 'X SYNDROME'
        add(title.split(",")[0])
    return out


def eligible_drugs(Y, drug_ids, disease_ids, drug2mesh, dis2mesh, ctd):
    """Drugs for which a hit is possible at all.

    This applies DRIMC's section 3.4 filtering RULE to whatever dataset you
    pass -- it does not try to reproduce their count. DRIMC ran it on LRSSL
    (763 drugs) and got 635; the same rule on Cdataset (658 drugs) gives a
    different number, and that number is itself a result worth reporting.

    The rule: keep a drug if it has at least one CTD curated therapeutic
    relationship that is NOT already in the training matrix. Drugs failing it
    could not have scored no matter how good the model is, so including them
    in a denominator measures the dataset, not the method.

    Report both rates. eligible-only is methodologically comparable to DRIMC
    (same rule, different dataset); unconditional is the honest end-to-end
    figure.

    IMPORTANT: two very different things exclude a drug here, and DRIMC's
    single number conflates them --
      * unmapped         : the DrugBank ID never reached a MeSH chemical, so
                           CTD could not be queried at all. A tooling failure.
      * mapped_no_record : CTD was queried and has nothing novel. Real absence
                           of evidence.
    The returned stats dict separates them. Reporting that split is strictly
    better than what the paper does, and it is only credible when the mapping
    rate is high -- which is why the mapping work was worth doing.

    Returns (eligible_ids, targets) where targets maps a drug to the set of
    disease ids that would count as a hit for it.
    """
    elig, targets = set(), {}
    n_unmapped = n_no_record = 0
    for i, d in enumerate(drug_ids):
        mc = drug2mesh.get(str(d))
        if not mc:
            n_unmapped += 1          # technical failure, NOT absence of evidence
            continue
        hits = set()
        row = Y[i]
        for j, p in enumerate(disease_ids):
            if row[j] > 0:                       # already known -> not novel
                continue
            md = dis2mesh.get(str(p))
            if not md:
                continue
            rec = ctd.get((mc, md))
            if rec and "therapeutic" in rec["evidence"]:
                hits.add(str(p))
        if hits:
            elig.add(str(d))
            targets[str(d)] = hits
        else:
            n_no_record += 1         # genuinely nothing in CTD to find
    stats = {"total": len(drug_ids), "eligible": len(elig),
             "unmapped": n_unmapped, "mapped_no_record": n_no_record}
    sys.stderr.write(
        f"[elig] {len(elig)}/{len(drug_ids)} drugs have >=1 novel CTD "
        f"therapeutic relation\n"
        f"       excluded: {n_unmapped} unmapped to MeSH, "
        f"{n_no_record} mapped but no novel CTD record\n")
    return elig, targets, stats


def load_ctd_disease_labels():
    """CTD disease ID -> primary name. For auditing what a mapping landed on."""
    path = os.path.join(CACHE, "CTD_diseases.tsv.gz")
    out = {}
    for row in read_ctd(path, DISEASE_COLS):
        did = norm_id(row["DiseaseID"])
        if did and did not in out:
            out[did] = row["DiseaseName"].strip()
    return out


def audit_disease_mapping(mapping, via, omim_names, labels, dis_name_index):
    """Rows for eyeballing an OMIM->MeSH mapping, plus the collisions.

    Two independent things can make a name match differ from the preferred
    title, and they are NOT equally weak:

      used_alt : matched on an alternativeTitle instead of the preferred one.
                 Still an EXACT string match -- 'ADIE PUPIL' also carries
                 'HOLMES-ADIE SYNDROME', and MeSH files it under the eponym.
                 Nothing is lost; this is as strong as the xref route.
      peeled   : a qualifier had to be stripped off before anything matched,
                 which WIDENS meaning ('MYOPATHY, DISTAL, INFANTILE-ONSET' ->
                 'Muscular Diseases'). Only these need adjudication.

    Reporting them in one column conflates an exact synonym hit with a
    semantic broadening, so they are separate here. `peeled` agrees with
    via == 'omim-name-peeled' by construction.

    Returns (audit_rows, collisions) as lists of dicts.
    """
    rows = []
    for src, tgt in mapping.items():
        if via.get(src) not in ("omim-name", "omim-name-peeled") or not tgt:
            continue
        num = re.sub(r"^[A-Za-z]+", "", str(src))
        titles = omim_names.get(num) or [""]
        if isinstance(titles, str):
            titles = [titles]
        title = titles[0]
        matched, src_title = "", ""
        # pass 1 mirrors resolve_disease_axis: any FULL title, exact
        for t in titles:
            if dis_name_index.get(norm_name(clean_omim_title(t))) == tgt:
                matched, src_title = clean_omim_title(t), t
                break
        # pass 2: a peeled variant of some title
        if not matched:
            for t in titles:
                for k, cand in enumerate(omim_title_variants(t)):
                    if k == 0:
                        continue
                    if dis_name_index.get(norm_name(cand)) == tgt:
                        matched, src_title = cand, t
                        break
                if matched:
                    break
        peeled = (norm_name(matched) != norm_name(clean_omim_title(src_title))
                  if matched else True)
        rows.append({
            "disease_id": src,
            "omim_title": title,                       # preferred title
            "matched_title": src_title,                # the title that hit
            "matched_on": matched,                     # after any peeling
            "used_alt": norm_name(src_title) != norm_name(title),
            "peeled": peeled,
            "route": via.get(src, ""),
            "ctd_id": tgt,
            "ctd_name": labels.get(tgt, ""),
        })
        # the audit must never disagree with the tier logic
        assert peeled == (via.get(src) == "omim-name-peeled"), (
            f"audit/route mismatch for {src}: peeled={peeled}, via={via.get(src)}")
    rows.sort(key=lambda r: (not r["peeled"], not r["used_alt"], r["omim_title"]))

    back = defaultdict(list)
    for src, tgt in mapping.items():
        if tgt:
            back[tgt].append(src)
    collisions = [{"ctd_id": t, "ctd_name": labels.get(t, ""),
                   "n": len(s), "disease_ids": ", ".join(sorted(s)),
                   "omim_titles": " | ".join(
                       (lambda t: (t[0] if isinstance(t, list) else t) or "")(
                           omim_names.get(re.sub(r"^[A-Za-z]+", "", str(x)), ""))[:40]
                       for x in sorted(s))}
                  for t, s in back.items() if len(s) > 1]
    collisions.sort(key=lambda r: -r["n"])
    return rows, collisions


def resolve_disease_axis(ids, tree, omim_map, dis_name_index=None,
                         omim_names=None, manual=None):
    """Work out whether a disease axis is MeSH or OMIM, and map it onto CTD.

    Both look like D+6 digits in some datasets, so guessing from the string is
    unsafe -- and guessing wrong fails silently, producing zero confirmed pairs
    with no error. We try both readings and keep whichever actually resolves.

    For an OMIM axis there are two routes into CTD, used in order:
      1. CTD's own OMIM cross-references (AltDiseaseIDs)   -- an explicit link
      2. the OMIM title matched against CTD disease names  -- broader, looser

    Returns (mapping, report, via).
    """
    ids = [str(i).strip() for i in ids]
    omim_names = omim_names or {}
    manual = manual or {}

    as_mesh = {i: norm_id(i) for i in ids}
    n_mesh = sum(1 for v in as_mesh.values() if v in tree)

    as_omim, via = {}, {}
    for i in ids:
        if i in manual:
            as_omim[i], via[i] = norm_id(manual[i]), "manual"
            continue
        num = re.sub(r"^[A-Za-z]+", "", i).lstrip("0") or i
        hit = omim_map.get(num) or omim_map.get(i.lstrip("Dd"))
        if hit:
            as_omim[i], via[i] = hit, "omim-xref"
            continue
        if dis_name_index and omim_names:
            titles = omim_names.get(num) or []
            if isinstance(titles, str):
                titles = [titles]
            # Pass 1: any FULL title (preferred or alternative) matching exactly.
            # Alternatives matter -- MeSH often files a rare disease under its
            # eponym ('Rosenthal-Kloepfer syndrome') rather than OMIM's
            # descriptive preferred title.
            hit = None
            for t in titles:
                hit = dis_name_index.get(norm_name(clean_omim_title(t)))
                if hit:
                    as_omim[i], via[i] = hit, "omim-name"
                    break
            if hit:
                continue
            # Pass 2: allow qualifier peeling, which widens meaning.
            for t in titles:
                for k, cand in enumerate(omim_title_variants(t)):
                    if k == 0:
                        continue
                    m = dis_name_index.get(norm_name(cand))
                    if m:
                        as_omim[i], via[i] = m, "omim-name-peeled"
                        hit = m
                        break
                if hit:
                    break
            if hit:
                continue
        as_omim.setdefault(i, None)
        via.setdefault(i, "unmapped")

    n_omim = sum(1 for v in as_omim.values() if v)

    if n_omim > n_mesh:
        kind, mapping, n_ok = "omim", as_omim, n_omim
    else:
        kind = "mesh"
        mapping = {i: (v if v in tree else None) for i, v in as_mesh.items()}
        via = {i: ("mesh" if mapping[i] else "unmapped") for i in ids}
        n_ok = n_mesh

    counts = defaultdict(int)
    for v in via.values():
        counts[v] += 1
    report = {"kind": kind, "resolved": n_ok, "total": len(ids),
              "as_mesh": n_mesh, "as_omim": n_omim, "via": dict(counts)}
    return mapping, report, via


def diagnose_mapping(db_ids, drug_names, chem_index, cas_index=None,
                     loose_index=None, n_show=8):
    """Report exactly which stage of DrugBank -> name -> MeSH is failing.

    Call this when the mapped count looks wrong. A total failure (0/N) is
    almost always one broken stage, not many bad names.
    """
    db_ids = list(db_ids)
    print(f"input                     : {len(db_ids)} DrugBank IDs")
    print(f"  sample                  : {db_ids[:5]}")

    looks_db = sum(bool(re.match(r'^DB\d{5}$', str(d), re.I)) for d in db_ids)
    print(f"  match ^DB\\d{{5}}$         : {looks_db}/{len(db_ids)}")
    if looks_db == 0:
        print("  !! these are not DrugBank IDs -- check drugs_features.index")
        return

    def rec_of(d):
        r = drug_names.get(d)
        return {"name": r, "cas": "", "syn": []} if isinstance(r, str) else (r or {})

    named = [d for d in db_ids if rec_of(d).get("name")]
    print(f"stage 1  DB id -> name    : {len(named)}/{len(db_ids)}")
    if not named:
        print("  !! name lookup returned nothing. Load the vocabulary first:")
        print("       vocab = vp.load_drugbank_vocab('drugbank_vocab.tsv')")
        return
    print(f"  sample                  : "
          f"{[(d, rec_of(d)['name']) for d in named[:4]]}")

    print(f"chem index size           : {len(chem_index)}")
    if len(chem_index) < 1000:
        print("  !! CTD chemical index looks empty -- re-run the cache download")
        return

    hits = {d: match_drug(rec_of(d), chem_index, cas_index, loose_index)
            for d in named}
    how = defaultdict(int)
    for d, (m, h) in hits.items():
        how[h] += 1
    print(f"stage 2  name -> MeSH     : "
          f"{sum(1 for m, _ in hits.values() if m)}/{len(named)} mapped")
    for k in ("cas", "name", "synonym", "name-desalted",
              "synonym-desalted", "unmapped"):
        if how[k]:
            print(f"    via {k:<9}: {how[k]}")

    failed = [d for d in named if not hits[d][0]]
    if failed:
        print(f"\nstill unmapped ({len(failed)}), first {n_show}:")
        for d in failed[:n_show]:
            r = rec_of(d)
            print(f"  {d}  {r['name']!r}  cas={r.get('cas') or '-'}  "
                  f"syn={len(r.get('syn') or [])}")


def resolve_drug_names(db_ids, cache_path="drug_names.csv", sleep=0.25):
    """DrugBank IDs -> drug names via PubChem, cached to disk and resumable.

    You need names because CTD keys chemicals by MeSH, and name/synonym
    matching is the only free route from a DrugBank ID. This is slow
    (~2 requests per drug) but you only pay it once: results are appended to
    cache_path, so re-running skips everything already resolved. Safe to
    interrupt with Ctrl-C and restart.
    """
    known = {}
    if os.path.exists(cache_path):
        for row in csv.DictReader(open(cache_path)):
            if row.get("drug_name"):
                known[row["drugbank_id"]] = row["drug_name"]
        sys.stderr.write(f"[names] {len(known)} cached\n")

    todo = [d for d in db_ids if d not in known]
    if not todo:
        return known

    sys.stderr.write(f"[names] resolving {len(todo)} via PubChem "
                     f"(~{len(todo) * sleep * 2 / 60:.1f} min)\n")
    new_file = not os.path.exists(cache_path)
    with open(cache_path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["drugbank_id", "drug_name"])
        for i, db in enumerate(todo, 1):
            try:
                nm = pubchem_name(db)
            except KeyboardInterrupt:
                sys.stderr.write("\n[names] interrupted; progress saved\n")
                break
            except Exception:
                nm = None
            if nm:
                known[db] = nm
            w.writerow([db, nm or ""])
            fh.flush()
            if i % 25 == 0 or i == len(todo):
                sys.stderr.write(f"[names]   {i}/{len(todo)}\n")
            time.sleep(sleep)
    return known


def load_disease_tree():
    """MeSH disease ID -> set of tree numbers, for the relaxed match."""
    path = os.path.join(CACHE, "CTD_diseases.tsv.gz")
    tree = {}
    for row in read_ctd(path, DISEASE_COLS):
        did = norm_id(row["DiseaseID"])
        tns = set(t for t in row["TreeNumbers"].split("|") if t)
        if tns:
            tree[did] = tns
    return tree


def hierarchically_related(a, b, tree):
    """True if a and b are ancestor/descendant in the MeSH disease tree."""
    ta, tb = tree.get(a, set()), tree.get(b, set())
    for x in ta:
        for y in tb:
            if x == y or x.startswith(y + ".") or y.startswith(x + "."):
                return True
    return False


# ------------------------------------------------------------ ChEMBL route

def drugbank_to_chembl(db_id):
    """DrugBank ID -> ChEMBL ID via UniChem (DrugBank is src_id 2).

    NOTE: verify this response shape once against a known drug, e.g.
      DB00619 (imatinib) should resolve to CHEMBL941.
    """
    r = get_json("https://www.ebi.ac.uk/unichem/api/v1/compounds",
                 {"type": "sourceID", "compound": db_id, "sourceID": 2})
    hits = find_all(r, r"CHEMBL\d+")
    return hits[0] if hits else None


def chembl_indications(chembl_id):
    """ChEMBL ID -> {mesh_id: max_phase}"""
    url = ("https://www.ebi.ac.uk/chembl/api/data/drug_indication.json"
           f"?molecule_chembl_id={chembl_id}&limit=1000")
    r = get_json(url)
    out = {}
    if not r:
        return out
    for rec in r.get("drug_indications", []):
        mid = norm_id(rec.get("mesh_id") or "")
        if not mid:
            continue
        try:
            ph = float(rec.get("max_phase_for_ind") or 0)
        except (TypeError, ValueError):
            ph = 0.0
        out[mid] = max(out.get(mid, 0.0), ph)
    return out


# ------------------------------------------------------------ PubMed route

def pubmed_hits(drug, disease, api_key=None, retmax=5):
    if not drug or not disease:
        return 0, ""
    term = f'"{drug}"[tiab] AND "{disease}"[tiab]'
    q = {"db": "pubmed", "term": term, "retmax": retmax, "retmode": "json"}
    if api_key:
        q["api_key"] = api_key
    url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
           + urllib.parse.urlencode(q))
    r = get_json(url)
    time.sleep(0.12 if api_key else 0.35)      # NCBI rate limit
    if not r:
        return 0, ""
    res = r.get("esearchresult", {})
    return int(res.get("count", 0)), ",".join(res.get("idlist", []))


# ---------------------------------------------------------------- pipeline

def cmd_verify(args):
    rows = list(csv.DictReader(open(args.predictions)))
    if not rows:
        sys.exit("no rows in predictions file")
    for req in ("drugbank_id", "mesh_disease_id"):
        if req not in rows[0]:
            sys.exit(f"predictions file needs a '{req}' column")

    ctd = load_ctd_therapeutic()
    chem_idx = load_ctd_chem_names()
    tree = load_disease_tree() if not args.no_hierarchy else {}

    # ---- map DrugBank -> MeSH chemical, and DrugBank -> ChEMBL
    db_ids = sorted({r["drugbank_id"].strip() for r in rows})
    names = {r["drugbank_id"].strip(): (r.get("drug_name") or "").strip()
             for r in rows}
    db2mesh, db2chembl, db2ind = {}, {}, {}

    for db in db_ids:
        nm = names.get(db, "")
        if not nm and not args.no_pubchem:
            nm = pubchem_name(db) or ""
            names[db] = nm
        mesh = chem_idx.get(norm_name(nm)) if nm else None
        db2mesh[db] = mesh
        if not args.no_chembl:
            ch = drugbank_to_chembl(db)
            db2chembl[db] = ch
            db2ind[db] = chembl_indications(ch) if ch else {}
        sys.stderr.write(f"[map] {db:<10} name={nm[:28]:<28} "
                         f"mesh={mesh or '-':<12} chembl={db2chembl.get(db) or '-'}\n")

    unmapped = [d for d in db_ids if not db2mesh[d]]
    if unmapped:
        sys.stderr.write(f"[map] WARNING {len(unmapped)}/{len(db_ids)} drugs "
                         f"unmapped to MeSH: {', '.join(unmapped[:10])}\n")

    # ---- annotate each predicted pair
    out_cols = ["rank", "drugbank_id", "drug_name", "mesh_disease_id",
                "disease_name", "score", "ctd_therapeutic", "ctd_evidence",
                "ctd_pmids", "ctd_therapeutic_relaxed", "chembl_max_phase",
                "pubmed_count", "pubmed_pmids", "verdict"]
    out = []
    for i, r in enumerate(rows, 1):
        db = r["drugbank_id"].strip()
        dis = norm_id(r["mesh_disease_id"])
        mesh = db2mesh.get(db)

        hit = ctd.get((mesh, dis)) if mesh else None
        therapeutic = bool(hit and "therapeutic" in hit["evidence"])

        relaxed = False
        if mesh and not therapeutic and tree:
            for (c, d), v in ctd.items():
                if c == mesh and "therapeutic" in v["evidence"] \
                        and hierarchically_related(dis, d, tree):
                    relaxed = True
                    break

        phase = db2ind.get(db, {}).get(dis, "")
        pc, pmids = (0, "")
        if args.pubmed:
            pc, pmids = pubmed_hits(names.get(db, ""),
                                    (r.get("disease_name") or "").strip(),
                                    args.ncbi_key)

        if therapeutic:
            verdict = "CTD_THERAPEUTIC"
        elif phase != "" and float(phase) >= 4:
            verdict = "CHEMBL_APPROVED"
        elif phase != "" and float(phase) > 0:
            verdict = f"CHEMBL_PHASE_{phase}"
        elif relaxed:
            verdict = "CTD_RELATED_TERM"
        elif pc > 0:
            verdict = "LITERATURE_ONLY"
        elif not mesh:
            verdict = "UNMAPPED"
        else:
            verdict = "NO_EVIDENCE"

        out.append({
            "rank": i, "drugbank_id": db, "drug_name": names.get(db, ""),
            "mesh_disease_id": dis,
            "disease_name": r.get("disease_name", ""),
            "score": r.get("score", ""),
            "ctd_therapeutic": int(therapeutic),
            "ctd_evidence": hit["evidence"] if hit else "",
            "ctd_pmids": hit["pmids"] if hit else "",
            "ctd_therapeutic_relaxed": int(relaxed),
            "chembl_max_phase": phase,
            "pubmed_count": pc, "pubmed_pmids": pmids,
            "verdict": verdict,
        })

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out_cols, delimiter="\t")
        w.writeheader()
        w.writerows(out)

    # sidecar: which CTD release produced this table, and with what flags
    side = os.path.splitext(args.out)[0] + ".provenance.txt"
    with open(side, "w") as fh:
        fh.write(f"generated   : {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
        fh.write(f"predictions : {os.path.abspath(args.predictions)}\n")
        fh.write(f"cache dir   : {CACHE}\n")
        fh.write(f"flags       : pubmed={args.pubmed} chembl={not args.no_chembl} "
                 f"hierarchy={not args.no_hierarchy}\n")
        fh.write(f"drugs       : {len(db_ids)}  unmapped: {len(unmapped)}\n\n")
        cache_prov = os.path.join(CACHE, "PROVENANCE.txt")
        if os.path.exists(cache_prov):
            fh.write(open(cache_prov).read())
    print(f"wrote {side}")

    # ---- summary: per-drug hit rate at top-K, DRIMC style
    by_drug = defaultdict(list)
    for o in out:
        by_drug[o["drugbank_id"]].append(o)
    print(f"\nwrote {args.out}   ({len(out)} pairs, {len(by_drug)} drugs)")
    print(f"{'':<10}{'top-1':>8}{'top-5':>8}{'top-10':>8}   (>=1 CTD-therapeutic hit)")
    hits = {}
    for K in (1, 5, 10):
        h = sum(1 for d, lst in by_drug.items()
                if any(o["ctd_therapeutic"] for o in lst[:K]))
        hits[K] = h
    print(f"{'drugs':<10}" + "".join(f"{hits[K]:>8}" for K in (1, 5, 10)))
    print(f"{'rate':<10}" + "".join(f"{hits[K]/len(by_drug):>8.1%}"
                                    for K in (1, 5, 10)))
    n_unmapped = sum(1 for o in out if o["verdict"] == "UNMAPPED")
    if n_unmapped:
        print(f"\n{n_unmapped} pairs UNMAPPED -- fix drug names before "
              f"quoting any hit rate.")


def pubchem_name(db_id):
    """Fallback DrugBank ID -> preferred name via PubChem."""
    r = get_json("https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/"
                 f"sourceid/DrugBank/{db_id}/cids/JSON")
    cids = find_all(r, r"\d{2,}")
    if not cids:
        return None
    r2 = get_json("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                  f"{cids[0]}/synonyms/JSON")
    syns = (r2 or {}).get("InformationList", {}).get("Information", [{}])[0] \
        .get("Synonym", [])
    return syns[0] if syns else None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cache", help="download CTD bulk files")
    c.set_defaults(func=cmd_cache)

    v = sub.add_parser("verify", help="verify a predictions CSV")
    v.add_argument("predictions")
    v.add_argument("-o", "--out", default="verified.tsv")
    v.add_argument("--pubmed", action="store_true",
                   help="also query PubMed co-occurrence (slow, weak evidence)")
    v.add_argument("--ncbi-key", default=os.environ.get("NCBI_API_KEY"))
    v.add_argument("--no-chembl", action="store_true")
    v.add_argument("--no-pubchem", action="store_true")
    v.add_argument("--no-hierarchy", action="store_true")
    v.set_defaults(func=cmd_verify)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()