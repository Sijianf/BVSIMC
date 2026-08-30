# Draft — CTD validation of novel predictions (short version)

> Numbers filled from Tables `tab:cdataset-verification-drugs` and
> `tab:cdataset-verification-pairs`. Two items still marked **[needs 9c]** /
> **[needs 9e]** — see the notes at the end before writing them as claims.

---

## Methods (one paragraph)

To assess whether the highest-scoring unobserved pairs correspond to genuine
therapeutic relationships, we validated them against the Comparative Toxicogenomics
Database (CTD; release ⟨YYYY-MM-DD⟩), retaining only curated relationships annotated as
*therapeutic* and discarding all computationally inferred associations. Because CTD
indexes by MeSH while Cdataset uses DrugBank accessions and OMIM MIM numbers, both axes
were mapped: drugs through CAS registry number, preferred name, synonym, and finally
salt-stripped name (657/658 resolved, 99.8%), and diseases through
⟨CTD's curated OMIM cross-references / cross-references plus exact matches of a full
OMIM title against a CTD disease name or synonym⟩ (⟨N⟩/409 resolved). Each method was
fitted to the complete Cdataset matrix with its optimal hyper-parameters, all entries
zero in the training matrix were ranked by predicted probability, and the ranking was
evaluated in two ways following Zhang *et al.*: a **per-drug hit rate**, the fraction of
drugs with at least one curated therapeutic relationship — absent from the training
matrix — among their top *K* predicted diseases; and **precision@N**, the fraction of
the top *N* pairs pooled across all drugs supported by such a record. As in DRIMC, the
per-drug denominator is restricted to the 361 of 658 drugs (54.9%) for which at least
one such relationship exists, since the remainder could not have been recovered by any
method. All methods were processed through an identical pipeline — same matrix, same
mappings, same CTD release, same denominator — so the comparison isolates the ranking
each model produces.

---

## Results — §X.4 Validation of novel drug–disease associations

**¶1.** Table ⟨T1⟩ and Fig. ⟨X⟩a report the per-drug hit rate. BVSIMC ranked a curated
therapeutic relationship first for 14.1% of the 361 eligible drugs, and within the top
5 and top 10 for 33.0% and 42.1% respectively — the best result at every cut-off. The
second-best method, SGIMC, reached 12.5%, 29.4% and 39.9%; DRIMC and IMC were close
behind at top-10 (both 39.3%) but separated more clearly at top-1 (6.9% and 11.6%).
**[needs 9c]** ⟨The advantage over SGIMC was significant at *K* = ⟨…⟩ (McNemar's exact
test, *p* = ⟨…⟩).⟩

**¶2.** The pair-level ranking (Table ⟨T2⟩, Fig. ⟨X⟩b) separates the methods more
sharply. Among the top 500 novel associations, 11.0% of BVSIMC's predictions were
supported by a curated therapeutic record, against 7.6% for SGIMC, 5.6% for DRIMC and
4.6% for IMC — a 45% relative improvement over the second-best method. The margin
persisted to *N* = 1500 (9.1% vs 6.9%). Notably, precision decreased with *N* for
BVSIMC and SGIMC (11.0%→9.1% and 7.6%→6.9%) but increased for IMC and DRIMC
(4.6%→5.7% and 5.6%→6.8%), indicating that only the former two concentrate confirmed
associations at the very top of the ranking, which is the property that matters when a
short candidate list is taken forward for experimental follow-up.

**¶3.** Two features of this benchmark deserve comment. First, only 54.9% of Cdataset
drugs have any curated therapeutic relationship outside the training matrix, compared
with 83.2% of LRSSL drugs in the original DRIMC evaluation; Cdataset's diseases are
drawn from OMIM and are predominantly rare Mendelian phenotypes, for which CTD holds
few therapeutic records. Absolute rates on this dataset are therefore bounded well
below those reported on LRSSL and are not directly comparable to them. Second, as a
reference point, ranking the same unobserved pairs by the product of drug and disease
degree — using no side information and no model — recovered a confirmed indication for
0.0%, 0.0%, 0.3% and 1.9% of drugs at *K* = 1, 3, 5 and 10, confirming that the
observed rates reflect the learned representation rather than the degree structure of
the association matrix. **[needs 9e]** ⟨The ranking of methods was unchanged under all
three disease-mapping criteria (Supplementary Table ⟨S⟩).⟩

---

## Captions

> **Figure ⟨X⟩. Validation of novel drug–disease predictions against CTD curated
> therapeutic records (Cdataset).** (**a**) Per-drug hit rate: percentage of drugs with
> at least one curated CTD therapeutic relationship, absent from the training matrix,
> among their top *K* predicted diseases (n = 361 eligible drugs). (**b**) Precision@N:
> percentage of the top *N* novel pairs, pooled across all drugs, supported by such a
> record. All methods were fitted to the complete Cdataset matrix with their optimal
> hyper-parameters and evaluated through an identical mapping and verification pipeline.
> CTD release ⟨YYYY-MM-DD⟩.

> **Table ⟨T1⟩. Per-drug hit rate on Cdataset (n = 361 eligible drugs), verified against
> CTD curated therapeutic records.** Best result in each column in **bold**.
> Rates over all 658 drugs are given in Supplementary Table ⟨S⟩.

> **Table ⟨T2⟩. Precision among the top *N* novel drug–disease pairs on Cdataset,
> verified against CTD curated therapeutic records.**

---

## Before submission

1. **[needs 9c] The per-drug margins are small in absolute terms.** BVSIMC leads SGIMC
   by ~6 drugs at top-1, ~4 at top-3, ~13 at top-5 and ~8 at top-10, out of 361. Run
   McNemar's exact test; where it is not significant, write *comparable to* rather than
   *outperforms*. The precision@N margins (17–33 pairs, 32–45% relative) are the more
   defensible claim and ¶2 is written to lead with them.
2. **[needs 9e] State the disease-mapping criterion and show the others.** Report the
   tier used and confirm in supplementary material that the ranking holds under the
   other two.
3. **NRLMF's numbers should not be published as they stand** — see the note sent with
   this draft. They are not in the paragraphs above for that reason.
4. **Case study table.** DRIMC's Table 4 is the part biologists read. Take 5–6 drugs
   from the `confirmed` frame, list their top-5 diseases, and mark CTD-verified and
   literature-supported pairs *separately*.
