// Human-readable names + descriptions for known metric keys.
// Keys not listed here fall back to the raw key with no tooltip (forward-compat).
const METRIC_META = {
  citation_correctness: {
    name: "Correctness",
    desc: "Proportion of cited claims whose cited documents support them.",
  },
  citation_recall: {
    name: "Recall",
    desc: "LongCite-style citation recall: fraction of claims supported by their citations, averaged over claims.",
  },
  citation_precision: {
    name: "Precision",
    desc: "LongCite-style citation precision: fraction of citations that support their claim, averaged over citations.",
  },
  citation_f1: {
    name: "Citation F1",
    desc: "Harmonic mean of citation precision and recall.",
  },
  citation_lengths: {
    name: "Citation lengths",
    desc: "Average number of tokens per cited claim; a verifiability proxy (shorter is easier to check).",
  },
  citation_number: {
    name: "Citation number",
    desc: "Average number of inline citation markers per response.",
  },
  coverage: {
    name: "Citation coverage",
    desc: "Proportion of claims that carry at least one citation.",
  },
  cvcp: {
    name: "CVCP",
    desc: "Coefficient of variation of citation positions: how evenly citations are distributed across the answer.",
  },
  hallucination_rate: {
    name: "Hallucination rate",
    desc: "Proportion of cited document references that do not exist among the provided documents.",
  },
  supported_claim_rate: {
    name: "Supported claim rate",
    desc: "Proportion of all claims (cited or not) supported by their cited documents.",
  },
  token_overlap: {
    name: "Token overlap",
    desc: "Average Jaccard token overlap between claims and their cited documents.",
  },
};
