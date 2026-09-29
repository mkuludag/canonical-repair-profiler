#!/usr/bin/env python3
"""
Technique 9 (semantic): cluster claims by the free-text 3 C's (tech comment / correction)
instead of codes -- surfaces repair patterns that coded fields miss. Memory-scoped: works on
a bounded sample, TF-IDF (capped features) + MiniBatchKMeans. Reports cluster size, top terms,
and within-cluster cost variability (the "contradiction" signal).

Output: grouping/out/text_clusters.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans

PARQUET = os.environ.get("REPAIR_PARQUET", "data/repair_profile.parquet")
OUT = "grouping/out"; os.makedirs(OUT, exist_ok=True)
SAMPLE = int(os.environ.get("TEXT_SAMPLE", "120000"))
K = int(os.environ.get("TEXT_K", "40"))
TEXT = "paws_comment_trail"   # rich 3C narrative trail (tech_comment is mostly a templated pointer string)


def main():
    df = pd.read_parquet(PARQUET, columns=[TEXT, "det_approved_amt", "rep_sub_cat_label"])
    df = df[df[TEXT].notna() & (df[TEXT].str.len() > 15)]
    df["det_approved_amt"] = pd.to_numeric(df["det_approved_amt"], errors="coerce")
    if len(df) > SAMPLE:
        df = df.sample(SAMPLE, random_state=42)
    print(f"clustering {len(df):,} tech-comment texts into K={K} ...")

    vec = TfidfVectorizer(max_features=4000, stop_words="english",
                          ngram_range=(1, 2), min_df=5, max_df=0.7,
                          token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")  # alphabetic words only (drop DTC/$ numbers)
    X = vec.fit_transform(df[TEXT].values)
    km = MiniBatchKMeans(n_clusters=K, random_state=42, n_init=3, batch_size=2048)
    lab = km.fit_predict(X)
    df = df.assign(cluster=lab)
    terms = np.array(vec.get_feature_names_out())
    centers = km.cluster_centers_.argsort()[:, ::-1]

    rows = []
    for k in range(K):
        m = df["cluster"] == k
        c = df.loc[m, "det_approved_amt"].dropna()
        c = c[c > 0]
        cv = float(c.std() / c.mean()) if len(c) >= 3 and c.mean() else np.nan
        rows.append({
            "cluster": k, "n": int(m.sum()),
            "top_terms": ", ".join(terms[centers[k, :8]]),
            "approved_med": round(float(c.median()), 2) if len(c) else np.nan,
            "approved_cv": round(cv, 3) if cv == cv else np.nan,
        })
    res = pd.DataFrame(rows).sort_values("n", ascending=False)
    res.to_csv(f"{OUT}/text_clusters.csv", index=False)
    pd.set_option("display.width", 200, "display.max_colwidth", 70)
    print(res.head(25).to_string(index=False))
    print(f"\nWrote {OUT}/text_clusters.csv")
    print(f"median within-cluster approved-$ CV: {res['approved_cv'].median():.3f} "
          f"(lower = text clusters are more cost-coherent than code groups)")


if __name__ == "__main__":
    main()
