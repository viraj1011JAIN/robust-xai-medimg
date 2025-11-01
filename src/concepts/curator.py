from __future__ import annotations

import os
import random
import shutil

import pandas as pd

POS, NEG = "pos", "neg"


class ConceptCurator:
    def __init__(
        self, dataset_csv: str, images_root: str, out_root: str, seed: int = 42
    ):
        self.df = pd.read_csv(dataset_csv)
        self.images_root = images_root
        self.out_root = out_root
        random.seed(seed)
        os.makedirs(out_root, exist_ok=True)

    def _copy(self, rows: pd.DataFrame, concept: str, split: str):
        dest = os.path.join(self.out_root, concept, split)
        os.makedirs(dest, exist_ok=True)
        for _, r in rows.iterrows():
            src = os.path.join(self.images_root, r["image_path"])
            dst = os.path.join(dest, os.path.basename(r["image_path"]))
            if os.path.isfile(src):
                shutil.copy2(src, dst)

    def curate_clinical_concept(
        self,
        concept: str,
        pos_query: str,
        neg_query: str,
        n_pos: int = 80,
        n_neg: int = 80,
    ):
        pos_df = self.df.query(pos_query)
        neg_df = self.df.query(neg_query)
        pos_df = pos_df.sample(n=min(n_pos, len(pos_df)), random_state=0)
        neg_df = neg_df.sample(n=min(n_neg, len(neg_df)), random_state=1)
        self._copy(pos_df, concept, POS)
        self._copy(neg_df, concept, NEG)

    def curate_artifact_concept(
        self,
        concept: str = "artifacts_neg",
        artifact_flag_col: str = "has_artifact",
        n_pos: int = 120,
        n_neg: int = 120,
    ):
        if artifact_flag_col not in self.df.columns:
            raise ValueError(f"Missing artifact flag column: {artifact_flag_col}")
        pos_df = self.df[self.df[artifact_flag_col] == 1].sample(
            n=min(n_pos, (self.df[artifact_flag_col] == 1).sum()), random_state=2
        )
        neg_df = self.df[self.df[artifact_flag_col] == 0].sample(
            n=min(n_neg, (self.df[artifact_flag_col] == 0).sum()), random_state=3
        )
        self._copy(pos_df, concept, POS)
        self._copy(neg_df, concept, NEG)
