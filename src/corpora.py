"""Unified corpus loaders and schema harmonization for IPVS and MDVR-KCL.

Unified schema:
    corpus: str ("ipvs" | "mdvr_kcl")
    subject_id: str (unique across corpora, e.g. "IPVS_PD01", "KCL_ID00")
    group: str ("PD" | "HC")
    label: int (1 for PD, 0 for HC)
    task: str ("read_passage" | "spontaneous_dialogue" | "vowel_a" | ...)
    task_rep: int
    language: str ("it" | "en")
    age: float (NaN if unpublished)
    sex: str ("M" | "F" | NaN if unpublished)
    severity_hy: float (Hoehn & Yahr, NaN if unavailable)
    severity_u2: float (UPDRS II-5 speech, NaN if unavailable)
    severity_u3: float (UPDRS III-18 speech, NaN if unavailable)
    severity_bin: str (e.g. "1-5", "6-10" for IPVS, NaN for KCL)
    path: str (absolute or relative filesystem path)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from . import parsing as P


KCL_FILENAME_RE = re.compile(
    r"^ID(?P<id_num>\d+)"
    r"_?(?P<group>hc|pd)"
    r"_(?P<hy>\d+(?:\.\d+)?)"
    r"_(?P<u2>\d+(?:\.\d+)?)"
    r"_(?P<u3>\d+(?:\.\d+)?)"
    r"\.wav$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedKCLName:
    subject_id: str
    group: str
    label: int
    hy: float
    u2: float
    u3: float


def parse_kcl_filename(name: str) -> ParsedKCLName | None:
    match = KCL_FILENAME_RE.match(name)
    if match is None:
        return None
    d = match.groupdict()
    group_str = "PD" if d["group"].lower() == "pd" else "HC"
    return ParsedKCLName(
        subject_id=f"KCL_ID{int(d['id_num']):02d}",
        group=group_str,
        label=int(group_str == "PD"),
        hy=float(d["hy"]),
        u2=float(d["u2"]),
        u3=float(d["u3"]),
    )


def load_mdvr_kcl(kcl_root: Path | str) -> tuple[pd.DataFrame, list[Path]]:
    """Walk MDVR-KCL directory and return (metadata_df, unparsed_paths)."""
    kcl_root = Path(kcl_root)
    records: list[dict] = []
    unparsed: list[Path] = []

    if not kcl_root.is_dir():
        raise FileNotFoundError(f"MDVR-KCL directory not found: {kcl_root}")

    for path in sorted(kcl_root.rglob("*.wav")):
        parsed = parse_kcl_filename(path.name)
        if parsed is None:
            unparsed.append(path)
            continue

        parent_names = [p.name for p in path.parents]
        if "ReadText" in parent_names or "readtext" in [p.lower() for p in parent_names]:
            task = "read_passage"
        elif "SpontaneousDialogue" in parent_names or "spontaneous" in [p.lower() for p in parent_names]:
            task = "spontaneous_dialogue"
        else:
            task = "unknown"

        records.append(
            {
                "corpus": "mdvr_kcl",
                "subject_id": parsed.subject_id,
                "group": parsed.group,
                "label": parsed.label,
                "task": task,
                "task_rep": 1,
                "language": "en",
                "age": np.nan,
                "sex": np.nan,
                "severity_hy": parsed.hy,
                "severity_u2": parsed.u2,
                "severity_u3": parsed.u3,
                "severity_bin": np.nan,
                "folder_path": str(path.parent.relative_to(kcl_root)),
                "path": str(path.resolve()),
            }
        )

    if not records:
        raise RuntimeError(f"no valid MDVR-KCL .wav files found under {kcl_root}")

    df = pd.DataFrame.from_records(records)
    return df.sort_values(["group", "subject_id", "task"]).reset_index(drop=True), unparsed


def load_ipvs(ipvs_root: Path | str, metadata_parquet: Path | str | None = None) -> pd.DataFrame:
    """Load IPVS and conform to the unified schema."""
    ipvs_root = Path(ipvs_root)
    if metadata_parquet and Path(metadata_parquet).exists():
        meta = pd.read_parquet(metadata_parquet)
    else:
        meta, _ = P.build_metadata(ipvs_root)
        meta = P.add_audio_properties(meta)

    # Filter to analysis cohort: PD and eHC
    meta = meta[meta["group"].isin(["PD", "eHC"])].copy()

    records: list[dict] = []
    for _, row in meta.iterrows():
        # Task mapping: map "read_text" to "read_passage" for harmonization
        raw_task = row["task"]
        task = "read_passage" if raw_task == "read_text" else raw_task

        # Parse severity bin from folder_path if available (e.g. "1-5", "6-10", "11-16", "17-28")
        folder_parts = str(row["folder_path"]).split("/")
        severity_bin = folder_parts[1] if len(folder_parts) > 2 and folder_parts[0].startswith("28 People") else np.nan

        records.append(
            {
                "corpus": "ipvs",
                "subject_id": f"IPVS_{row['subject_id']}",
                "group": "PD" if row["group"] == "PD" else "HC",
                "label": int(row["group"] == "PD"),
                "task": task,
                "task_rep": int(row["task_rep"]),
                "language": "it",
                "age": float(row["age"]),
                "sex": str(row["sex"]),
                "severity_hy": np.nan,
                "severity_u2": np.nan,
                "severity_u3": np.nan,
                "severity_bin": severity_bin,
                "folder_path": str(row["folder_path"]),
                "path": str(Path(row["path"]).resolve()),
            }
        )

    df = pd.DataFrame.from_records(records)
    return df.sort_values(["group", "subject_id", "task", "task_rep"]).reset_index(drop=True)


def load_harmonized_read_passage(
    ipvs_root: Path | str,
    kcl_root: Path | str,
    ipvs_meta_parquet: Path | str | None = None,
) -> pd.DataFrame:
    """Return unified frame containing only the read passage task across both corpora."""
    ipvs_df = load_ipvs(ipvs_root, ipvs_meta_parquet)
    kcl_df, _ = load_mdvr_kcl(kcl_root)

    combined = pd.concat(
        [
            ipvs_df[ipvs_df["task"] == "read_passage"],
            kcl_df[kcl_df["task"] == "read_passage"],
        ],
        ignore_index=True,
    )
    return combined
