"""Guards for the two failure modes that would silently invalidate the study.

These are tests rather than notebook assertions on purpose: a notebook cell
that raises still leaves a committed notebook that looks fine, whereas a failing
test blocks the run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import metrics as M
from src import parsing as P
from src import splits as S


# --------------------------------------------------------------------------- #
# Filename parsing: every irregular name observed in the real corpus.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "name,task_base,code,birth_year,sex",
    [
        ("B1APGANRET55F170320171104.wav", "B", "APGANRET", 55, "F"),
        ("VU2APGANRET55F170320171112.wav", "VU", "APGANRET", 55, "F"),
        ("FB1ubguot_t40M1606161807.wav", "FB", "ubguot_t", 40, "M"),   # underscore
        ("B1VLIAT OP47M100220171314.wav", "B", "VLIAT OP", 47, "M"),   # space
        ("PR1LBULCAAS94M100120171015..wav", "PR", "LBULCAAS", 94, "M"),  # double dot
        ("PR11LBULCAAS94M100120171057.wav", "PR", "LBULCAAS", 94, "M"),  # two-digit rep
    ],
)
def test_parse_irregular_filenames(name, task_base, code, birth_year, sex):
    parsed = P.parse_filename(name)
    assert parsed is not None, f"failed to parse {name}"
    assert parsed.task_base == task_base
    assert parsed.code == code
    assert parsed.birth_year == birth_year
    assert parsed.sex == sex


def test_code_signature_is_order_insensitive():
    """The scrambling differs between sessions for the same participant."""
    a = P.parse_filename("B1rriovbie49M2605161841.wav")
    b = P.parse_filename("B1RROIBVEI49M240120171859.wav")
    assert a.subject_key == b.subject_key


def test_different_code_is_a_different_participant():
    a = P.parse_filename("B1VSIPTIOZ46M240120171920.wav")
    b = P.parse_filename("B1VSIOTLOP47M100220171328.wav")
    assert a.code_signature != b.code_signature


def test_anagram_codes_are_only_separable_by_birth_year():
    """Two elderly controls whose scrambled codes are anagrams of each other."""
    a = P.parse_filename("B1VAILTFOO49M230320171029.wav")
    b = P.parse_filename("B1VLIATFOO55M300320171237.wav")
    assert a.code_signature == b.code_signature
    assert a.participant_key(a.birth_year) != b.participant_key(b.birth_year)


def test_session_date_drops_the_time_of_day():
    """Two readings minutes apart are one session, not two."""
    a = P.parse_filename("B1vsiptioz46M1606161659.wav")
    b = P.parse_filename("B2vsiptioz46M1606161701.wav")
    assert a.session_date == b.session_date == "160616"


def test_unparsable_name_returns_none():
    assert P.parse_filename("not-a-corpus-file.wav") is None


def _fake_corpus(root):
    """Minimal corpus exercising the typo and anagram cases together.

    ``build_metadata`` only reads filenames, so empty files are enough.
    """
    layout = {
        "28 People with Parkinson's disease/1-5/Antonia G": [
            # Fifteen files agree on 52, one carries a typo.
            *[f"{t}AGNUTGOL52F100220171041.wav" for t in ("B1", "B2", "D1", "VA1")],
            "VI2AGNUTGOL63F100220171052.wav",
        ],
        "22 Elderly Healthy Control/VITO A": [
            f"{t}VAILTFOO49M230320171029.wav" for t in ("B1", "B2")
        ],
        "22 Elderly Healthy Control/VITO L": [
            f"{t}VLIATFOO55M300320171237.wav" for t in ("B1", "B2")
        ],
        "15 Young Healthy Control/Someone A": ["B1AAAAAAAA94M100120171015.wav"],
    }
    for folder, names in layout.items():
        (root / folder).mkdir(parents=True, exist_ok=True)
        for name in names:
            (root / folder / name).touch()
    return root


def test_directory_scoped_typo_does_not_fork_a_participant(tmp_path):
    meta, unparsed = P.build_metadata(_fake_corpus(tmp_path))
    assert not unparsed
    pd_rows = meta[meta["group"] == "PD"]
    assert pd_rows["subject_id"].nunique() == 1, "birth-year typo forked a participant"
    assert pd_rows["age"].nunique() == 1
    assert meta["birth_year_typo"].sum() == 1


def test_anagram_controls_are_not_merged(tmp_path):
    meta, _ = P.build_metadata(_fake_corpus(tmp_path))
    ehc = meta[meta["group"] == "eHC"]
    assert ehc["subject_id"].nunique() == 2, "anagram codes were merged into one person"


def test_anonymise_directories_removes_names_but_keeps_structure(tmp_path):
    meta, _ = P.build_metadata(_fake_corpus(tmp_path))
    safe = P.anonymise_directories(meta)

    joined = " ".join(safe["folder_path"]) + " ".join(safe["folder_name"])
    for name in ("Antonia G", "VITO A", "VITO L", "Someone A"):
        assert name not in joined, f"{name} survived anonymisation"

    # Distinct directories stay distinct, and identical ones stay identical.
    assert safe["folder_path"].nunique() == meta["folder_path"].nunique()
    assert (
        safe.groupby("subject_id")["folder_path"].nunique()
        == meta.groupby("subject_id")["folder_path"].nunique()
    ).all()
    # The enclosing group directory is preserved so the tables stay readable.
    assert safe["folder_path"].str.startswith("28 People").any()


def test_committed_notebook_outputs_contain_no_real_names():
    """Guard the artefacts that actually get committed.

    Skips when the corpus is absent, since the names come from directory names.
    """
    root = Path(__file__).resolve().parents[1]
    corpus = root / "data" / "raw" / "ipvs"
    if not corpus.is_dir():
        pytest.skip("corpus not present")

    # Only leaf directories name a participant; the intermediate ones are
    # severity bins such as "6-10".
    names = {
        d.name
        for group in P.GROUP_DIRS
        for d in (corpus / group).rglob("*")
        if d.is_dir() and any(f.suffix.lower() == ".wav" for f in d.iterdir())
    }
    assert names, "no participant directories found"

    targets = list((root / "notebooks").glob("*.ipynb"))
    targets += list((root / "results").glob("*.csv"))
    offenders = []
    for path in targets:
        text = path.read_text(errors="ignore")
        for name in names:
            if name in text:
                offenders.append(f"{path.name}: {name!r}")
    assert not offenders, "real participant names in committed output:\n" + "\n".join(
        sorted(offenders)[:20]
    )


def test_redact_removes_every_identifying_column(tmp_path):
    meta, _ = P.build_metadata(_fake_corpus(tmp_path))
    public = P.redact(meta)
    for column in ("path", "folder_name", "folder_path", "subject_key",
                   "session_stamp", "session_date", "code_signature"):
        assert column not in public.columns, f"{column} survived redaction"
    assert "subject_id" in public.columns and "age" in public.columns


# --------------------------------------------------------------------------- #
# Split integrity.
# --------------------------------------------------------------------------- #

def _toy_metadata() -> pd.DataFrame:
    """Reproduces the corpus hazard: one participant filed under two folders."""
    rows = []
    for i in range(8):
        subject = f"PD{i:02d}"
        # PD00 was recorded twice and filed under two different directories.
        folders = ["PD/1-5/A", "PD/11-16/A"] if i == 0 else [f"PD/1-5/S{i}"]
        for session, folder in enumerate(folders):
            for rep in (1, 2):
                rows.append(
                    {
                        "subject_id": subject,
                        "folder_path": folder,
                        "label": 1,
                        "session_index": session + 1,
                        "task_rep": rep,
                    }
                )
    for i in range(8):
        subject = f"eHC{i:02d}"
        for rep in (1, 2):
            rows.append(
                {
                    "subject_id": subject,
                    "folder_path": f"eHC/{subject}",
                    "label": 0,
                    "session_index": 1,
                    "task_rep": rep,
                }
            )
    return pd.DataFrame(rows)


def test_arm_c_never_shares_a_participant():
    meta = _toy_metadata()
    y = meta["label"].to_numpy()
    for _, train_idx, test_idx in S.iter_splits(meta, y, "C_subject_key", n_repeats=3):
        assert not S.leaking_participants(meta, train_idx, test_idx)


def test_arm_a_does_share_participants():
    """If this ever passes, the experiment has lost its contrast."""
    meta = _toy_metadata()
    y = meta["label"].to_numpy()
    leaked = [
        len(S.leaking_participants(meta, tr, te))
        for _, tr, te in S.iter_splits(meta, y, "A_recording", n_repeats=3)
    ]
    assert max(leaked) > 0


def test_arm_b_leaks_only_the_two_session_participant():
    meta = _toy_metadata()
    y = meta["label"].to_numpy()
    leaked = set()
    for _, tr, te in S.iter_splits(meta, y, "B_folder_path", n_repeats=5):
        leaked |= S.leaking_participants(meta, tr, te)
    assert leaked <= {"PD00"}, f"unexpected leakage: {leaked}"


def test_every_recording_is_tested_exactly_once_per_repeat():
    meta = _toy_metadata()
    y = meta["label"].to_numpy()
    seen = {}
    for repeat, _, test_idx in S.iter_splits(meta, y, "C_subject_key", n_repeats=2):
        seen.setdefault(repeat, []).extend(test_idx.tolist())
    for repeat, indices in seen.items():
        assert sorted(indices) == list(range(len(meta))), f"repeat {repeat} incomplete"


# --------------------------------------------------------------------------- #
# Metric integrity.
# --------------------------------------------------------------------------- #

def test_aggregation_collapses_to_one_row_per_participant():
    ids = np.array(["a", "a", "b", "b", "c"])
    y = np.array([1, 1, 0, 0, 1])
    scores = np.array([0.9, 0.7, 0.2, 0.4, 0.8])
    out = M.aggregate_to_participant(ids, y, scores)
    assert len(out) == 3
    assert out.loc[out.subject_id == "a", "y_score"].iloc[0] == pytest.approx(0.8)


def test_aggregation_rejects_inconsistent_labels():
    with pytest.raises(ValueError):
        M.aggregate_to_participant(
            np.array(["a", "a"]), np.array([0, 1]), np.array([0.1, 0.2])
        )


def test_bootstrap_ci_widens_with_fewer_participants():
    rng = np.random.default_rng(0)

    def ci_width(n):
        y = np.r_[np.ones(n // 2), np.zeros(n // 2)].astype(int)
        s = np.r_[rng.normal(1, 1, n // 2), rng.normal(0, 1, n // 2)]
        _, lo, hi = M.bootstrap_auroc_ci(y, s, n_boot=400, seed=1)
        return hi - lo

    assert ci_width(40) > ci_width(400)


def test_safe_auroc_handles_single_class_resample():
    assert np.isnan(M.safe_auroc(np.ones(5, dtype=int), np.arange(5.0)))
