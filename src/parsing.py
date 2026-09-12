"""Parse the IPVS corpus layout into a tidy, pseudonymised metadata table.

IPVS ships one directory per participant, named with the participant's real
first name and surname initial. The recording filenames encode the task, a
scrambled form of the name, the two-digit birth year, sex, and the recording
timestamp, e.g.::

    B1APGANRET55F170320171104.wav
    ^^ task     ^^ birth year (1955)
      ^^^^^^^^ scrambled name   ^ sex
                        ^^^^^^^^^^^^ ddmmyyyy + hhmm

Neither the directory name nor the scrambled code is a safe participant key on
its own; see ``build_metadata`` and ``docs/DATA_QUALITY.md`` for the specific
collisions found in this corpus.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Recording year for every session in this corpus release is 2016 or 2017; the
# two-digit birth year is unambiguous because no participant was born after 1999.
_RECORDING_YEAR_FALLBACK = 2017

GROUP_DIRS = {
    "28 People with Parkinson's disease": "PD",
    "22 Elderly Healthy Control": "eHC",
    "15 Young Healthy Control": "yHC",
}

# `[A-Za-z_ ]` rather than `[A-Za-z]`: one participant's code contains an
# underscore (`ubguot_t`) and another contains a space (`VLIAT OP`).
# `\d+` rather than `\d` for the task index: one file is named `PR11...`.
FILENAME_RE = re.compile(
    r"^(?P<task_base>[A-Z]+)"
    r"(?P<task_rep>\d+)"
    r"(?P<code>[A-Za-z_ ]+?)"
    r"(?P<birth_year>\d{2})"
    r"(?P<sex>[MF])"
    r"(?P<stamp>\d+)"
    r"\.+wav$",
    re.IGNORECASE,
)

TASK_LABELS = {
    "B": "read_text",       # brano: phonetically balanced passage, read twice
    "D": "ddk",             # diadochokinetic /pa/ /ta/
    "VA": "vowel_a",
    "VE": "vowel_e",
    "VI": "vowel_i",
    "VO": "vowel_o",
    "VU": "vowel_u",
    "PR": "words",
    "FB": "phrases",
}


@dataclass(frozen=True)
class ParsedName:
    task_base: str
    task_rep: int
    code: str
    birth_year: int
    sex: str
    stamp: str

    @property
    def code_signature(self) -> str:
        """Order-insensitive signature of the scrambled name code.

        The scrambling is not stable across sessions: the same participant
        appears as ``rriovbie`` in one session and ``RROIBVEI`` in another, so
        the letters must be sorted before comparing.
        """
        letters = [c for c in self.code.lower() if c.isalpha()]
        return "".join(sorted(letters))

    @property
    def subject_key(self) -> str:
        """Strict key including the birth year, kept for provenance checks."""
        return f"{self.code_signature}|{self.birth_year:02d}|{self.sex}"

    def participant_key(self, birth_year: int) -> str:
        """Grouping key, using a birth year resolved at directory level.

        The birth year cannot be dropped from the key: ``VAILTFOO49M`` and
        ``VLIATFOO55M`` are two different elderly controls whose scrambled codes
        are anagrams of each other, so the year is the only thing separating
        them. It also cannot be trusted as typed: fifteen files in one session
        read ``AGNUTGOL52F`` and one reads ``AGNUTGOL63F``.

        ``build_metadata`` therefore resolves the year by majority vote *within a
        directory* — a typo is a minority variant inside one session, whereas two
        different people occupy two different directories.
        """
        return f"{self.code_signature}|{birth_year:02d}|{self.sex}"

    @property
    def age(self) -> int:
        return _RECORDING_YEAR_FALLBACK - (1900 + self.birth_year)

    @property
    def recording_year(self) -> int:
        date = self.session_date
        return int(date[4:8]) if len(date) == 8 else 2000 + int(date[4:6])

    @property
    def session_date(self) -> str:
        """Date part of the timestamp, dropping the trailing ``hhmm``.

        The stamp is either ``ddmmyy`` or ``ddmmyyyy`` followed by ``hhmm``, so
        the date is everything except the last four digits. Ranking on the full
        stamp instead would make every recording its own session, since the
        two readings of the passage are minutes apart.
        """
        return self.stamp[:-4] if len(self.stamp) > 4 else self.stamp


def parse_filename(name: str) -> ParsedName | None:
    """Return the decoded fields, or ``None`` if the name does not match."""
    match = FILENAME_RE.match(name)
    if match is None:
        return None
    parts = match.groupdict()
    return ParsedName(
        task_base=parts["task_base"].upper(),
        task_rep=int(parts["task_rep"]),
        code=parts["code"],
        birth_year=int(parts["birth_year"]),
        sex=parts["sex"].upper(),
        stamp=parts["stamp"],
    )


def _pseudonym(group: str, subject_key: str, index: int) -> str:
    return f"{group}{index:02d}"


def build_metadata(ipvs_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    """Walk the corpus and return ``(metadata, unparsed_files)``.

    The returned frame carries three competing participant keys so that the
    downstream experiment can quantify what each choice costs:

    ``folder_path``
        Full relative directory. Looks like a participant key but splits the
        three participants who were recorded in two sessions and filed under
        two different severity bins.
    ``folder_name``
        Directory basename. Merges participants who share a first name and
        surname initial.
    ``subject_key``
        Sorted name-code signature plus birth year and sex. The only key that
        survives both checks.
    """
    ipvs_root = Path(ipvs_root)
    records: list[dict] = []
    unparsed: list[Path] = []

    for group_dir, group in GROUP_DIRS.items():
        base = ipvs_root / group_dir
        if not base.is_dir():
            raise FileNotFoundError(f"missing group directory: {base}")
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() != ".wav":
                continue
            parsed = parse_filename(path.name)
            if parsed is None:
                unparsed.append(path)
                continue
            records.append(
                {
                    "group": group,
                    "label": int(group == "PD"),
                    "task_base": parsed.task_base,
                    "task": TASK_LABELS.get(parsed.task_base, parsed.task_base.lower()),
                    "task_rep": parsed.task_rep,
                    "session_stamp": parsed.stamp,
                    "session_date": parsed.session_date,
                    "recording_year": parsed.recording_year,
                    "birth_year_raw": parsed.birth_year,
                    "sex": parsed.sex,
                    "strict_key": parsed.subject_key,
                    "code_signature": parsed.code_signature,
                    "folder_name": path.parent.name,
                    "folder_path": str(path.parent.relative_to(ipvs_root)),
                    "path": str(path),
                }
            )

    if not records:
        raise RuntimeError(f"no parsable .wav files under {ipvs_root}")

    meta = pd.DataFrame.from_records(records)

    # Resolve the hand-typed birth year to the majority value within each
    # directory, then key on it. See ParsedName.participant_key for why the year
    # is neither dropped nor trusted as typed.
    resolved = (
        meta.groupby(["folder_path", "code_signature", "sex"])["birth_year_raw"]
        .agg(lambda s: int(s.mode().iat[0]))
        .rename("birth_year")
    )
    meta = meta.join(resolved, on=["folder_path", "code_signature", "sex"])
    meta["birth_year_typo"] = meta["birth_year_raw"] != meta["birth_year"]
    meta["age"] = _RECORDING_YEAR_FALLBACK - (1900 + meta["birth_year"])
    meta["subject_key"] = (
        meta["code_signature"]
        + "|"
        + meta["birth_year"].map("{:02d}".format)
        + "|"
        + meta["sex"]
    )

    # Stable pseudonyms: ordered by a hash of the key so the mapping does not
    # depend on filesystem ordering, and never leaks the alphabetical position
    # of the real name.
    for group, chunk in meta.groupby("group", sort=True):
        keys = sorted(
            chunk["subject_key"].unique(),
            key=lambda k: hashlib.sha256(k.encode()).hexdigest(),
        )
        mapping = {k: _pseudonym(group, k, i + 1) for i, k in enumerate(keys)}
        meta.loc[meta["group"] == group, "subject_id"] = meta.loc[
            meta["group"] == group, "subject_key"
        ].map(mapping)

    # Session index within participant, so a two-session participant is visible
    # without exposing the recording date.
    meta["session_index"] = (
        meta.groupby("subject_id")["session_date"].rank(method="dense").astype(int)
    )
    meta = meta.sort_values(["group", "subject_id", "task", "task_rep"]).reset_index(
        drop=True
    )
    return meta, unparsed


def add_audio_properties(meta: pd.DataFrame) -> pd.DataFrame:
    """Attach container properties read from the file headers.

    ``samplerate`` is metadata, not signal: it is available before a single audio
    sample is decoded, so if it correlates with the label then the label is
    partly predictable from the file header alone. In this corpus it does.
    """
    import soundfile as sf

    meta = meta.copy()
    props = [sf.info(path) for path in meta["path"]]
    meta["samplerate"] = [p.samplerate for p in props]
    meta["channels"] = [p.channels for p in props]
    meta["subtype"] = [p.subtype for p in props]
    meta["duration"] = [p.frames / p.samplerate for p in props]
    meta["batch"] = meta["recording_year"].map(lambda y: f"{y}_batch")
    return meta


def batch_confound_report(meta: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulate acquisition batch against label."""
    return (
        meta.drop_duplicates(["subject_id", "batch"])
        .pivot_table(
            index="batch",
            columns="group",
            values="subject_id",
            aggfunc="nunique",
            fill_value=0,
        )
    )


def key_collisions(meta: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Report where the three candidate participant keys disagree."""
    split_by_path = (
        meta.groupby("subject_key")["folder_path"]
        .nunique()
        .loc[lambda s: s > 1]
        .rename("n_folder_paths")
        .reset_index()
    )
    merged_by_name = (
        meta.groupby("folder_name")["subject_key"]
        .nunique()
        .loc[lambda s: s > 1]
        .rename("n_subject_keys")
        .reset_index()
    )
    typos = meta.loc[
        meta["birth_year_typo"],
        ["group", "subject_id", "folder_name", "task", "task_rep",
         "birth_year_raw", "birth_year"],
    ]
    return {
        "one_person_split_across_folders": split_by_path,
        "two_people_sharing_folder_name": merged_by_name,
        "birth_year_typos": typos,
    }


def anonymise_directories(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace directory names and paths with stable anonymous tokens.

    Used for anything that gets printed or committed. The corpus itself publishes
    the participants' names, but restating them next to a diagnosis in a notebook
    output or a figure adds a linkage this project has no reason to create. The
    tokens keep the structural facts — which directories collide, which are
    nested — while carrying no name.
    """
    frame = frame.copy()
    for column in ("folder_path", "folder_name"):
        if column not in frame.columns:
            continue
        tokens = {}
        for value in sorted(frame[column].dropna().unique()):
            digest = hashlib.sha256(str(value).encode()).hexdigest()[:6]
            # Preserve the enclosing structure (group, severity bin), drop the name.
            parts = str(value).split("/")
            prefix = "/".join(parts[:-1])
            tokens[value] = f"{prefix}/dir-{digest}" if prefix else f"dir-{digest}"
        frame[column] = frame[column].map(tokens)
    return frame


def redact(meta: pd.DataFrame) -> pd.DataFrame:
    """Drop every column that can point back to a named individual."""
    drop = ["subject_key", "strict_key", "code_signature", "folder_name",
            "folder_path", "path", "session_stamp", "session_date",
            "birth_year_raw", "birth_year"]
    return meta.drop(columns=[c for c in drop if c in meta.columns])
