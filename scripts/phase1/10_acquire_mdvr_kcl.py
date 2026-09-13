# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 10 — Acquire MDVR-KCL
#
# Download and verify the Mobile Device Voice Recordings at King's College London (MDVR-KCL)
# corpus from Zenodo (DOI: 10.5281/zenodo.2867216).
#
# **Input:** Zenodo download endpoint (https://zenodo.org/api/records/2867216/files/26_29_09_2017_KCL.zip/content).
# **Output:** Extracted audio files in `data/raw/mdvr_kcl/` and SHA-256 provenance manifest.

# %%
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

# Walk up to the repository root
ROOT = Path.cwd().resolve()
while not (ROOT / "requirements.txt").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src import corpora as C  # noqa: E402

MDVR_DIR = ROOT / "data" / "raw" / "mdvr_kcl"
MDVR_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = MDVR_DIR / "26_29_09_2017_KCL.zip"
ZENODO_URL = "https://zenodo.org/api/records/2867216/files/26_29_09_2017_KCL.zip/content"

print(f"MDVR-KCL target directory: {MDVR_DIR}")

# %% [markdown]
# ## Download archive if not present

# %%
wav_files = list(MDVR_DIR.rglob("*.wav"))
if not wav_files:
    if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size < 500_000_000:
        print(f"Downloading {ZIP_PATH.name} from Zenodo...")
        
        def reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            mb = count * block_size / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            if count % 2000 == 0 or count * block_size >= total_size:
                sys.stdout.write(f"\r  Downloaded {mb:.1f}/{total_mb:.1f} MB ({percent}%)")
                sys.stdout.flush()

        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", "Mozilla/5.0 (Python research script)")]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(ZENODO_URL, ZIP_PATH, reporthook=reporthook)
        print("\nDownload complete.")

    print(f"Extracting {ZIP_PATH.name}...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(MDVR_DIR)
    print("Extraction complete.")
else:
    print(f"Found {len(wav_files)} existing .wav files. Skipping download.")

# %% [markdown]
# ## Compute SHA-256 Checksum & File Manifest

# %%
if ZIP_PATH.exists():
    print(f"Calculating SHA-256 for {ZIP_PATH.name}...")
    h = hashlib.sha256()
    with open(ZIP_PATH, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            h.update(chunk)
    zip_sha256 = h.hexdigest()
    print(f"  SHA-256 ({ZIP_PATH.name}): {zip_sha256}")
else:
    zip_sha256 = "already_extracted"

# %% [markdown]
# ## Parse filenames and verify clinical metadata

# %%
df, unparsed = C.load_mdvr_kcl(MDVR_DIR)
print(f"\nLoaded {len(df)} recordings from MDVR-KCL")
print(f"Unparsed files: {len(unparsed)}")
if unparsed:
    for p in unparsed:
        print("  UNPARSED:", p)
    raise AssertionError("Unparsed .wav files found in MDVR-KCL")

print("\nBreakdown by task and group:")
summary = df.pivot_table(index="group", columns="task", values="subject_id", aggfunc="count", fill_value=0)
print(summary.to_string())

print("\nParticipants:")
parts = df.drop_duplicates("subject_id")
print(f"  Total unique subjects: {len(parts)} ({sum(parts.group == 'PD')} PD, {sum(parts.group == 'HC')} HC)")

print("\nSample records with clinical scores (Hoehn & Yahr, UPDRS II-5, UPDRS III-18):")
print(df[["subject_id", "group", "task", "severity_hy", "severity_u2", "severity_u3"]].head(10).to_string())

print("\nStage 10 completed successfully.")
