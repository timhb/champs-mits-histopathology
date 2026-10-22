"""Inventory the whole-slide image store.

One row per slide file: ``slide_id`` (file name without extension),
``wsi_path`` (relative to the data root), ``site`` (first directory),
``case_id`` (the study id of the case directory after ``Cases``),
``study_id_in_name`` (the study id embedded in the file name, if any),
``id_conflict`` (both exist and differ), and the scanner metadata read from
the file: ``objective_power``, ``mpp_x``, and ``scanner_power`` (the objective
power, or 40 when the resolution is finer than 0.35 microns per pixel and 20
otherwise). ``probe_error`` names the reason when a file cannot be opened.
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from champs_pipeline.data_prep.identifiers import derive_case_id, study_id_in_name

EXTENSIONS = (".svs", ".ndpi", ".tif", ".tiff", ".mrxs", ".scn", ".bif")
EXCLUDE_DIRS = {"derived", "embeddings", "hf-cache", "reports", "level-2-deidentified-data"}
MPP_40X_BELOW = 0.35


def list_slides(root):
    """Absolute paths of every slide file under the root, sorted; excluded directories skipped."""
    paths = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for name in files:
            if name.lower().endswith(EXTENSIONS):
                paths.append(os.path.join(dirpath, name))
    return sorted(paths)


def scanner_power(objective_power, mpp):
    """The objective power, or the magnification implied by the resolution."""
    if objective_power is not None:
        return objective_power
    if mpp is None:
        return None
    return 40.0 if mpp < MPP_40X_BELOW else 20.0


def probe(path):
    """Open the file with OpenSlide and read its scan properties; record the error if it fails."""
    import openslide

    try:
        slide = openslide.OpenSlide(path)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {str(exc)[:80]}"
        return {"objective_power": None, "mpp_x": None, "scanner_power": None, "probe_error": error}
    try:
        power = slide.properties.get("openslide.objective-power")
        mpp = slide.properties.get("openslide.mpp-x")
    finally:
        slide.close()
    power = float(power) if power else None
    mpp = float(mpp) if mpp else None
    return {
        "objective_power": power,
        "mpp_x": mpp,
        "scanner_power": scanner_power(power, mpp),
        "probe_error": None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    root = os.path.abspath(args.data_root)
    paths = list_slides(root)
    relative = [os.path.relpath(p, root) for p in paths]

    # Identity of each slide: from the file name and from the case directory.
    rows = pd.DataFrame({
        "slide_id": [os.path.splitext(os.path.basename(p))[0] for p in paths],
        "wsi_path": relative,
        "site": [p.split(os.sep)[0] for p in relative],
        "case_id": [derive_case_id(p) for p in paths],
    })
    rows["study_id_in_name"] = rows["slide_id"].map(study_id_in_name)
    named = rows["study_id_in_name"].notna()
    rows["id_conflict"] = named & (rows["study_id_in_name"] != rows["case_id"])

    # Scanner metadata, read from every file.
    with ThreadPoolExecutor(args.workers) as pool:
        probes = list(pool.map(probe, paths))
    rows = pd.concat([rows, pd.DataFrame(probes)], axis=1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows.to_csv(args.out, index=False)
    print(f"{len(rows)} files, {rows['slide_id'].nunique()} slide ids")
    print(f"unreadable: {int(rows['probe_error'].notna().sum())}")
    print(f"study id in the name differs from the case directory: {int(rows['id_conflict'].sum())}")
    print(f"scanner power: {rows['scanner_power'].value_counts(dropna=False).to_dict()}")


if __name__ == "__main__":
    main()
