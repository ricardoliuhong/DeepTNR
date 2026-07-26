#!/usr/bin/env python3
"""Visium HD tumor / periphery / tissue domain assignment for DeepTNR.

Reproduces the Fig. 4 tumor-periphery logic from Oliveira et al.
(Nature Genetics 2025; HumanColonCancer_VisiumHD):
  Tumor bins from deconvolution labels,
  then non-tumor tissue bins within ``distance_um`` of qualified tumor seeds
  (seeds with at least ``min_tumor_neighbors`` tumor neighbors).

Callable like Interface.py::

    import VisiumHD_domain as vhd
    vhd.visiumhd_domain(
        'CRC1.h5ad',
        metadata_path='P1CRC_Metadata.parquet',
        spaceranger_root='HumanColonCancer_VisiumHD',
        output_h5ad_path='CRC1_VisiumHD_domain.h5ad',
    )
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import KDTree

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Path / Space Ranger helpers
# ---------------------------------------------------------------------------

def _as_path(path: Optional[PathLike]) -> Optional[Path]:
    if path is None or path == "":
        return None
    return Path(path)


def spaceranger_spatial_dir(path_sr: PathLike, bin_size_um: int = 8) -> Path:
    return Path(path_sr) / "binned_outputs" / f"square_{bin_size_um:03d}um" / "spatial"


def filter_to_bin_size(
    df: pd.DataFrame, barcode_col: str = "barcode", bin_size_um: int = 8
) -> pd.DataFrame:
    if barcode_col not in df.columns:
        return df.copy()
    token = f"s_{bin_size_um:03d}um"
    barcode = df[barcode_col].astype(str)
    mask = barcode.str.contains(token, regex=False)
    if mask.any():
        return df.loc[mask].copy()
    warnings.warn(f"No barcode contains {token!r}; keeping all rows.")
    return df.copy()


def normalize_tissue_mask(values) -> np.ndarray:
    s = pd.Series(values)
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float).gt(0).to_numpy()
    token = s.astype(str).str.strip().str.lower()
    return token.isin(["1", "1.0", "true", "t", "yes", "y"]).to_numpy()


def read_spot_diameter_fullres(
    path_sr: Optional[PathLike] = None,
    bin_size_um: int = 8,
    adata=None,
):
    if path_sr:
        candidate = spaceranger_spatial_dir(path_sr, bin_size_um) / "scalefactors_json.json"
        if candidate.exists():
            with open(candidate, "r") as f:
                scales = json.load(f)
            return float(scales["spot_diameter_fullres"]), str(candidate)

    if adata is not None and "spatial" in adata.uns:
        for library_id, library in adata.uns["spatial"].items():
            scales = library.get("scalefactors", {}) if isinstance(library, dict) else {}
            if "spot_diameter_fullres" in scales:
                return float(scales["spot_diameter_fullres"]), f"adata.uns['spatial'][{library_id!r}]"

    return None, None


def read_10x_tissue_positions(path_sr: PathLike, bin_size_um: int = 8):
    spatial_dir = spaceranger_spatial_dir(path_sr, bin_size_um)
    parquet_path = spatial_dir / "tissue_positions.parquet"
    csv_path = spatial_dir / "tissue_positions.csv"

    if parquet_path.exists():
        pos = pd.read_parquet(parquet_path)
        source = str(parquet_path)
    elif csv_path.exists():
        pos = pd.read_csv(csv_path)
        source = str(csv_path)
    else:
        return None, None

    rename_map = {
        "in_tissue": "tissue",
        "array_row": "row",
        "array_col": "col",
        "pxl_row_in_fullres": "imagerow",
        "pxl_col_in_fullres": "imagecol",
    }
    pos = pos.rename(columns=rename_map)
    required = {"barcode", "tissue", "row", "col", "imagerow", "imagecol"}
    if not required.issubset(pos.columns):
        pos = pos.iloc[:, :6].copy()
        pos.columns = ["barcode", "tissue", "row", "col", "imagerow", "imagecol"]

    pos = filter_to_bin_size(pos, barcode_col="barcode", bin_size_um=bin_size_um)
    pos["barcode"] = pos["barcode"].astype(str)
    pos.attrs["coordinate_unit"] = "fullres_px"
    return pos, source


def positions_from_adata(adata) -> pd.DataFrame:
    obs = adata.obs.copy()
    pos = pd.DataFrame({"barcode": adata.obs_names.astype(str)})

    if "in_tissue" in obs.columns:
        pos["tissue"] = obs["in_tissue"].to_numpy()
    elif "tissue" in obs.columns:
        pos["tissue"] = obs["tissue"].to_numpy()
    else:
        pos["tissue"] = 1

    has_array = {"array_row", "array_col"}.issubset(obs.columns)
    if has_array:
        pos["row"] = obs["array_row"].to_numpy()
        pos["col"] = obs["array_col"].to_numpy()

    if "spatial" in adata.obsm:
        spatial = np.asarray(adata.obsm["spatial"])
        pos["imagecol"] = spatial[:, 0]
        pos["imagerow"] = spatial[:, 1]
        pos.attrs["coordinate_unit"] = "fullres_px"
    elif has_array:
        pos["imagecol"] = pos["col"].astype(float)
        pos["imagerow"] = pos["row"].astype(float)
        pos.attrs["coordinate_unit"] = "array_bin"
    else:
        raise ValueError("No usable coordinates found in adata.")

    return pos


# ---------------------------------------------------------------------------
# Metadata / domain table construction
# ---------------------------------------------------------------------------

def _obs_with_barcode_column(adata) -> pd.DataFrame:
    """Return a copy of ``adata.obs`` with a unique ``barcode`` column.

    Keeps row order aligned to ``adata.obs_names`` so AnnData assignment
    does not rewrite observation names to a RangeIndex.
    """
    barcodes = pd.Index(adata.obs_names.astype(str))
    obs = adata.obs.copy()
    # Drop index labels that would make ``barcode`` ambiguous with a column.
    if isinstance(obs.index, pd.MultiIndex):
        obs = obs.reset_index(drop=True)
    else:
        obs.index = pd.RangeIndex(len(obs))
    if "barcode" in obs.columns:
        obs = obs.drop(columns=["barcode"])
    obs["barcode"] = barcodes.to_numpy()
    obs.index = barcodes
    obs.index.name = None
    return obs


def merge_metadata_into_adata(adata, metadata_path: Optional[Path], bin_size_um: int = 8):
    if metadata_path is None or not metadata_path.exists():
        if metadata_path is not None:
            warnings.warn(
                f"Metadata file not found: {metadata_path}. Continuing with adata.obs only."
            )
        adata.obs = _obs_with_barcode_column(adata)
        return adata, None

    meta = pd.read_parquet(metadata_path)
    meta = filter_to_bin_size(meta, barcode_col="barcode", bin_size_um=bin_size_um)
    meta["barcode"] = meta["barcode"].astype(str)
    meta = meta.drop_duplicates(subset="barcode", keep="first")

    obs = _obs_with_barcode_column(adata)
    overlap = [c for c in meta.columns if c in obs.columns and c != "barcode"]
    if overlap:
        obs = obs.drop(columns=overlap)
    adata.obs = obs.join(meta.set_index("barcode"), on="barcode", how="left")
    return adata, meta


def build_domain_input_table(adata, path_sr: Optional[Path], bin_size_um: int) -> pd.DataFrame:
    pos, pos_source = (None, None)
    if path_sr is not None and spaceranger_spatial_dir(path_sr, bin_size_um).exists():
        pos, pos_source = read_10x_tissue_positions(path_sr, bin_size_um=bin_size_um)

    if pos is None:
        pos = positions_from_adata(adata)
        pos_source = "adata"

    annot_cols = [
        "barcode",
        "DeconvolutionClass",
        "DeconvolutionLabel1",
        "DeconvolutionLabel2",
        "MacrophageSubtype",
        "GobletSubcluster",
        "UnsupervisedL1",
        "UnsupervisedL2",
        "Periphery",
    ]
    annot = _obs_with_barcode_column(adata)
    keep = [c for c in annot_cols if c in annot.columns]
    annot = annot[keep].drop_duplicates(subset="barcode", keep="first")

    df = pos.merge(annot, on="barcode", how="left")
    df.attrs["coordinate_unit"] = pos.attrs.get("coordinate_unit", "fullres_px")
    df.attrs["position_source"] = pos_source
    return df


def infer_tumor_mask(
    df: pd.DataFrame,
    tumor_label: Optional[str],
    label_col: str,
    fallback_periphery: bool = True,
):
    tissue = normalize_tissue_mask(df["tissue"])

    if tumor_label:
        if label_col not in df.columns:
            raise KeyError(f"{label_col!r} is missing; cannot use tumor_label.")
        tumor = tissue & df[label_col].astype(str).eq(str(tumor_label)).to_numpy()
        return tumor, f"{label_col} == {tumor_label!r}"

    if label_col in df.columns:
        labels = df.loc[tissue, label_col].dropna().astype(str)
        counts = labels.value_counts()
        pattern = re.compile(r"tumou?r|cancer|malignant|carcinoma|neoplastic", re.IGNORECASE)
        candidates = [label for label in counts.index if pattern.search(label)]
        if candidates:
            chosen = candidates[0]
            tumor = tissue & df[label_col].astype(str).eq(str(chosen)).to_numpy()
            return tumor, f"auto: {label_col} == {chosen!r}"

    if fallback_periphery and "Periphery" in df.columns:
        tumor = tissue & df["Periphery"].astype(str).eq("Tumor").to_numpy()
        if tumor.sum() > 0:
            return tumor, "fallback: Periphery == 'Tumor'"

    top = (
        df[label_col].value_counts().head(30)
        if label_col in df.columns
        else "label column missing"
    )
    raise ValueError(
        "Could not infer tumor bins. Set tumor_label to the tumor class.\n"
        f"Top labels:\n{top}"
    )


def assign_tumor_periphery_domain(
    df: pd.DataFrame,
    tumor_mask: np.ndarray,
    spot_diameter_fullres: Optional[float],
    distance_um: float,
    bin_size_um: int,
    min_tumor_neighbors: int,
    output_col: str,
    coordinate_unit: str,
) -> pd.DataFrame:
    out = df.copy()

    if coordinate_unit == "fullres_px":
        if spot_diameter_fullres is None:
            if {"row", "col"}.issubset(out.columns):
                warnings.warn(
                    "No spot_diameter_fullres found; falling back to array row/col coordinates."
                )
                out["imagecol"] = out["col"].astype(float)
                out["imagerow"] = out["row"].astype(float)
                coordinate_unit = "array_bin"
            else:
                raise ValueError("spot_diameter_fullres is required for fullres pixel coordinates.")
        else:
            radius_coord = distance_um * spot_diameter_fullres / bin_size_um

    if coordinate_unit == "array_bin":
        radius_coord = distance_um / bin_size_um

    if coordinate_unit not in {"fullres_px", "array_bin"}:
        raise ValueError(f"Unsupported coordinate_unit={coordinate_unit!r}")

    coords = out[["imagecol", "imagerow"]].to_numpy(dtype=float)
    valid_coords = np.isfinite(coords).all(axis=1)
    tissue = normalize_tissue_mask(out["tissue"]) & valid_coords
    tumor = np.asarray(tumor_mask, dtype=bool) & tissue
    tumor_coords = coords[tumor]

    if tumor_coords.shape[0] == 0:
        raise ValueError("No tumor bins found. Check tumor_label or metadata.")

    tumor_tree = KDTree(tumor_coords)
    tumor_neighbor_counts = tumor_tree.query_radius(
        tumor_coords, r=radius_coord, count_only=True
    )
    seed_mask = tumor_neighbor_counts >= min_tumor_neighbors
    seed_coords = tumor_coords[seed_mask]

    if seed_coords.shape[0] == 0:
        raise ValueError(
            f"No qualified tumor seeds found with min_tumor_neighbors={min_tumor_neighbors}. "
            "Lower min_tumor_neighbors or inspect tumor-label sparsity."
        )

    seed_tree = KDTree(seed_coords)
    dist_coord = np.full(len(out), np.nan, dtype=float)
    dist_valid, _ = seed_tree.query(coords[valid_coords], k=1)
    dist_coord[valid_coords] = dist_valid[:, 0]

    if coordinate_unit == "fullres_px":
        dist_um = dist_coord * bin_size_um / spot_diameter_fullres
    else:
        dist_um = dist_coord * bin_size_um

    near_tumor = dist_coord < radius_coord
    domain = np.full(len(out), pd.NA, dtype=object)
    domain[tissue] = "Tissue"
    domain[tissue & near_tumor & ~tumor] = f"{distance_um:g} micron"
    domain[tumor] = "Tumor"

    out[output_col] = domain
    out["is_tumor_py"] = tumor
    out["is_periphery_py"] = tissue & near_tumor & ~tumor
    out["distance_to_tumor_um"] = dist_um
    out["domain_radius_coord"] = radius_coord
    out["domain_coordinate_unit"] = coordinate_unit
    out.attrs["n_tumor_bins"] = int(tumor.sum())
    out.attrs["n_qualified_tumor_seeds"] = int(seed_coords.shape[0])
    return out


def attach_domain_to_adata(adata, domain_df: pd.DataFrame, domain_col: str):
    keep = [
        "barcode",
        domain_col,
        "is_tumor_py",
        "is_periphery_py",
        "distance_to_tumor_um",
        "domain_radius_coord",
        "domain_coordinate_unit",
    ]
    add = domain_df[keep].drop_duplicates(subset="barcode", keep="first").set_index("barcode")

    obs = _obs_with_barcode_column(adata)
    overlap = [c for c in add.columns if c in obs.columns]
    if overlap:
        obs = obs.drop(columns=overlap)
    adata.obs = obs.join(add, on="barcode", how="left")
    return adata


# ---------------------------------------------------------------------------
# Public API (Interface.py style)
# ---------------------------------------------------------------------------

def annotate_visiumhd_domain(
    adata,
    metadata_path: Optional[PathLike] = None,
    spaceranger_root: Optional[PathLike] = None,
    bin_size_um: int = 8,
    distance_um: float = 50.0,
    min_tumor_neighbors: int = 25,
    tumor_label: Optional[str] = None,
    label_col: str = "DeconvolutionLabel1",
    output_col: str = "VisiumHD_domain_py",
    sample_id: str = "sample",
):
    """Annotate an AnnData object with Visium HD tumor-periphery domains.

    Writes ``output_col`` (default ``VisiumHD_domain_py``) into ``adata.obs``
    with values: Tumor / ``{distance_um} micron`` / Tissue / NA.
    """
    metadata_path = _as_path(metadata_path)
    spaceranger_root = _as_path(spaceranger_root)

    adata.var_names_make_unique()
    adata, _ = merge_metadata_into_adata(adata, metadata_path, bin_size_um)

    if spaceranger_root is None or not spaceranger_spatial_dir(
        spaceranger_root, bin_size_um
    ).exists():
        if spaceranger_root is not None:
            warnings.warn(
                "No valid Space Ranger spatial directory; using AnnData coordinates."
            )
        spaceranger_root = None

    domain_input = build_domain_input_table(adata, spaceranger_root, bin_size_um)
    spot_diameter_fullres, spot_source = read_spot_diameter_fullres(
        spaceranger_root, bin_size_um, adata
    )

    tumor_mask, tumor_source = infer_tumor_mask(
        domain_input,
        tumor_label=tumor_label,
        label_col=label_col,
        fallback_periphery=True,
    )

    domain_df = assign_tumor_periphery_domain(
        df=domain_input,
        tumor_mask=tumor_mask,
        spot_diameter_fullres=spot_diameter_fullres,
        distance_um=distance_um,
        bin_size_um=bin_size_um,
        min_tumor_neighbors=min_tumor_neighbors,
        output_col=output_col,
        coordinate_unit=domain_input.attrs.get("coordinate_unit", "fullres_px"),
    )

    adata = attach_domain_to_adata(adata, domain_df, output_col)
    adata.uns["VisiumHD_domain_py"] = {
        "sample_id": sample_id,
        "bin_size_um": bin_size_um,
        "distance_um": distance_um,
        "min_tumor_neighbors": min_tumor_neighbors,
        "tumor_seed_source": tumor_source,
        "position_source": str(domain_input.attrs.get("position_source")),
        "coordinate_unit": str(domain_df["domain_coordinate_unit"].dropna().iloc[0]),
        "spot_diameter_fullres": (
            None if spot_diameter_fullres is None else float(spot_diameter_fullres)
        ),
        "spot_diameter_source": spot_source,
        "n_tumor_bins": domain_df.attrs.get("n_tumor_bins"),
        "n_qualified_tumor_seeds": domain_df.attrs.get("n_qualified_tumor_seeds"),
    }
    return adata, domain_df


def visiumhd_domain(
    h5ad_path: PathLike,
    metadata_path: Optional[PathLike] = None,
    spaceranger_root: Optional[PathLike] = None,
    output_h5ad_path: Optional[PathLike] = None,
    bin_size_um: int = 8,
    distance_um: float = 50.0,
    min_tumor_neighbors: int = 25,
    tumor_label: Optional[str] = None,
    label_col: str = "DeconvolutionLabel1",
    output_col: str = "VisiumHD_domain_py",
    sample_id: Optional[str] = None,
    outdir: Optional[PathLike] = None,
    write_tables: bool = True,
):
    """Path-in / path-out entry point (mirrors Interface.Interface).

    Parameters
    ----------
    h5ad_path :
        Visium HD AnnData (typically 8 um bins).
    metadata_path :
        Optional parquet with DeconvolutionLabel1 / Periphery (e.g. P1CRC_Metadata.parquet).
    spaceranger_root :
        Optional Space Ranger root containing
        ``binned_outputs/square_008um/spatial/``.
    output_h5ad_path :
        If set, write annotated AnnData to this path.
    """
    h5ad_path = Path(h5ad_path)
    if sample_id is None:
        sample_id = h5ad_path.stem

    adata = sc.read_h5ad(h5ad_path)
    adata, domain_df = annotate_visiumhd_domain(
        adata,
        metadata_path=metadata_path,
        spaceranger_root=spaceranger_root,
        bin_size_um=bin_size_um,
        distance_um=distance_um,
        min_tumor_neighbors=min_tumor_neighbors,
        tumor_label=tumor_label,
        label_col=label_col,
        output_col=output_col,
        sample_id=sample_id,
    )

    if outdir is not None and write_tables:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        domain_csv = outdir / f"{sample_id}_VisiumHD_domain_py.csv"
        summary_csv = outdir / f"{sample_id}_VisiumHD_domain_summary.csv"
        domain_df.to_csv(domain_csv, index=False)
        domain_df[output_col].value_counts(dropna=False).rename_axis("domain").reset_index(
            name="n_bins"
        ).to_csv(summary_csv, index=False)

        if "Periphery" in domain_df.columns:
            compare = domain_df[["Periphery", output_col]].copy()
            compare["Periphery"] = compare["Periphery"].astype("string")
            compare[output_col] = compare[output_col].astype("string")
            ctab = pd.crosstab(compare["Periphery"], compare[output_col], dropna=False)
            ctab.to_csv(outdir / f"{sample_id}_Periphery_vs_domain_py_crosstab.csv")

    if output_h5ad_path is not None:
        Path(output_h5ad_path).parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(output_h5ad_path)

    return adata


# Keep a PascalCase alias consistent with Interface.Interface naming style.
VisiumHD_domain = visiumhd_domain


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign Visium HD Tumor / periphery / Tissue domains "
            "(Oliveira et al. Nat Genet 2025 Fig. 4 logic)."
        )
    )
    parser.add_argument("--adata", required=True, help="Input Visium HD h5ad.")
    parser.add_argument(
        "--metadata",
        default="",
        help="Optional metadata parquet (DeconvolutionLabel1 / Periphery).",
    )
    parser.add_argument(
        "--spaceranger-root",
        default="",
        help="Space Ranger root with binned_outputs/square_XXXum/spatial.",
    )
    parser.add_argument(
        "--output-h5ad",
        default="",
        help="Output annotated h5ad path.",
    )
    parser.add_argument("--outdir", default="", help="Optional directory for CSV tables.")
    parser.add_argument("--sample-id", default="")
    parser.add_argument("--bin-size-um", type=int, default=8)
    parser.add_argument("--distance-um", type=float, default=50.0)
    parser.add_argument("--min-tumor-neighbors", type=int, default=25)
    parser.add_argument("--tumor-label", default="")
    parser.add_argument("--label-col", default="DeconvolutionLabel1")
    parser.add_argument("--output-col", default="VisiumHD_domain_py")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sample_id = args.sample_id or Path(args.adata).stem
    output_h5ad = args.output_h5ad or f"{sample_id}_VisiumHD_domain.h5ad"

    adata = visiumhd_domain(
        h5ad_path=args.adata,
        metadata_path=args.metadata or None,
        spaceranger_root=args.spaceranger_root or None,
        output_h5ad_path=output_h5ad,
        bin_size_um=args.bin_size_um,
        distance_um=args.distance_um,
        min_tumor_neighbors=args.min_tumor_neighbors,
        tumor_label=args.tumor_label or None,
        label_col=args.label_col,
        output_col=args.output_col,
        sample_id=sample_id,
        outdir=args.outdir or None,
        write_tables=bool(args.outdir),
    )
    print("Domain counts:")
    print(adata.obs[args.output_col].value_counts(dropna=False).to_string())
    print("wrote:", output_h5ad)
    print("DONE")


if __name__ == "__main__":
    main()
