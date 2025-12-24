import argparse
import os
import subprocess
from typing import Dict, Tuple

import numpy as np
import SimpleITK as sitk
try:
    import pydicom
except Exception:
    pydicom = None
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    # TPTBox provides vertebra naming utilities
    from TPTBox import v_name2idx  # type: ignore
except Exception:
    v_name2idx = None


def _find_best_series_directory(dicom_root: str) -> Tuple[str, str]:
    """Recursively search for the directory and series ID with the most files."""
    reader = sitk.ImageSeriesReader()
    best_dir = ""
    best_series_id = ""
    best_count = -1

    for current_dir, _, _ in os.walk(dicom_root):
        try:
            series_ids = reader.GetGDCMSeriesIDs(current_dir)
        except Exception:
            continue
        if not series_ids:
            continue
        for sid in series_ids:
            try:
                files = reader.GetGDCMSeriesFileNames(current_dir, sid)
            except Exception:
                continue
            count = len(files)
            if count > best_count:
                best_count = count
                best_dir = current_dir
                best_series_id = sid

    if best_count <= 0:
        raise RuntimeError(f"No DICOM series found (recursively) under {dicom_root}")
    return best_dir, best_series_id


def convert_dicom_to_nifti(dicom_dir: str, out_nifti_path: str) -> Tuple[str, Tuple[float, float, float]]:

    # First try SimpleITK recursive series discovery
    try:
        series_dir, series_id = _find_best_series_directory(dicom_dir)
        reader = sitk.ImageSeriesReader()
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        series_files = reader.GetGDCMSeriesFileNames(series_dir, series_id)
        reader.SetFileNames(series_files)
        image = reader.Execute()
        sitk.WriteImage(image, out_nifti_path)
        spacing = tuple(image.GetSpacing())
        return out_nifti_path, spacing
    except Exception:
        pass

    # Fallback: manual stack using pydicom if available
    if pydicom is None:
        raise RuntimeError(
            "No DICOM series found via SimpleITK and pydicom is not installed for fallback. Install pydicom: pip install pydicom"
        )

    dicom_files = [
        os.path.join(dicom_dir, f)
        for f in os.listdir(dicom_dir)
        if f.lower().endswith(".dcm") and os.path.isfile(os.path.join(dicom_dir, f))
    ]
    if not dicom_files:
        raise RuntimeError(f"No DICOM files (*.dcm) found in {dicom_dir}")

    # Read headers and sort slices
    datasets = []
    for fp in dicom_files:
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=False, force=True)
            if not ("PixelData" in ds and getattr(ds, "Rows", None) and getattr(ds, "Columns", None)):
                continue
            datasets.append((fp, ds))
        except Exception:
            continue
    if not datasets:
        raise RuntimeError(f"Could not parse any DICOM slices in {dicom_dir}")

    def sort_key(item):
        _fp, ds = item
        ipp = getattr(ds, "ImagePositionPatient", None)
        inst = getattr(ds, "InstanceNumber", None)
        if ipp is not None and len(ipp) == 3:
            return float(ipp[2])
        if inst is not None:
            return float(inst)
        return 0.0

    datasets.sort(key=sort_key)
    first = datasets[0][1]
    rows = int(first.Rows)
    cols = int(first.Columns)
    pixel_spacing = getattr(first, "PixelSpacing", [1.0, 1.0])
    try:
        slice_thickness = float(getattr(first, "SliceThickness", 1.0))
    except Exception:
        slice_thickness = 1.0

    volume = np.zeros((len(datasets), rows, cols), dtype=np.int16)
    for i, (_fp, ds) in enumerate(datasets):
        try:
            arr = ds.pixel_array
        except Exception:
            continue
        # Apply Rescale slope/intercept if present
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = (arr.astype(np.float32) * slope + intercept).astype(np.int16)
        volume[i] = arr

    image = sitk.GetImageFromArray(volume)  # z, y, x
    spacing = (float(pixel_spacing[1]), float(pixel_spacing[0]), float(slice_thickness))
    image.SetSpacing(spacing)
    sitk.WriteImage(image, out_nifti_path)

    return out_nifti_path, spacing


def run_vibe_segmentator(img_path: str, out_seg_path: str, device: str, dataset_id: int, project_root: str) -> None:

    script_path = os.path.join(project_root, "run_VIBESegmentator.py")
    cmd = [
        "python",
        script_path,
        "--img",
        img_path,
        "--out_path",
        out_seg_path,
        "--dataset_id",
        str(dataset_id),
        "--ddevice",
        device,
    ]

    subprocess.run(cmd, check=True)


def run_spine_instance_segmentation(img_path: str, out_seg_path: str, device: str, project_root: str) -> None:
    script_path = os.path.join(project_root, "run_instance_spine_segmentation.py")
    cmd = [
        "python",
        script_path,
        "--img",
        img_path,
        "--out_path",
        out_seg_path,
        "--ddevice",
        device,
        "--override",
    ]
    subprocess.run(cmd, check=True)


def compute_volumes(seg_path: str) -> Dict[int, Dict[str, float]]:

    image = sitk.ReadImage(seg_path)
    spacing = image.GetSpacing()  # (x, y, z) mm
    voxel_volume_mm3 = float(spacing[0] * spacing[1] * spacing[2])

    seg = sitk.GetArrayFromImage(image).astype(np.int32)  # z, y, x

    labels_of_interest = {
        65: "subcutaneous_fat",  # SAT
        66: "muscle",
        67: "inner_fat",        # interpretable as VAT in this context
    }

    result: Dict[int, Dict[str, float]] = {}
    for lbl, name in labels_of_interest.items():
        count = int((seg == lbl).sum())
        volume_ml = (count * voxel_volume_mm3) / 1000.0  # mm^3 to mL
        result[lbl] = {"name": name, "voxel_count": float(count), "volume_ml": float(volume_ml)}

    # Derived totals
    sat_ml = result.get(65, {}).get("volume_ml", 0.0)
    vat_ml = result.get(67, {}).get("volume_ml", 0.0)
    total_fat_ml = sat_ml + vat_ml
    result[1000] = {"name": "total_fat_SAT_plus_VAT", "voxel_count": 0.0, "volume_ml": float(total_fat_ml)}

    return result


def write_csv(volumes: Dict[int, Dict[str, float]], out_csv_path: str) -> None:

    lines = ["label_id,label_name,voxel_count,volume_ml"]
    for lbl in sorted(volumes.keys()):
        entry = volumes[lbl]
        lines.append(f"{lbl},{entry['name']},{entry['voxel_count']},{entry['volume_ml']}")

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_sat_vat_only(seg_path: str, out_path: str) -> None:
    img = sitk.ReadImage(seg_path)
    arr = sitk.GetArrayFromImage(img).astype(np.int32)
    mask = np.zeros_like(arr, dtype=np.int32)
    mask[arr == 65] = 65
    mask[arr == 67] = 67
    out_img = sitk.GetImageFromArray(mask)
    out_img.CopyInformation(img)
    sitk.WriteImage(out_img, out_path)


def compute_intervertebral_slice(instance_seg_path: str, v1_name: str, v2_name: str) -> int | None:
    if v_name2idx is None:
        return None
    try:
        v1_idx = v_name2idx[v1_name]
        v2_idx = v_name2idx[v2_name]
    except Exception:
        return None
    img = sitk.ReadImage(instance_seg_path)
    arr = sitk.GetArrayFromImage(img).astype(np.int32)
    z_indices_v1 = np.where((arr == v1_idx).any(axis=(1, 2)))[0]
    z_indices_v2 = np.where((arr == v2_idx).any(axis=(1, 2)))[0]
    if z_indices_v1.size == 0 or z_indices_v2.size == 0:
        return None
    z_v1 = int(np.round(z_indices_v1.mean()))
    z_v2 = int(np.round(z_indices_v2.mean()))
    z_target = int(np.round((z_v1 + z_v2) / 2))
    return z_target


def compute_area_on_slice(seg_path: str, z_index: int) -> Dict[str, float]:
    img = sitk.ReadImage(seg_path)
    spacing = img.GetSpacing()  # (x,y,z) in mm
    inplane_mm2 = float(spacing[0] * spacing[1])
    arr = sitk.GetArrayFromImage(img).astype(np.int32)
    z_index = int(np.clip(z_index, 0, arr.shape[0] - 1))
    slice2d = arr[z_index]
    sat_px = int((slice2d == 65).sum())
    vat_px = int((slice2d == 67).sum())
    return {
        "z_index": float(z_index),
        "sat_area_mm2": float(sat_px * inplane_mm2),
        "vat_area_mm2": float(vat_px * inplane_mm2),
    }


def save_slice_overlay(img_path: str, seg_path: str, z_index: int, out_png: str) -> None:
    img = sitk.ReadImage(img_path)
    seg = sitk.ReadImage(seg_path)
    img_arr = sitk.GetArrayFromImage(img).astype(np.float32)
    seg_arr = sitk.GetArrayFromImage(seg).astype(np.int32)
    z_index = int(np.clip(z_index, 0, img_arr.shape[0] - 1))
    base = img_arr[z_index]
    base = (base - np.percentile(base, 2)) / (np.percentile(base, 98) - np.percentile(base, 2) + 1e-6)
    base = np.clip(base, 0, 1)
    sat_mask = (seg_arr[z_index] == 65)
    vat_mask = (seg_arr[z_index] == 67)
    rgb = np.stack([base, base, base], axis=-1)
    # SAT in green, VAT in red
    rgb[sat_mask, :] = [0.0, 1.0, 0.0]
    rgb[vat_mask, :] = [1.0, 0.0, 0.0]
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.axis('off')
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0)
    plt.close()


def main() -> None:

    parser = argparse.ArgumentParser(description="Convert DICOM to NIfTI, segment with VIBESegmentator, and quantify SAT/VAT/muscle volumes.")
    parser.add_argument("--dicom_dir", required=True, help="Path to the DICOM series directory")
    parser.add_argument("--out_dir", required=True, help="Output directory for NIfTI, segmentation, and CSV")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Device for inference")
    parser.add_argument("--dataset_id", type=int, default=100, help="VIBESegmentator dataset id (default 100)")
    parser.add_argument("--compute_l4l5", action="store_true", help="Compute SAT/VAT area at L4-L5 and save slice image")
    parser.add_argument("--compute_l2l3", action="store_true", help="Compute SAT/VAT area at L2-L3 and save slice image")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    os.makedirs(args.out_dir, exist_ok=True)

    nifti_path = os.path.join(args.out_dir, "input_from_dicom.nii.gz")
    seg_path = os.path.join(args.out_dir, "segmentation.nii.gz")
    csv_path = os.path.join(args.out_dir, "quantification.csv")

    # 1) DICOM -> NIfTI
    convert_dicom_to_nifti(args.dicom_dir, nifti_path)

    # 2) Run segmentation
    run_vibe_segmentator(nifti_path, seg_path, args.device, args.dataset_id, project_root)

    # 3) Quantify
    volumes = compute_volumes(seg_path)
    write_csv(volumes, csv_path)

    # 4) Emit SAT/VAT-only segmentation
    sat_vat_only_path = os.path.join(args.out_dir, "segmentation_sat_vat_only.nii.gz")
    save_sat_vat_only(seg_path, sat_vat_only_path)

    # 5) Optional: L4-L5 SAT/VAT area and slice image
    if args.compute_l4l5 or args.compute_l2l3:
        try:
            instance_seg_path = os.path.join(args.out_dir, "spine_instance_seg.nii.gz")
            run_spine_instance_segmentation(nifti_path, instance_seg_path, args.device, project_root)
            if args.compute_l4l5:
                z_l4l5 = compute_intervertebral_slice(instance_seg_path, "L4", "L5")
                if z_l4l5 is not None:
                    areas45 = compute_area_on_slice(seg_path, z_l4l5)
                    l4l5_csv = os.path.join(args.out_dir, "l4l5_areas.csv")
                    with open(l4l5_csv, "w", encoding="utf-8") as f:
                        f.write("z_index,sat_area_mm2,vat_area_mm2\n")
                        f.write(f"{int(areas45['z_index'])},{areas45['sat_area_mm2']},{areas45['vat_area_mm2']}\n")
                    slice_png45 = os.path.join(args.out_dir, "l4l5_slice.png")
                    save_slice_overlay(nifti_path, seg_path, int(areas45["z_index"]), slice_png45)
            if args.compute_l2l3:
                z_l2l3 = compute_intervertebral_slice(instance_seg_path, "L2", "L3")
                if z_l2l3 is not None:
                    areas23 = compute_area_on_slice(seg_path, z_l2l3)
                    l2l3_csv = os.path.join(args.out_dir, "l2l3_areas.csv")
                    with open(l2l3_csv, "w", encoding="utf-8") as f:
                        f.write("z_index,sat_area_mm2,vat_area_mm2\n")
                        f.write(f"{int(areas23['z_index'])},{areas23['sat_area_mm2']},{areas23['vat_area_mm2']}\n")
                    slice_png23 = os.path.join(args.out_dir, "l2l3_slice.png")
                    save_slice_overlay(nifti_path, seg_path, int(areas23["z_index"]), slice_png23)
        except Exception:
            pass

    print(
        "Done. Outputs:\n"
        f"  NIfTI: {nifti_path}\n"
        f"  Segmentation: {seg_path}\n"
        f"  Quantification CSV: {csv_path}\n"
        f"  SAT/VAT only: {sat_vat_only_path}\n"
        f"  (Optional) L4L5 area CSV and slice image if requested.\n"
    )


if __name__ == "__main__":
    main()


