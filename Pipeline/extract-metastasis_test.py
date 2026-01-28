import openslide
import os
from pathlib import Path
import numpy as np
import shutil
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# =========================
# PROJECT ROOT
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "Data"

# =========================
# FUNCTIONS (from previous scripts)
# =========================
def load_slide(path):
    return openslide.OpenSlide(path)

def is_tissue_patch(
    patch_img,
    tissue_threshold=0.40,
    intensity_var_threshold=25.0,
    gradient_var_threshold=5.0
):
    """
    Filtre strict des patchs non informatifs.
    Logique :
    1. Rejette les fonds noirs/blancs (verre)
    2. Rejette les patchs trop homogènes (brume, flou)
    3. Rejette les patchs sans structure locale (pas de noyaux, pas de bords)
    """

    arr = np.array(patch_img)

    # Sécurité : image RGB attendue
    if arr.ndim != 3 or arr.shape[2] != 3:
        return False

    # Conversion en niveaux de gris
    gray = arr.mean(axis=2)

    # -------------------------
    # 1. Ratio de pixels tissu
    # -------------------------
    tissue_pixels = (gray > 30) & (gray < 220)
    tissue_ratio = tissue_pixels.sum() / tissue_pixels.size

    if tissue_ratio < tissue_threshold:
        return False

    # -------------------------
    # 2. Variance d'intensité
    # -------------------------
    intensity_variance = gray.var()
    if intensity_variance < intensity_var_threshold:
        return False

    # -------------------------
    # 2.b Saturation couleur (évite patchs gris / vitreux)
    # -------------------------
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    color_std = np.std(np.stack([r, g, b], axis=2), axis=2).mean()

    if color_std < 8.0:
        return False

    # -------------------------
    # 3. Variance du gradient (structure locale)
    # -------------------------
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)

    gradient_magnitude = np.sqrt(gx[:-1, :]**2 + gy[:, :-1]**2)
    gradient_variance = gradient_magnitude.var()

    if gradient_variance < gradient_var_threshold:
        return False

    return True

def compute_tissue_mask(slide, mask_level, threshold=0.8):
    """
    Calcule un masque tissu à basse résolution.
    Retourne un numpy array booléen où True = tissu.
    """
    img = slide.read_region((0, 0), mask_level, slide.level_dimensions[mask_level]).convert("RGB")
    arr = np.array(img)
    gray = arr.mean(axis=2)

    # Pixels non blancs / non noirs
    tissue = (gray > 30) & (gray < 220)

    # Ratio local de tissu
    return tissue

def split_into_patches(slide, level=2, patch_size=224):
    """
    Découpe une WSI (au niveau `level`) en patches carrés `patch_size x patch_size`.
    Retourne une liste de dictionnaires contenant les métadonnées des patches valides.
    """
    # === Étape 1: masque tissu basse résolution ===
    mask_level = slide.level_count - 1
    tissue_mask = compute_tissue_mask(slide, mask_level)
    mask_h, mask_w = tissue_mask.shape

    downsample = slide.level_downsamples[level]
    width, height = slide.level_dimensions[level]

    out = []

    scale = slide.level_downsamples[level] / slide.level_downsamples[mask_level]

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):

            # Coordonnées correspondantes dans le masque tissu
            mx = int(x * scale)
            my = int(y * scale)

            if mx >= mask_w or my >= mask_h:
                continue

            # Skip si pas de tissu à cet endroit
            if not tissue_mask[my, mx]:
                continue

            w = min(patch_size, width - x)
            h = min(patch_size, height - y)

            # Lecture du patch (coordonnées ramenées au niveau 0 via downsample)
            region = slide.read_region(
                location=(int(x * downsample), int(y * downsample)),
                level=level,
                size=(w, h)
            ).convert("RGB")

            # Filtrage des patchs sans tissu (fond noir / verre)
            if not is_tissue_patch(region):
                continue

            # Pour un dataset ML, on garde une taille fixe: on ignore les patches incomplets des bords
            if w != patch_size or h != patch_size:
                continue

            out.append({
                "x": x,
                "y": y,
                "level": level,
                "patch_size": patch_size
            })

    return out

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extraction des patchs (TEST) pour un patient CAMELYON17"
    )
    parser.add_argument(
        "--patient-id",
        required=True,
        help="Identifiant patient (ex: patient_017)"
    )
    return parser.parse_args()

# =========================
# MAIN PIPELINE (PATIENT LEVEL)
# =========================
def process_slide(
    slide_path: Path,
    output_dir: Path,
    level: int = 2,
    patch_size: int = 224
) -> int:
    """Traite une slide (un node) d'un patient et écrit directement les patches dans output_dir.

    Retourne le nombre de patches extraits pour cette slide.
    """
    slide_id = slide_path.stem
    print(f"Traitement de {slide_id}")

    slide = load_slide(slide_path)

    saved = 0

    # --- Génération des patches candidats (tissu) ---
    patches_meta = split_into_patches(
        slide,
        level=level,
        patch_size=patch_size
    )

    downsample = slide.level_downsamples[level]

    for patch in patches_meta:
        x = patch["x"]
        y = patch["y"]

        # Lecture du patch final et écriture directe
        region = slide.read_region(
            location=(int(x * downsample), int(y * downsample)),
            level=level,
            size=(patch_size, patch_size)
        ).convert("RGB")

        file_name = f"{slide_id}_x{str(x).zfill(6)}_y{str(y).zfill(6)}.png"
        dst = output_dir / file_name
        region.save(dst)
        saved += 1

    slide.close()
    return saved

def process_patient(patient_id: str):
    images_dir = DATA_ROOT / "CAMELYON17" / "images"

    patient_slides = sorted(images_dir.glob(f"{patient_id}_node_*.tif"))

    dataset_root = DATA_ROOT / "patches" / "Test" / patient_id
    dataset_root.mkdir(parents=True, exist_ok=True)

    total_patches = 0

    # Nombre de workers = min(5 slides, CPU disponibles - 1)
    max_workers = min(len(patient_slides), max(os.cpu_count() - 1, 1))

    print(f"⚙️ Parallélisation slides: {max_workers} workers")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_slide,
                slide_path,
                dataset_root,
                2,
                224,
            )
            for slide_path in patient_slides
        ]

        for future in as_completed(futures):
            saved = future.result()
            total_patches += saved

    print("=================================")
    print(f"Patient traité (TEST) : {patient_id}")
    print(f"Patches extraits : {total_patches}")
    print(f"Dataset final : {dataset_root.resolve()}")

def main():
    args = parse_args()
    process_patient(args.patient_id)

if __name__ == "__main__":
    main()