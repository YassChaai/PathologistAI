import openslide
from lxml import etree
import os
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon, box
from shapely.strtree import STRtree
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

def load_annotations(xml_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    annotations = root.findall(".//Annotation")
    return annotations

def annotation_to_polygon(annotation):
    coords = annotation.findall(".//Coordinate")
    points = [(float(c.get("X")), float(c.get("Y")))
              for c in coords if c.get("X") and c.get("Y")]
    return Polygon(points)

def compute_global_annotation_bbox(polygons):
    """
    Calcule une bounding box globale englobant toutes les annotations.
    Retourne un shapely box ou None s'il n'y a pas d'annotations.
    """
    if not polygons:
        return None

    min_x = min(p.bounds[0] for p in polygons)
    min_y = min(p.bounds[1] for p in polygons)
    max_x = max(p.bounds[2] for p in polygons)
    max_y = max(p.bounds[3] for p in polygons)

    return box(min_x, min_y, max_x, max_y)

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

def is_patch_cancer(patch_meta, slide, annotation_polygons, annotation_tree, global_annotation_bbox):
    x = patch_meta["x"]
    y = patch_meta["y"]
    level = patch_meta["level"]
    patch_size = patch_meta["patch_size"]

    downsample = slide.level_downsamples[level]

    patch_bbox = box(
        x * downsample,
        y * downsample,
        (x + patch_size) * downsample,
        (y + patch_size) * downsample
    )

    if global_annotation_bbox is not None:
        if not patch_bbox.intersects(global_annotation_bbox):
            return False

    if annotation_tree is not None:
        candidates = annotation_tree.query(patch_bbox)

        for candidate in candidates:
            # Shapely 2.x: STRtree.query() renvoie souvent des indices (numpy.int64),
            # pas directement des géométries. On supporte donc:
            # - int / numpy integer -> index dans annotation_polygons
            # - Geometry shapely -> utilisé tel quel
            if isinstance(candidate, (int, np.integer)):
                poly = annotation_polygons[int(candidate)]
            else:
                # Sécurité: on ignore tout ce qui n'est pas une vraie géométrie shapely
                if not hasattr(candidate, "geom_type"):
                    continue
                poly = candidate

            if patch_bbox.intersects(poly):
                return True
    else:
        for poly in annotation_polygons:
            if patch_bbox.intersects(poly):
                return True

    return False

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extraction des patchs (cancer / normal) pour un patient CAMELYON17"
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
def process_slide(slide_path: Path, annotations_dir: Path, cancer_dir: Path, normal_dir: Path, level: int = 2, patch_size: int = 224) -> tuple[int, int]:
    """Traite une slide (un node) d'un patient et écrit directement les patches dans cancer/normal.

    Retourne (nb_cancer, nb_normal) pour cette slide.
    """
    slide_id = slide_path.stem
    print(f"Traitement de {slide_id}")

    slide = load_slide(slide_path)

    # --- Annotations + index spatial ---
    xml_path = annotations_dir / f"{slide_id}.xml"
    if xml_path.exists():
        annotations = load_annotations(xml_path)
        annotation_polygons = [annotation_to_polygon(a) for a in annotations]
        global_annotation_bbox = compute_global_annotation_bbox(annotation_polygons)
        annotation_tree = STRtree(annotation_polygons) if annotation_polygons else None
    else:
        annotation_polygons = []
        global_annotation_bbox = None
        annotation_tree = None

    # --- Génération des patches candidats (tissu) ---
    patches_meta = split_into_patches(
        slide,
        level=level,
        patch_size=patch_size
    )

    slide_cancer = 0
    slide_normal = 0

    downsample = slide.level_downsamples[level]

    for patch in patches_meta:
        is_cancer = is_patch_cancer(
            patch,
            slide,
            annotation_polygons,
            annotation_tree,
            global_annotation_bbox
        )

        x = patch["x"]
        y = patch["y"]

        # Lecture du patch final et écriture directe
        region = slide.read_region(
            location=(int(x * downsample), int(y * downsample)),
            level=level,
            size=(patch_size, patch_size)
        ).convert("RGB")

        dst_dir = cancer_dir if is_cancer else normal_dir
        file_name = f"{slide_id}_x{str(x).zfill(6)}_y{str(y).zfill(6)}.png"
        dst = dst_dir / file_name
        region.save(dst)

        if is_cancer:
            slide_cancer += 1
        else:
            slide_normal += 1

    slide.close()
    return slide_cancer, slide_normal

def process_patient(patient_id: str):
    images_dir = DATA_ROOT / "CAMELYON17" / "images"
    annotations_dir = DATA_ROOT / "CAMELYON17" / "annotations"

    patient_slides = sorted(images_dir.glob(f"{patient_id}_node_*.tif"))

    dataset_root = DATA_ROOT / "patches" / patient_id
    cancer_dir = dataset_root / "cancer"
    normal_dir = dataset_root / "normal"

    cancer_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)

    total_cancer = 0
    total_normal = 0

    # Nombre de workers = min(5 slides, CPU disponibles - 1)
    max_workers = min(len(patient_slides), max(os.cpu_count() - 1, 1))

    print(f"⚙️ Parallélisation slides: {max_workers} workers")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_slide,
                slide_path,
                annotations_dir,
                cancer_dir,
                normal_dir,
                2,
                224,
            )
            for slide_path in patient_slides
        ]

        for future in as_completed(futures):
            slide_cancer, slide_normal = future.result()
            total_cancer += slide_cancer
            total_normal += slide_normal

    print("=================================")
    print(f"Patient traité : {patient_id}")
    print(f"Patches cancer : {total_cancer}")
    print(f"Patches normaux : {total_normal}")
    print(f"Dataset final : {dataset_root.resolve()}")

def main():
    args = parse_args()
    process_patient(args.patient_id)

if __name__ == "__main__":
    main()