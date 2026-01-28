from __future__ import annotations
import argparse

import csv
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple
import time
from collections import defaultdict


S3_BUCKET = "camelyon-dataset"
S3_PREFIX = "CAMELYON17"
S3_IMAGES_PREFIX = f"{S3_PREFIX}/images"
S3_ANN_PREFIX = f"{S3_PREFIX}/annotations"
S5CMD_BIN = "s5cmd"

# Chemins locaux (racine projet = dossier Rendu)
DEFAULT_TRAIN_MAPPING_CSV = Path("../Data/dataset_lists/train_mapping.csv")

LOCAL_DATA_ROOT = Path("../Data") / "CAMELYON17"
LOCAL_IMAGES_DIR = LOCAL_DATA_ROOT / "images"
LOCAL_ANN_DIR = LOCAL_DATA_ROOT / "annotations"

# Téléchargement: 5 slides par patient (node 0..4)
NODES = range(5)

# Parallélisme (ajuste selon ta connexion + machine)
MAX_WORKERS = 6


@dataclass(frozen=True)
class Task:
    s3_key: str
    local_path: Path
    kind: str  # "image" or "annotation"

@dataclass(frozen=True)
class PatientDownload:
    patient_id: str
    image_tasks: List[Task]
    annotation_tasks: List[Task]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Télécharge depuis S3 (bucket public) les images (5 nodes) et les annotations "
            "associées aux patients listés dans un fichier CSV de mapping. "
            "Peut être utilisé pour TRAIN ou TEST. Les annotations sont optionnelles."
        )
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_TRAIN_MAPPING_CSV,
        help=(
            "Chemin vers le CSV contenant au minimum la colonne 'patient_id'. "
            "Par défaut: ../Data/dataset_lists/train_mapping.csv"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retélécharger même si le fichier local existe déjà.",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Ne télécharge pas les annotations XML (utile pour le TEST).",
    )
    return parser.parse_args()


def ensure_dirs(download_annotations: bool) -> None:
    LOCAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    if download_annotations:
        LOCAL_ANN_DIR.mkdir(parents=True, exist_ok=True)


def read_patient_ids(train_csv: Path) -> List[str]:
    if not train_csv.exists():
        raise FileNotFoundError(f"train_mapping.csv introuvable: {train_csv.resolve()}")

    patient_ids: Set[str] = set()
    with train_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "patient_id" not in reader.fieldnames:
            raise ValueError(f"Colonne 'patient_id' manquante dans {train_csv}")

        for row in reader:
            pid = (row.get("patient_id") or "").strip()
            if pid:
                patient_ids.add(pid)

    return sorted(patient_ids)


def build_tasks(patient_ids: Iterable[str]) -> List[Task]:
    tasks: List[Task] = []

    for pid in patient_ids:
        for node in NODES:
            # Images
            tif_name = f"{pid}_node_{node}.tif"
            tasks.append(
                Task(
                    s3_key=f"{S3_IMAGES_PREFIX}/{tif_name}",
                    local_path=LOCAL_IMAGES_DIR / tif_name,
                    kind="image",
                )
            )

            # Annotations (peuvent ne pas exister)
            xml_name = f"{pid}_node_{node}.xml"
            tasks.append(
                Task(
                    s3_key=f"{S3_ANN_PREFIX}/{xml_name}",
                    local_path=LOCAL_ANN_DIR / xml_name,
                    kind="annotation",
                )
            )

    return tasks

def group_tasks_by_patient(tasks: List[Task]) -> List[PatientDownload]:
    grouped: dict[str, dict[str, List[Task]]] = defaultdict(lambda: {"image": [], "annotation": []})

    for t in tasks:
        pid = t.local_path.name.split("_node_")[0]
        grouped[pid][t.kind].append(t)

    patients: List[PatientDownload] = []
    for pid, kinds in grouped.items():
        patients.append(
            PatientDownload(
                patient_id=pid,
                image_tasks=sorted(kinds["image"], key=lambda x: x.s3_key),
                annotation_tasks=sorted(kinds["annotation"], key=lambda x: x.s3_key),
            )
        )
    return patients


def file_ok(path: Path) -> bool:
    # Simple: si le fichier existe et n'est pas vide, on le considère OK
    return path.exists() and path.is_file() and path.stat().st_size > 0


# Estimation du volume total pour une liste de tâches
def estimate_total_size(tasks: List[Task]) -> int:
    total_bytes = 0
    for t in tasks:
        # estimation moyenne CAMELYON17 ≈ 2.5 Go par WSI
        if t.kind == "image":
            total_bytes += int(2.5 * 1024**3)
        else:
            # annotations XML très légères (~100 Ko)
            total_bytes += int(100 * 1024)
    return total_bytes


def run_aws_cp(s3_key: str, local_path: Path) -> Tuple[int, str]:
    """
    Retourne (returncode, stderr_or_msg).
    On utilise aws s3 cp avec --no-sign-request car bucket public.
    """
    s3_uri = f"s3://{S3_BUCKET}/{s3_key}"

    cmd = [
        "aws",
        "s3",
        "cp",
        s3_uri,
        str(local_path),
        "--no-sign-request",
        "--only-show-errors",
    ]

    # On capture stderr pour détecter proprement les "404" sur les annotations
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    msg = (proc.stderr or proc.stdout or "").strip()
    return proc.returncode, msg

def download_patient_images_recursive(patient_id: str, force: bool) -> None:
    """
    Télécharge toutes les images (nodes) d’un patient via s5cmd (rapide).
    """
    print(f"[{patient_id}] ▶ Téléchargement IMAGES (s5cmd)")

    cmd = [
        S5CMD_BIN,
        "--no-sign-request",
        "cp",
        f"s3://{S3_BUCKET}/{S3_IMAGES_PREFIX}/{patient_id}_node_*.tif",
        str(LOCAL_IMAGES_DIR),
    ]

    subprocess.run(cmd, check=True)

    print(f"[{patient_id}] ✅ Images téléchargées")

def download_patient_annotations_recursive(patient_id: str, force: bool) -> None:
    """
    Télécharge toutes les annotations d’un patient via aws s3 cp.
    L’absence d’annotations est un cas NORMAL pour CAMELYON17.
    """
    print(f"[{patient_id}] ▶ Téléchargement ANNOTATIONS (aws s3 cp)")

    cmd = [
        "aws",
        "s3",
        "cp",
        f"s3://{S3_BUCKET}/{S3_ANN_PREFIX}/",
        str(LOCAL_ANN_DIR),
        "--recursive",
        "--exclude", "*",
        "--include", f"{patient_id}_node_*.xml",
        "--no-sign-request",
        "--only-show-errors",
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # aws s3 cp ne renvoie pas toujours un code != 0 pour les fichiers absents
    # On considère l’absence d’annotations comme un cas normal
    if proc.returncode != 0:
        msg = (proc.stderr or "").lower()
        if "404" in msg or "not found" in msg or "nosuchkey" in msg:
            print(f"[{patient_id}] ℹ️ Aucune annotation trouvée (normal)")
            return
        raise RuntimeError(f"[{patient_id}] ❌ Erreur AWS annotations: {proc.stderr.strip()}")

    print(f"[{patient_id}] ✅ Annotations téléchargées")

def download_patient(
    patient: PatientDownload,
    force: bool,
    download_annotations: bool,
) -> None:
    print(f"\n🧑‍⚕️ Début téléchargement patient {patient.patient_id}")

    # Images (commande AWS unique par patient)
    download_patient_images_recursive(patient.patient_id, force)

    # Annotations (commande AWS unique par patient)
    if download_annotations:
        download_patient_annotations_recursive(patient.patient_id, force)

def download_one(task: Task, force: bool) -> Tuple[Task, str]:
    """
    Télécharge un objet. Pour les annotations, si ça n'existe pas, on ne crash pas.
    """
    if not force and file_ok(task.local_path):
        return task, "SKIP (déjà présent)"

    # Ensure parent folder exists
    task.local_path.parent.mkdir(parents=True, exist_ok=True)

    code, msg = run_aws_cp(task.s3_key, task.local_path)

    if code == 0:
        return task, "OK"

    # Les annotations peuvent être absentes: on les ignore
    if task.kind == "annotation":
        # aws renvoie souvent "404" ou "Not Found" ou "NoSuchKey"
        lowered = msg.lower()
        if ("404" in lowered) or ("not found" in lowered) or ("nosuchkey" in lowered) or ("no such key" in lowered):
            # On nettoie un éventuel fichier vide
            if task.local_path.exists() and task.local_path.stat().st_size == 0:
                try:
                    task.local_path.unlink()
                except OSError:
                    pass
            return task, "MISSING (annotation absente)"

    # Sinon, erreur réelle
    return task, f"ERROR: {msg or 'aws cp failed'}"


def print_summary(results: List[Tuple[Task, str]]) -> None:
    ok = sum(1 for _, s in results if s == "OK")
    skip = sum(1 for _, s in results if s.startswith("SKIP"))
    missing = sum(1 for _, s in results if s.startswith("MISSING"))
    err = sum(1 for _, s in results if s.startswith("ERROR"))

    print("\nRésumé téléchargement")
    print(f"- OK: {ok}")
    print(f"- SKIP: {skip}")
    print(f"- MISSING annotations: {missing}")
    print(f"- ERROR: {err}")

    if err:
        print("\nDétails erreurs:")
        for t, s in results:
            if s.startswith("ERROR"):
                print(f"  - {t.kind}: {t.s3_key} -> {t.local_path.name} | {s}")


# Fonction de téléchargement par étape avec barre de progression et ETA
def run_download_stage(
    stage_name: str,
    tasks: List[Task],
    force: bool,
    max_workers: int,
):
    print(f"\n===== {stage_name} =====")
    print(f"Nombre de fichiers : {len(tasks)}")

    total_bytes = estimate_total_size(tasks)
    print(f"Volume estimé : {total_bytes / (1024**3):.2f} Go")

    start_time = time.time()
    downloaded_bytes = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(download_one, t, force) for t in tasks]

        for fut in as_completed(futures):
            task, status = fut.result()
            completed += 1

            if task.kind == "image":
                downloaded_bytes += int(2.5 * 1024**3)
            else:
                downloaded_bytes += int(100 * 1024)

            elapsed = time.time() - start_time
            speed = downloaded_bytes / elapsed if elapsed > 0 else 0
            remaining = total_bytes - downloaded_bytes
            eta = remaining / speed if speed > 0 else 0

            print(
                f"[{stage_name}] {completed}/{len(tasks)} | "
                f"{downloaded_bytes / (1024**3):.2f}/{total_bytes / (1024**3):.2f} Go | "
                f"ETA ~ {eta/60:.1f} min",
                end="\r",
            )

    print(f"\n{stage_name} terminé.\n")


def main() -> None:
    args = parse_args()

    download_annotations = not args.no_annotations

    ensure_dirs(download_annotations)

    patient_ids = read_patient_ids(args.mapping)
    print(f"Patients dans mapping CSV ({args.mapping}): {len(patient_ids)}")
    print(f"Sortie images: {LOCAL_IMAGES_DIR.resolve()}")
    if download_annotations:
        print(f"Sortie annotations: {LOCAL_ANN_DIR.resolve()}")
    else:
        print("Annotations: désactivées (--no-annotations)")

    all_tasks = build_tasks(patient_ids)
    patient_jobs = group_tasks_by_patient(all_tasks)

    effective_workers = min(MAX_WORKERS, len(patient_jobs))

    print(f"Patients à télécharger : {len(patient_jobs)}")
    print(f"Téléchargements parallèles (patients): {effective_workers}")

    start = time.time()

    with ThreadPoolExecutor(max_workers=effective_workers) as ex:
        futures = [
            ex.submit(download_patient, p, args.force, download_annotations)
            for p in patient_jobs
        ]

        for f in as_completed(futures):
            f.result()

    elapsed = time.time() - start
    print(f"\n⏱️ Téléchargement batch terminé en {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()