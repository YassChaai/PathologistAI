import argparse
import subprocess
import csv
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
IMAGES_DIR = DATA_DIR / "CAMELYON17" / "images"

EXTRACT_SCRIPT = PROJECT_ROOT / "DEV" / "extract-metastasis_test.py"

WORKERS = 3


def load_patients_from_batch(mapping_csv: Path) -> list[str]:
    patients = []
    with open(mapping_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patients.append(row["patient_id"])
    return sorted(set(patients))


def run_patient_extraction(patient_id: str) -> None:
    print(f"🧠 Traitement du patient {patient_id}")

    cmd = [
        "uv", "run", "python",
        str(EXTRACT_SCRIPT),
        "--patient-id", patient_id
    ]

    subprocess.run(cmd, check=True)
    return patient_id


def cleanup_local_data() -> None:
    print("🧹 Nettoyage des images locales (TEST)")

    if IMAGES_DIR.exists():
        for f in IMAGES_DIR.glob("*.tif"):
            f.unlink()


def main():
    parser = argparse.ArgumentParser(description="Process a local batch of patients")
    parser.add_argument(
        "--batch-mapping",
        required=True,
        type=Path,
        help="CSV file containing patient_id list for this batch"
    )
    args = parser.parse_args()

    patients = load_patients_from_batch(args.batch_mapping)

    print(f"📦 Nombre de patients dans le batch : {len(patients)}")

    max_workers = min(WORKERS, len(patients))
    print(f"⚙️ Parallélisation activée : {max_workers} patients en parallèle")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_patient_extraction, patient_id): patient_id
            for patient_id in patients
        }

        for future in as_completed(futures):
            pid = futures[future]
            try:
                future.result()
                print(f"✅ Patient terminé : {pid}")
            except Exception as e:
                print(f"❌ Erreur sur le patient {pid}: {e}")
                raise

    cleanup_local_data()
    print("✅ Batch terminé avec succès")


if __name__ == "__main__":
    main()