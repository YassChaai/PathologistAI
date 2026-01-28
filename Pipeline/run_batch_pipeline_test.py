from pathlib import Path
import pandas as pd
import subprocess
import shutil
import math

# =========================
# CONFIG
# =========================

TEST_MAPPING = Path("../Data/dataset_lists/test_mapping.csv")
BATCH_SIZE = 6
TEST_MODE = False  # Mettre à False pour lancer tout le pipeline
PATCHES_ROOT = Path("../Data/patches/Test")

# Dossiers temporaires (WSI)
CAMELYON_ROOT = Path("../Data/CAMELYON17")
IMAGES_DIR = CAMELYON_ROOT / "images"

# Scripts existants
DOWNLOAD_SCRIPT = "download_train_from_aws.py"
PROCESS_BATCH_SCRIPT = "process_batch_test.py"


# =========================
# UTILS
# =========================

def run(cmd):
    print(f"\n▶ RUN: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# =========================
# MAIN PIPELINE
# =========================

def main():
    df = pd.read_csv(TEST_MAPPING)
    patient_ids = df["patient_id"].unique().tolist()
    if TEST_MODE:
        patient_ids = patient_ids[:BATCH_SIZE]

    if TEST_MODE:
        n_batches = 1
    else:
        n_batches = math.ceil(len(patient_ids) / BATCH_SIZE)

    print(f"Patients totaux: {len(patient_ids)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Nombre de batchs: {n_batches}")

    for i in range(n_batches):
        batch_patients = patient_ids[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

        patients_a_traiter = []

        for pid in batch_patients:
            patient_patch_dir = PATCHES_ROOT / pid

            if patient_patch_dir.exists() and any(patient_patch_dir.iterdir()):
                print(f"⏭️ Patient déjà traité, skip : {pid}")
            else:
                patients_a_traiter.append(pid)

        if not patients_a_traiter:
            print("⚠️ Tous les patients du batch ont déjà été traités, batch ignoré.")
            continue

        print("\n" + "=" * 60)
        print(f"🚀 BATCH {i + 1}/{n_batches}")
        print("Patients:", patients_a_traiter)

        # 1️⃣ Créer un mapping temporaire pour le batch
        batch_df = df[df["patient_id"].isin(patients_a_traiter)]
        batch_mapping = Path(f"../Data/dataset_lists/batches_test/batch_mapping_batch_{i + 1:03d}.csv")
        batch_df.to_csv(batch_mapping, index=False)

        try:
            # 2️⃣ Télécharger images + annotations (max 5 patients en parallèle dans le script de download)
            run([
                "uv", "run", "python",
                DOWNLOAD_SCRIPT,
                "--mapping", str(batch_mapping),
                "--no-annotations"
            ])

            # 3️⃣ Traitement batch local (extraction + labellisation + cleanup)
            run([
                "uv", "run", "python",
                PROCESS_BATCH_SCRIPT,
                "--batch-mapping", str(batch_mapping)
            ])
        finally:
            pass

    print("\n✅ PIPELINE TERMINÉ AVEC SUCCÈS")


if __name__ == "__main__":
    main()