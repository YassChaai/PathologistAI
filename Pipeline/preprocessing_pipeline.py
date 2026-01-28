from pathlib import Path
import argparse
from subprocess import run, CalledProcessError
import sys

# =========================
# CONFIGURATION
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEV_DIR = PROJECT_ROOT / "DEV"

RUN_TRAIN_PIPELINE = DEV_DIR / "run_batch_pipeline_train.py"
RUN_TEST_PIPELINE = DEV_DIR / "run_batch_pipeline_test.py"

# =========================
# ARGUMENTS
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dispatcher preprocessing pipeline (Train ou Test)"
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["train", "test"],
        help="Mode d'exécution de la pipeline: train ou test",
    )

    return parser.parse_args()

# =========================
# MAIN
# =========================

def main() -> None:
    args = parse_args()

    if args.mode == "train":
        pipeline_script = RUN_TRAIN_PIPELINE
        print("🚀 Lancement de la pipeline TRAIN")
    else:
        pipeline_script = RUN_TEST_PIPELINE
        print("🚀 Lancement de la pipeline TEST")

    if not pipeline_script.exists():
        print(f"❌ Script introuvable: {pipeline_script}")
        sys.exit(1)

    try:
        run(
            ["uv", "run", "python", str(pipeline_script)],
            check=True,
        )
    except CalledProcessError as e:
        print("❌ Erreur lors de l'exécution de la pipeline")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
