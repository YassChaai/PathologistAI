# PathologistAI Copilot Instructions

## Project Overview
PathologistAI is a **histopathological image analysis pipeline** for automatic metastasis detection using the CAMELYON17 dataset. It implements a three-stage architecture:
1. **Preprocessing**: Download WSI (Whole Slide Images) → Extract patches at 20× magnification → Filter tissue quality
2. **Training**: Patch-level classification with deep learning (patch embeddings + patient-level aggregation)
3. **Interpretation**: Patient-level predictions & visualization

Data flows: `Data/CAMELYON17/{images,annotations}` → `Data/patches/{patient_id}/` → ML model → `ML/outputs/`

## Critical Architecture Decisions
- **Batch-based processing**: Pipeline splits patients into configurable batches (default 6/batch in `run_batch_pipeline_train.py`) to manage memory and allow distributed execution
- **Dual preprocessing split**: `Tools/` generates CSV mappings; `Pipeline/` implements download/extraction (DO NOT mix concerns)
- **CPU vs GPU separation**: Preprocessing (16 vCPU, 64GB) runs separately from training (GPU L4); scripts support this via independent entry points
- **Lazy data download**: WSIs downloaded per-patient batch, not globally (see `download_train_from_aws.py` concurrent.futures ThreadPoolExecutor with MAX_WORKERS=6)
- **Quality filters in patch extraction**: Tissue detection uses histogram + gradient variance thresholds (see `extract-metastasis_train.py::is_tissue_patch()`)

## Project Structure
- **Tools/**: Patient mapping & validation scripts (`extract-patients-train_set.py`, `extract-patients-test_set.py`). Output: CSV files in `Data/dataset_lists/`
- **Pipeline/**: Core preprocessing workflow entry points:
  - `preprocessing_pipeline.py` (main dispatcher; accepts `--mode train|test`)
  - `run_batch_pipeline_train.py` & `run_batch_pipeline_test.py` (batch orchestrators)
  - `process_batch_train.py` & `process_batch_test.py` (parallel patient extraction via ProcessPoolExecutor)
  - `download_train_from_aws.py` (S3 download with concurrent.futures; requires AWS CLI + s5cmd)
  - `extract-metastasis_train.py` & `extract-metastasis_test.py` (patch extraction per patient; uses OpenSlide + shapely geometry)
- **ML/**: Jupyter notebook (`Groupe-2_ML_Camelyon17.ipynb`) for training/visualization (depends on preprocessed patches in `Data/patches/`)
- **Data/**: Local data directory (not versioned). Structure:
  - `dataset_lists/`: Mapping CSVs (`train_mapping.csv`, `test_mapping.csv`, `annotations-patients.csv`)
  - `CAMELYON17/images/` & `CAMELYON17/annotations/`: Temporary WSI storage (cleaned after batch processing)
  - `patches/`: Output directory with structure `{patient_id}/slides/{slide_name}/patches/`

## Developer Workflows

### Setup
```bash
python -m pip install uv
uv venv && source .venv/bin/activate
uv sync
```

### Generate Mappings (One-time)
```bash
uv run python Tools/extract-patients-train_set.py
uv run python Tools/extract-patients-test_set.py
```
Outputs: `Data/dataset_lists/{train,test}_mapping.csv`

### Run Full Pipeline
```bash
uv run python Pipeline/preprocessing_pipeline.py --mode train   # or --mode test
```
This dispatches to `run_batch_pipeline_train.py` which orchestrates batches.

### Run Single Batch (Advanced)
```bash
uv run python Pipeline/process_batch_train.py --batch-mapping Data/dataset_lists/batches_train/batch_mapping_batch_001.csv
```

### Jupyter Notebooks
```bash
python -m ipykernel install --user --name pathologistai --display-name "PathologistAI (.venv)"
# Select kernel in VS Code/JupyterLab
```
⚠️ Requires preprocessed patches in `Data/patches/` (run pipeline first)

## Code Patterns & Conventions

### Imports & Paths
- Always resolve project root: `PROJECT_ROOT = Path(__file__).resolve().parents[1]`
- Use `pathlib.Path` for all file operations (not `os.path`)
- Relative paths in config: `TRAIN_MAPPING = Path("../Data/dataset_lists/train_mapping.csv")`

### Batch Processing
- Use `concurrent.futures.ProcessPoolExecutor` for multi-patient extraction (CPU-bound)
- Use `concurrent.futures.ThreadPoolExecutor` for S3 downloads (I/O-bound)
- Example: [process_batch_train.py:60-75](Process_batch_train.py#L60-L75) shows max_workers = min(3, len(patients))

### CSV Data Handling
- Load mappings with `pandas.read_csv()` then extract unique patient_ids: `df["patient_id"].unique().tolist()`
- Write batch mappings with `csv.DictWriter` (see `run_batch_pipeline_train.py`)

### Geometry & Patch Extraction
- Use `shapely.geometry.Polygon` for annotation boundaries
- Use `shapely.strtree.STRtree` for spatial indexing (patch containment queries)
- Patch coordinates are in WSI space (need conversion via OpenSlide level magnification)

### Filtering & Quality Control
- Patch quality check: see `is_tissue_patch()` in [extract-metastasis_train.py](extract-metastasis_train.py#L43) — combines tissue_ratio, intensity_variance, gradient_variance
- Skip patches if labeled outside annotation boundary (annotation→polygon→STRtree.contains check)

### Error Handling
- Use `subprocess.run(..., check=True)` in pipeline orchestrators (fail-fast on script errors)
- Use `try/except` in concurrent executors to log failures per patient (don't halt entire batch)

## Dependencies & Versions
- **Python 3.13** (see pyproject.toml `requires-python`)
- Key packages:
  - `openslide-python` 1.4.3: WSI reading
  - `torch` 2.9.1, `torchvision` 0.24.1: Deep learning (GPU in ML notebooks)
  - `pandas`, `numpy`: Data manipulation
  - `shapely` 2.1.2: Geometry operations
  - `pillow`: Image I/O
  - External CLI tools: `aws s3`, `s5cmd` (for S3 downloads)

## Common Pitfalls & Solutions

| Issue | Solution |
|-------|----------|
| "Script not found" when running pipeline | Check `DEV_DIR` path in `preprocessing_pipeline.py` — scripts were moved to `Pipeline/`, not `DEV/` |
| Patch extraction fails with "Annotation missing" | Ensure XML annotations exist in `CAMELYON17/annotations/` (run download script first) |
| Notebook kernel not found | Reinstall kernel: `python -m ipykernel install --user --name pathologistai` |
| Memory exhaustion during batch processing | Reduce `BATCH_SIZE` in `run_batch_pipeline_train.py` (default 6); or reduce `MAX_WORKERS` in `process_batch_train.py` |
| AWS download timeout | Adjust `MAX_WORKERS=6` in `download_train_from_aws.py` & ensure AWS CLI + s5cmd are installed |

## Testing & Debugging

- **Test mode**: Set `TEST_MODE=True` in `run_batch_pipeline_train.py` to process only first batch
- **Single patient**: Run `uv run python Pipeline/extract-metastasis_train.py --patient-id <ID>` directly
- **Inspect mappings**: `pandas.read_csv("Data/dataset_lists/train_mapping.csv").head(10)`
- **Check patches**: `ls -la Data/patches/{patient_id}/slides/*/patches/ | wc -l` (count extracted patches per patient)
