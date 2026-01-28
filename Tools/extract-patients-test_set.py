from pathlib import Path
import pandas as pd

# =========================
# CONFIGURATION
# =========================

STAGES_PATH = "../Data/example.csv"
HOSPITAL_MAPPING_PATH = "../Data/dataset_lists/hospital_mapping.csv"
OUTPUT_DIR = "../Data/dataset_lists"

# Test design
PATIENTS_PER_CENTER = 4        # 2 pN0, 1 pN1, 1 pN2
OOD_CENTER_ID = 3              # Centre hors-domaine

NEGATIVE_STAGES = ["pN0"]
POSITIVE_STAGES = ["pN1", "pN2"]

RANDOM_STATE = 42

# =========================
# LOADERS
# =========================

def load_patients(stages_path: str) -> pd.DataFrame:
    """
    Load patients from stages.csv and extract patient_id.
    Only rows corresponding to patient ZIPs are kept.
    """
    df = pd.read_csv(stages_path)

    patients_df = df[df["patient"].str.endswith(".zip")].copy()
    patients_df["patient_id"] = patients_df["patient"].str.replace(
        ".zip", "", regex=False
    )

    return patients_df

# =========================
# CENTER INFERENCE
# =========================

def infer_center_from_patient_id(patient_id: str) -> int:
    """
    CAMELYON17 convention:
    Patients are grouped by blocks of 20 per center.
    Example:
      patient_000–019 -> center 0
      patient_020–039 -> center 1
      ...
      patient_100–119 -> center 0
    """
    pid = int(patient_id.replace("patient_", ""))
    return (pid // 20) % 5

# =========================
# DATA ENRICHMENT
# =========================

def add_center_id(patients_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a center_id column inferred from patient_id.
    This function does NOT filter or select patients.
    """
    df = patients_df.copy()
    df["center_id"] = df["patient_id"].apply(infer_center_from_patient_id)
    return df

# =========================
# SPLIT BY CENTER (NO SELECTION)
# =========================

def split_by_center(patients_df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """
    Split the patients DataFrame by center_id.
    Returns a dictionary: {center_id: patients_df_for_center}
    No filtering, no sampling, no balancing.
    """
    centers = {}
    for center_id in sorted(patients_df["center_id"].unique()):
        centers[center_id] = patients_df[
            patients_df["center_id"] == center_id
        ].copy()

    return centers

# =========================
# TEST SELECTION PER CENTER
# =========================

def select_test_patients_for_center(
    center_df: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Select test patients for a single center with the following rule:
    - 2 patients pN0
    - 1 patient pN1
    - 1 patient pN2

    If there are not enough patients for a given category,
    all available patients from that category are taken.
    """
    selected_parts = []

    # 2 x pN0
    neg_df = center_df[center_df["stage"].isin(NEGATIVE_STAGES)]
    if not neg_df.empty:
        selected_parts.append(
            neg_df.sample(
                n=min(2, len(neg_df)),
                random_state=random_state,
            )
        )

    # 1 x pN1
    pn1_df = center_df[center_df["stage"] == "pN1"]
    if not pn1_df.empty:
        selected_parts.append(
            pn1_df.sample(
                n=1,
                random_state=random_state,
            )
        )

    # 1 x pN2
    pn2_df = center_df[center_df["stage"] == "pN2"]
    if not pn2_df.empty:
        selected_parts.append(
            pn2_df.sample(
                n=1,
                random_state=random_state,
            )
        )

    if not selected_parts:
        return pd.DataFrame(columns=center_df.columns)

    return pd.concat(selected_parts, ignore_index=True)

# =========================
# TEST ORCHESTRATION (ALL CENTERS)
# =========================

def build_test_selection(patients_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orchestrate test selection across all centers.

    Returns:
    - selected_df: all selected test patients (all centers combined)
    - excluded_df: all patients NOT selected for the test
    """
    # Split by center
    centers = split_by_center(patients_df)

    selected_parts = []
    for center_id, center_df in centers.items():
        selected_center_df = select_test_patients_for_center(center_df)
        if not selected_center_df.empty:
            selected_parts.append(selected_center_df)

    if selected_parts:
        selected_df = pd.concat(selected_parts, ignore_index=True)
    else:
        selected_df = pd.DataFrame(columns=patients_df.columns)

    # Compute excluded patients
    selected_ids = set(selected_df["patient_id"])
    excluded_df = patients_df[
        ~patients_df["patient_id"].isin(selected_ids)
    ].copy()

    return selected_df, excluded_df

# =========================
# BUILD TEST MAPPING ROWS
# =========================

def build_test_rows(selected_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the final test mapping DataFrame with:
    - patient_id
    - pN_stage
    - label (0: pN0, 1: pN1/pN2)
    - center_id
    - is_ood (1 if center_id == OOD_CENTER_ID)
    """
    df = selected_df.copy()

    df["pN_stage"] = df["stage"]
    df["label"] = df["stage"].apply(
        lambda s: 0 if s in NEGATIVE_STAGES else 1
    )
    df["is_ood"] = (df["center_id"] == OOD_CENTER_ID).astype(int)

    return df[
        ["patient_id", "pN_stage", "label", "center_id", "is_ood"]
    ]

# =========================
# OUTPUTS
# =========================

def save_outputs(
    output_dir: str,
    test_rows: pd.DataFrame,
    selected_df: pd.DataFrame,
    excluded_df: pd.DataFrame,
):
    """
    Save test mapping and traceability files to disk.
    Mirrors the structure of the train script outputs.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    infos_dir = Path(output_dir) / "infos"
    infos_dir.mkdir(parents=True, exist_ok=True)

    # Main test mapping
    test_rows.to_csv(
        f"{output_dir}/test_mapping.csv",
        index=False,
    )

    # Traceability files (same philosophy as train)
    selected_df[selected_df["stage"].isin(NEGATIVE_STAGES)] \
        .sort_values(by=["center_id", "patient_id"])["patient_id"] \
        .to_csv(
            f"{output_dir}/infos/test_patients_negative.txt",
            index=False,
            header=False,
        )

    selected_df[selected_df["stage"].isin(POSITIVE_STAGES)] \
        .sort_values(by=["center_id", "patient_id"])["patient_id"] \
        .to_csv(
            f"{output_dir}/infos/test_patients_positive.txt",
            index=False,
            header=False,
        )

    excluded_df \
        .sort_values(by=["center_id", "patient_id"])["patient_id"] \
        .to_csv(
            f"{output_dir}/infos/test_patients_excluded.txt",
            index=False,
            header=False,
        )

    print("\nListes TEST sauvegardées dans", output_dir)

# =========================
# MAIN
# =========================

def extract_test_patients():
    # Load
    patients_df = load_patients(STAGES_PATH)

    # Enrich
    patients_df = add_center_id(patients_df)

    # Select / exclude
    selected_df, excluded_df = build_test_selection(patients_df)

    selected_df = selected_df.sort_values(
        by=["center_id", "patient_id"]
    ).reset_index(drop=True)

    # Build final mapping
    test_rows = build_test_rows(selected_df)

    # Console summary (same spirit as train)
    print("\nDATASET DE TEST")
    print("Nombre total de patients:", len(test_rows))
    print("\nRépartition par label:")
    print(test_rows["label"].value_counts())
    print("\nRépartition par centre:")
    print(test_rows["center_id"].value_counts())
    print("\nPatients hors-domaine (OOD):")
    print(
        test_rows
        .query("is_ood == 1")["patient_id"]
        .tolist()
    )

    # Save
    save_outputs(
        OUTPUT_DIR,
        test_rows,
        selected_df,
        excluded_df,
    )


if __name__ == "__main__":
    extract_test_patients()