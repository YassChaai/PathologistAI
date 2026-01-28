from pathlib import Path
import pandas as pd

# =========================
# CONFIGURATION
# =========================

STAGES_PATH = "../Data/stages.csv"
ANNOTATIONS_PATH = "../Data/dataset_lists/annotations-patients.csv"
OUTPUT_DIR = "../Data/dataset_lists"

N_TARGET = 5
STRICT_UNIFORM_PER_CENTER = False

MAX_TOTAL_PATIENTS = 40 # None pour désactiver le plafond

NEGATIVE_STAGES = ["pN0"]
POSITIVE_STAGES = ["pN1", "pN2"]

CENTER_MAP = {
    0: "Radboud University Medical Center (Nijmegen)",
    1: "Canisius-Wilhelmina Hospital (Nijmegen)",
    2: "University Medical Center Utrecht",
    3: "Rijnstate Hospital (Arnhem)",
    4: "Laboratorium Pathologie Oost-Nederland (Hengelo)",
}

EXCLUDED_CENTER_ID = 3

# =========================
# LOADERS
# =========================

def load_patients(stages_path: str) -> pd.DataFrame:
    df = pd.read_csv(stages_path)

    patients_df = df[df["patient"].str.endswith(".zip")].copy()
    patients_df["patient_id"] = patients_df["patient"].str.replace(".zip", "", regex=False)

    return patients_df


def load_annotated_patients(path: str) -> set:
    annotated = pd.read_csv(path, header=None)[0].tolist()
    return set(annotated)

# =========================
# SPLIT LOGIC
# =========================

def split_patients(patients_df: pd.DataFrame, annotated_patients: set):
    patients_negative = patients_df[
        patients_df["stage"].isin(NEGATIVE_STAGES)
    ]

    patients_positive = patients_df[
        (patients_df["stage"].isin(POSITIVE_STAGES))
        & (patients_df["patient_id"].isin(annotated_patients))
    ]

    patients_excluded = patients_df[
        ~patients_df["patient_id"].isin(
            set(patients_negative["patient_id"])
            | set(patients_positive["patient_id"])
        )
    ]

    return patients_negative, patients_positive, patients_excluded

# =========================
# BALANCING
# =========================

def choose_patients_per_center(
    patients_negative: pd.DataFrame,
    patients_positive: pd.DataFrame,
    n_target: int,
    strict_uniform: bool,
) -> int:
    center_ids = sorted(
        set(patients_negative["center"].unique())
        | set(patients_positive["center"].unique())
    )

    per_center_cap = {}
    for cid in center_ids:
        neg_n = len(patients_negative[patients_negative["center"] == cid])
        pos_n = len(patients_positive[patients_positive["center"] == cid])
        per_center_cap[cid] = min(neg_n, pos_n)

    n_uniform = min(per_center_cap.values()) if per_center_cap else 0

    if strict_uniform:
        n_per_center = min(n_target, n_uniform)
    else:
        n_per_center = n_target

    print("\nCapacité par centre (min(neg,pos)):", per_center_cap)
    print("N_UNIFORM possible sur tous les centres:", n_uniform)
    print("N_TARGET demandé:", n_target)
    print(
        "N retenu par centre:",
        n_per_center,
        "(STRICT_UNIFORM_PER_CENTER=",
        strict_uniform,
        ")",
    )

    return n_per_center


def balanced_sample_by_center(
    df_neg: pd.DataFrame,
    df_pos: pd.DataFrame,
    n_per_center: int,
) -> pd.DataFrame:
    selected_rows = []

    for center_id in sorted(df_neg["center"].unique()):
        neg_center = df_neg[df_neg["center"] == center_id]
        pos_center = df_pos[df_pos["center"] == center_id]

        n = min(n_per_center, len(neg_center), len(pos_center))

        if n < n_per_center:
            print(
                f"[INFO] Centre {center_id}: capacité insuffisante "
                f"(neg={len(neg_center)}, pos={len(pos_center)}). "
                f"On prend n={n}."
            )

        if n == 0:
            continue

        selected_rows.append(neg_center.sample(n=n, random_state=42))
        selected_rows.append(pos_center.sample(n=n, random_state=42))

    return pd.concat(selected_rows, ignore_index=True)


# =========================
# MAX TOTAL PATIENTS
# =========================

def cap_total_patients(
    balanced_df: pd.DataFrame,
    max_total: int | None,
    random_state: int = 42,
) -> pd.DataFrame:
    if max_total is None:
        return balanced_df

    if len(balanced_df) <= max_total:
        return balanced_df

    print(f"[INFO] Plafonnement du nombre total de patients à {max_total}")

    return (
        balanced_df
        .sample(n=max_total, random_state=random_state)
        .reset_index(drop=True)
    )

# =========================
# OUTPUTS
# =========================

def build_train_rows(balanced_df: pd.DataFrame):
    rows = []

    for _, row in balanced_df.iterrows():
        rows.append(
            {
                "patient_id": row["patient_id"],
                "pN_stage": row["stage"],
                "label": 0 if row["stage"] in NEGATIVE_STAGES else 1,
                "center_id": int(row["center"]),
            }
        )

    return rows


def save_outputs(
    output_dir: str,
    train_rows: list,
    patients_negative: pd.DataFrame,
    patients_positive: pd.DataFrame,
    patients_excluded: pd.DataFrame,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [{"center_id": cid, "center_name": name} for cid, name in CENTER_MAP.items()]
    ).to_csv(f"{output_dir}/hospital_mapping.csv", index=False)

    train_df = pd.DataFrame(train_rows).sort_values(
        by=["center_id", "patient_id"],
        ascending=[True, True],
    ).reset_index(drop=True)

    train_df.to_csv(
        f"{output_dir}/train_mapping.csv", index=False
    )


    print("\nListes sauvegardées dans", output_dir)

# =========================
# MAIN
# =========================

def extract_patients():
    patients_df = load_patients(STAGES_PATH)
    # Exclude one center from the training set (OOD setup)
    patients_df = patients_df[
        patients_df["center"] != EXCLUDED_CENTER_ID
    ].copy()

    annotated_patients = load_annotated_patients(ANNOTATIONS_PATH)

    patients_negative, patients_positive, patients_excluded = split_patients(
        patients_df,
        annotated_patients,
    )

    n_per_center = choose_patients_per_center(
        patients_negative,
        patients_positive,
        N_TARGET,
        STRICT_UNIFORM_PER_CENTER,
    )

    balanced_df = balanced_sample_by_center(
        patients_negative,
        patients_positive,
        n_per_center,
    )
    balanced_df = cap_total_patients(
        balanced_df,
        MAX_TOTAL_PATIENTS,
    )

    # Order by center then patient_id for reproducibility
    balanced_df = balanced_df.sort_values(
        by=["center", "patient_id"],
        ascending=[True, True],
    ).reset_index(drop=True)

    train_rows = build_train_rows(balanced_df)

    print("\nDATASET D’ENTRAÎNEMENT ÉQUILIBRÉ")
    print("Nombre total de patients:", len(train_rows))
    print("\nRépartition par label:")
    print(pd.DataFrame(train_rows)["label"].value_counts())
    print("\nRépartition par centre:")
    print(pd.DataFrame(train_rows)["center_id"].value_counts())

    print("\nPatients EXCLUS:")
    for p in sorted(patients_excluded["patient_id"].tolist()):
        print(p)

    save_outputs(
        OUTPUT_DIR,
        train_rows,
        patients_negative,
        patients_positive,
        patients_excluded,
    )


if __name__ == "__main__":
    extract_patients()