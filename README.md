# PathologistAI

## 1. Présentation du projet
PathologistAI est un pipeline de détection automatique de métastases ganglionnaires à partir d’images histopathologiques CAMELYON17. Le projet couvre l’ensemble de la chaîne de traitement :
**prétraitement des WSI → extraction de patches → entraînement patch‑level → agrégation patient‑level → interprétabilité**.
L’objectif scientifique est d’évaluer une approche patch‑level avec agrégation patient‑level pour la décision clinique.

---

## 2. Pré‑requis système
- **Python 3.13** (voir `.python-version`, `pyproject.toml`)
- **Git**
- Accès à une machine locale ou une **VM cloud**
  (l’exécution locale est possible mais **fortement déconseillée** pour le prétraitement à grande échelle)
- Outils externes utilisés par la pipeline de téléchargement :
  - **AWS CLI** (appelé par `Pipeline/download_train_from_aws.py`)
  - **s5cmd** (appelé par `Pipeline/download_train_from_aws.py`)

---

## 3. Gestion des dépendances avec uv

Ordre strict demandé :

1) **Installer uv**
```bash
python -m pip install uv
```

2) **Créer l’environnement virtuel**
```bash
uv venv
```

3) **Activer l’environnement**
```bash
# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

4) **Synchroniser les dépendances**
```bash
uv sync
```

---

## 4. Structure du projet
```
PathologistAI/
├─ Data/                 # Répertoire de données local (non versionné)
├─ Tools/                # Scripts de mapping (train/test), inspection, annotations
├─ Pipeline/             # Pipeline de prétraitement + extraction de patches
├─ ML/                   # Notebooks Jupyter + outputs
├─ Rapport Ecrit/        # Rapport PDF
├─ pyproject.toml        # Dépendances (uv)
└─ README.md
```

**Important :**
- `Data/` est un répertoire **local de données** (non versionné).
  **Créez‑le manuellement si absent après clonage.**
- Les données volumineuses (ex. `Data/CAMELYON17`, `Data/patches`) ne sont pas versionnées.

---

## 5. Préparation des datasets (pipeline)

### 5.1 Génération des mappings (train/test)
Les scripts de `Tools/` produisent les fichiers de mapping utilisés par la pipeline :
- `Data/dataset_lists/train_mapping.csv`
- `Data/dataset_lists/test_mapping.csv`
- `Data/dataset_lists/hospital_mapping.csv`

Exemple :
```bash
uv run python Tools/extract-patients-train_set.py
uv run python Tools/extract-patients-test_set.py
```

### 5.2 Lancer la pipeline (script unique)
La pipeline est pilotée par **un seul script** et dépend de l’argument `--mode` :

```bash
# Mode entraînement
uv run python Pipeline/preprocessing_pipeline.py --mode train

# Mode test
uv run python Pipeline/preprocessing_pipeline.py --mode test
```

La pipeline :
- télécharge les données,
- extrait les patches,
- structure les datasets.

---

## 6. Utilisation des notebooks
Les notebooks se trouvent dans `ML/`, notamment :

- `ML/Groupe-2_ML_Camelyon17.ipynb`

### Connexion du kernel à l’environnement uv
```bash
python -m ipykernel install --user --name pathologistai --display-name "PathologistAI (.venv)"
```

Ouvrez ensuite le notebook via votre IDE (VS Code, JupyterLab) et sélectionnez le kernel **PathologistAI (.venv)**.

⚠️ L’exécution des notebooks suppose que la **pipeline de prétraitement a déjà été lancée**.

---

## 7. Exécution sur infrastructure Cloud (recommandé)

Configuration recommandée (basée sur le projet réel) :

### VM Preprocessing (CPU‑intensive)
- ~16 vCPU
- ~64 Go RAM
- SSD persistant ~150 Go

### VM Deep Learning (GPU‑intensive)
- **GPU NVIDIA L4**
- 16 vCPU
- 64 Go RAM
- SSD persistant ~150 Go

**Pourquoi séparer preprocessing et deep learning ?**
Le prétraitement (téléchargement + extraction de patches) est **CPU‑intensif**, tandis que l’entraînement est **GPU‑intensif**. Séparer les charges évite la contention et optimise les coûts.

Le projet est compatible **local / cloud**, mais le cloud est recommandé pour des temps d’exécution raisonnables.

---

## 8. Notes importantes
- Le projet est **conçu pour CAMELYON17**.
- Les coûts cloud observés sont restés **limités (ordre de grandeur ~40€)** pour l’ensemble du projet.
- Le projet est **académique / expérimental** : il vise la reproductibilité et l’analyse critique, pas un usage clinique direct.

---

## 9. Conclusion (démarrage rapide)
1. Installer **uv**, créer et activer l’environnement.
2. Créer `Data/` si nécessaire.
3. Générer les mappings (train/test) avec `Tools/`.
4. Lancer la pipeline (`--mode train` / `--mode test`).
5. Ouvrir le notebook dans `ML/` avec le kernel `.venv`.

Le pipeline est conçu pour être reproductible, documenté et orienté recherche.
