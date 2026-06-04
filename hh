# ============================================================
# compute_Automates_cb — Recette Python Dataiku
# Extraction de DataFrames par plages de lignes depuis un CSV
# contenu dans le dernier fichier ZIP d'un Managed Folder.
# ============================================================

import dataiku
import pandas as pd
import zipfile
import io
import logging

# ── Configuration du logging ────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# 1. MAPPING : plages de lignes → datasets cibles
#    - "start" et "end" sont des indices 0-based (row_index)
#    - La première ligne du CSV (index 0) est l'en-tête du 1er tableau
#    - Adapter ces valeurs à la réalité du fichier
# ============================================================
SLICE_CONFIG = [
    # (dataset_name,       start, end_inclusive, has_header)
    ("Automates_cb_OPT_",  0,   1,  True),   # lignes 1-2  → 1ère ligne = header
    ("Automates_cb_LBP",   3,   4,  True),   # lignes 4-5
    ("Contrats_cb_OPT",    5,   7,  True),   # lignes 6-8
    ("Contrat_cb_LBP",     8,  11,  True),   # lignes 9-12
    ("Contrats_cb_OPT_2", 12,  14,  True),   # lignes 13-15
    ("Contrats_cb_LBP_2", 15,  17,  True),   # lignes 16-18
    ("Contrats_cb_EZY",   18,  20,  True),   # lignes 19-21
]

# Dossier source dans le Managed Folder
FOLDER_ID   = "ls0G0CHQ"          # ← ID de ton Managed Folder
SOURCE_PATH = "/LBP/"             # ← sous-dossier cible (ou "/" pour tout)


# ============================================================
# 2. HELPERS
# ============================================================

def get_latest_zip(folder: dataiku.Folder, prefix: str) -> str:
    """Retourne le chemin du ZIP le plus récent dans le prefix donné."""
    all_files = folder.list_paths_in_partition()
    zip_files = [
        f for f in all_files
        if f.startswith(prefix) and f.endswith(".zip")
    ]
    if not zip_files:
        raise FileNotFoundError(
            f"Aucun fichier ZIP trouvé dans le dossier '{prefix}'."
        )
    # Trier par date de dernière modification (lastModified)
    latest = max(
        zip_files,
        key=lambda x: folder.get_path_details(x).get("lastModified", 0)
    )
    logger.info(f"ZIP sélectionné : {latest}")
    return latest


def read_csv_from_zip(folder: dataiku.Folder, zip_path: str,
                      delimiter: str = ";", encoding: str = "utf-8") -> list[list[str]]:
    """
    Lit le premier fichier CSV trouvé dans le ZIP.
    Retourne une liste de listes (lignes brutes, sans parsing pandas).
    """
    with folder.get_download_stream(zip_path) as stream:
        zip_bytes = stream.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"Aucun fichier CSV trouvé dans le ZIP '{zip_path}'.")
        csv_name = csv_names[0]
        logger.info(f"CSV lu : {csv_name}  ({len(csv_names)} fichier(s) CSV dans le ZIP)")

        with zf.open(csv_name) as csv_file:
            text = io.TextIOWrapper(csv_file, encoding=encoding, errors="ignore")
            import csv
            reader = csv.reader(text, delimiter=delimiter)
            rows = list(reader)

    if not rows:
        raise ValueError(f"Le fichier CSV '{csv_name}' est vide.")

    logger.info(f"Nombre total de lignes lues : {len(rows)}")
    return rows


def rows_to_dataframe(rows: list[list[str]],
                      start: int, end: int,
                      has_header: bool = True) -> pd.DataFrame:
    """
    Extrait les lignes [start : end+1] et les convertit en DataFrame.
    - Si has_header=True  → la 1ère ligne de la plage = noms de colonnes.
    - Si has_header=False → colonnes nommées col_0, col_1, …
    """
    total = len(rows)

    # ── Validation de la plage ──────────────────────────────
    if start < 0 or end < start:
        raise ValueError(
            f"Plage invalide : start={start}, end={end}. "
            "end doit être ≥ start et start ≥ 0."
        )
    if start >= total:
        raise IndexError(
            f"start={start} dépasse le nombre de lignes du CSV ({total})."
        )
    if end >= total:
        logger.warning(
            f"end={end} dépasse le nombre de lignes ({total}). "
            "Troncature appliquée."
        )
        end = total - 1

    slice_ = rows[start : end + 1]   # end inclusif → +1 pour le slicing Python

    if not slice_:
        raise ValueError(f"La plage [{start}:{end}] est vide après extraction.")

    if has_header:
        if len(slice_) < 2:
            logger.warning(
                f"Plage [{start}:{end}] : seulement l'en-tête, aucune donnée."
            )
            return pd.DataFrame(columns=slice_[0])
        header = slice_[0]
        data   = slice_[1:]
    else:
        header = [f"col_{i}" for i in range(len(slice_[0]))]
        data   = slice_

    # Harmoniser la longueur des lignes par rapport à l'en-tête
    n_cols = len(header)
    data_padded = [
        row[:n_cols] + [""] * max(0, n_cols - len(row))
        for row in data
    ]

    df = pd.DataFrame(data_padded, columns=header)

    # Nettoyage léger : strip des espaces dans les noms de colonnes et valeurs str
    df.columns = [str(c).strip() for c in df.columns]
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    return df


def write_to_dataiku(df: pd.DataFrame, dataset_name: str) -> None:
    """Écrit un DataFrame dans un dataset Dataiku (write_with_schema)."""
    ds = dataiku.Dataset(dataset_name)
    ds.write_with_schema(df)
    logger.info(
        f"  → Dataset '{dataset_name}' écrit : "
        f"{len(df)} ligne(s), {len(df.columns)} colonne(s)."
    )


# ============================================================
# 3. PIPELINE PRINCIPAL
# ============================================================

def main():
    logger.info("=== Démarrage de la recette compute_Automates_cb ===")

    # ── 3a. Récupérer le Managed Folder ────────────────────
    folder = dataiku.Folder(FOLDER_ID)

    # ── 3b. Récupérer le dernier ZIP ────────────────────────
    try:
        zip_path = get_latest_zip(folder, SOURCE_PATH)
    except FileNotFoundError as e:
        logger.error(str(e))
        raise

    # ── 3c. Lire toutes les lignes du CSV ───────────────────
    try:
        rows = read_csv_from_zip(folder, zip_path)
    except (ValueError, zipfile.BadZipFile) as e:
        logger.error(f"Erreur de lecture du ZIP/CSV : {e}")
        raise

    # ── 3d. Extraire chaque plage et écrire dans Dataiku ────
    errors = []
    for dataset_name, start, end, has_header in SLICE_CONFIG:
        logger.info(
            f"Traitement : '{dataset_name}'  lignes [{start}:{end}] "
            f"(has_header={has_header})"
        )
        try:
            df = rows_to_dataframe(rows, start, end, has_header)
            write_to_dataiku(df, dataset_name)
        except (ValueError, IndexError) as e:
            msg = f"[ERREUR] Dataset '{dataset_name}' — {e}"
            logger.error(msg)
            errors.append(msg)   # continue les autres datasets

    # ── 3e. Rapport final ───────────────────────────────────
    if errors:
        summary = "\n".join(errors)
        raise RuntimeError(
            f"{len(errors)} erreur(s) rencontrée(s) durant l'exécution :\n{summary}"
        )

    logger.info("=== Recette terminée avec succès ===")


main()
