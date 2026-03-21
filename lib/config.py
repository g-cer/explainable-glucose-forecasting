"""Configurazione centralizzata per la modalità dev/produzione."""

DEV_MODE = False
DEV_PATIENTS = 20

_RAW_DATA_DIR = "data/T1DiabetesGranada"
_SPLITS_DIR = "data/split_sets"
_STATIC_SPLITS_DIR = "data/static_split_sets"
_DEV_RAW_DATA_DIR = "data/dev/T1DiabetesGranada"
_DEV_SPLITS_DIR = "data/dev/split_sets"
_DEV_STATIC_SPLITS_DIR = "data/dev/static_split_sets"


def get_raw_data_dir():
    """Restituisce la directory dei dati grezzi."""
    return _DEV_RAW_DATA_DIR if DEV_MODE else _RAW_DATA_DIR


def get_splits_dir():
    """Restituisce la directory degli split."""
    return _DEV_SPLITS_DIR if DEV_MODE else _SPLITS_DIR


def get_static_splits_dir():
    """Restituisce la directory degli split statici."""
    return _DEV_STATIC_SPLITS_DIR if DEV_MODE else _STATIC_SPLITS_DIR


def get_raw_file(filename):
    """Restituisce il percorso completo di un file nella directory dati grezzi."""
    return f"{get_raw_data_dir()}/{filename}"
