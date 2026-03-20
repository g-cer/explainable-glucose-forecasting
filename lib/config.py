"""
Centralized configuration for dev/production mode switching.

Set DEV_MODE = True to use a reduced dataset (~5 patients) for fast iteration.
Set DEV_MODE = False to use the full dataset (~100 patients) for production runs.
"""

DEV_MODE = True
DEV_PATIENTS = 20

_RAW_DATA_DIR = "data/T1DiabetesGranada"
_SPLITS_DIR = "data/split_sets"
_STATIC_SPLITS_DIR = "data/static_split_sets"
_DEV_RAW_DATA_DIR = "data/dev/T1DiabetesGranada"
_DEV_SPLITS_DIR = "data/dev/split_sets"
_DEV_STATIC_SPLITS_DIR = "data/dev/static_split_sets"


def get_raw_data_dir():
    return _DEV_RAW_DATA_DIR if DEV_MODE else _RAW_DATA_DIR


def get_splits_dir():
    return _DEV_SPLITS_DIR if DEV_MODE else _SPLITS_DIR


def get_static_splits_dir():
    return _DEV_STATIC_SPLITS_DIR if DEV_MODE else _STATIC_SPLITS_DIR


def get_raw_file(filename):
    return f"{get_raw_data_dir()}/{filename}"
