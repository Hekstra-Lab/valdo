import glob
import json
import os
import sys


# (field_name, expected_type, required, default)
STAGE_SCHEMAS = {
    "standardize": [
        ("source_path",        str,   True,  None),
        ("destination_path",   str,   True,  None),
        ("mtz_file_pattern",   str,   True,  None),
        ("expcolumns",         list,  True,  None),
        ("ncpu",               int,   False, 1),
    ],
    "reindex": [
        ("input_files",        (str, list), True,  None),
        ("reference_file",     str,         True,  None),
        ("output_folder",      str,         True,  None),
        ("columns",            list,        True,  None),
        ("wcorr",              bool,        False, False),
        ("cc_min_dif",         float,       False, 0.2),
        ("ncpu",               int,         False, 1),
    ],
    "scale": [
        ("file_list",          (str, list), True,  None),
        ("reference_mtz",      str,         True,  None),
        ("columns",            list,        True,  None),
        ("output_folder",      str,         True,  None),
        ("prefix",             str,         False, ""),
        ("when_opt",           (str, float, int), False, 0.2),
        ("ncpu",               int,         False, 1),
    ],
    "filter": [
        ("refine_summary",  str,              True,  None),
        ("reindex_record",  (str, type(None)), False, None),
        ("scaled_dir",      str,              True,  None),
        ("metrics",         (str, type(None)), False, None),
        ("max_rfree",       float,            False, 0.45),
        ("min_cc",          float,            False, 0.55),
        ("output",          str,              True,  None),
    ],
    "preprocess": [
        ("file_list",          (str, list), True,  None),
        ("output_folder",      str,         True,  None),
        ("intersection_path",  str,         True,  None),
        ("union_path",         str,         True,  None),
        ("sigF_path",          str,         True,  None),
        ("amplitude_col",      str,         False, "F-obs-scaled"),
        ("error_col",          str,         False, "SIGF-obs-scaled"),
        ("include_errors",     bool,        False, True),
        ("prefix",             str,         False, ""),
    ],
    "train": [
        ("vae_input_path",     str,         True,  None),
        ("vae_output_path",    str,         True,  None),
        ("vae_sigF_path",      str,         False, None),
        ("output_path",        str,         True,  None),
        ("latent_dim",         int,         False, 7),
        ("n_hidden_layers",    list,        False, [3, 6]),
        ("n_hidden_size",      int,         False, 100),
        ("epochs",             int,         False, 500),
        ("train_fraction",     float,       False, 0.8),
        ("batch_size",         int,         False, 100),
        ("learning_rate",      float,       False, 0.001),
        ("w_kl",               float,       False, 1.0),
        ("eps",                float,       False, 0.02),
        ("stdof",              (int, type(None)), False, None),
        ("include_errors",     bool,        False, True),
        ("random_seed",        int,         False, 42),
        ("activation",         str,         False, "tanh"),
    ],
    "reconstruct": [
        ("vae_path",           str,         True,  None),
        ("vae_input_path",     str,         True,  None),
        ("output_path",        str,         True,  None),
        ("ml_recon",           bool,        False, True),
        ("repeats",            int,         False, 1),
    ],
    "rescale": [
        ("recons_path",        str,         True,  None),
        ("intersection_path",  str,         True,  None),
        ("union_path",         str,         True,  None),
        ("input_files",        (str, list), True,  None),
        ("info_folder",        str,         True,  None),
        ("output_folder",      str,         True,  None),
        ("amplitude_col",      str,         False, "F-obs-scaled"),
        ("ncpu",               int,         False, 1),
    ],
    "add_phases_and_blobs": [
        ("file_list",              (str, list), True,  None),
        ("phasing_path",           str,         True,  None),
        ("output_folder",          str,         True,  None),
        ("blob_output_folder",     str,         True,  None),
        ("model_folder",           str,         True,  None),
        ("phase_2FOFC_col_in",     str,         False, "PH2FOFCWT"),
        ("phase_FOFC_col_in",      str,         False, "PHFOFCWT"),
        ("phase_2FOFC_col_out",    str,         False, "PH2FOFCWT"),
        ("phase_FOFC_col_out",     str,         False, "PHFOFCWT"),
        ("rfree_label_in",         (str, type(None)), False, None),
        ("sigF_col",               str,         False, "SIGF-obs-scaled"),
        ("diff_col",               str,         False, "diff"),
        ("sigdF_pct",              float,       False, 95.0),
        ("absdF_pct",              float,       False, 99.99),
        ("F_col",                  str,         False, "F-obs-scaled"),
        ("recons_col",             str,         False, "recons"),
        ("extrapolate_factors",    list,        False, [2, 4, 6, 8, 16]),
        ("blob_diff_col",          str,         False, "WDF"),
        ("phase_col",              str,         False, "PH2FOFCWT"),
        ("cutoff",                 float,       False, 3.5),
        ("radius_in_A",            float,       False, 4.0),
        ("prefix",                 str,         False, ""),
        ("ncpu",                   int,         False, 1),
    ],
    "tag_blobs": [
        ("blob_stats_path",    str,         True,  None),
        ("model_folder",       str,         True,  None),
        ("bound_models_folder",str,         False, None),
        ("mtz_folder",         str,         True,  None),
        ("focal_seqid",        int,         True,  None),
        ("focal_tag_name",     str,         True,  None),
        ("output_folder",      str,         True,  None),
        ("focal_radius",       float,       False, 5.0),
        ("ncpu",               int,         False, 1),
    ],
}


def load_config(path):
    if not os.path.isfile(path):
        print(f"Error: config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        if path.endswith((".yaml", ".yml")):
            import yaml
            with open(path) as f:
                return yaml.safe_load(f)
        else:
            with open(path) as f:
                return json.load(f)
    except Exception as e:
        print(f"Error parsing config file '{path}': {e}", file=sys.stderr)
        sys.exit(1)


def validate_config(stage, cfg):
    schema = STAGE_SCHEMAS.get(stage)
    if schema is None:
        print(f"Error: no schema for stage '{stage}'", file=sys.stderr)
        sys.exit(1)

    errors = []
    for field, expected_type, required, default in schema:
        if field not in cfg:
            if required:
                errors.append(f"  missing required field: '{field}'")
            else:
                cfg[field] = default
        else:
            ok, val = _coerce_type(cfg[field], expected_type)
            if ok:
                cfg[field] = val
            else:
                errors.append(f"  field '{field}': expected {_type_name(expected_type)}, got {type(cfg[field]).__name__}")

    if errors:
        print(f"Config validation failed for stage '{stage}':", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)

    return cfg


def expand_glob_field(value):
    if isinstance(value, list):
        return sorted(value)
    if isinstance(value, str) and value.endswith(".txt") and os.path.isfile(value):
        with open(value) as f:
            paths = [line.strip() for line in f if line.strip()]
        return sorted(paths)
    matches = sorted(glob.glob(value))
    if not matches:
        print(f"Warning: glob pattern '{value}' matched no files.", file=sys.stderr)
    return matches


def _coerce_type(val, expected_type):
    """Check val against expected_type, returning (ok, value).

    YAML parses `1` as an int, so a config written as `w_kl: 1` would otherwise
    be rejected by a strict float check; accept an int wherever a float is
    expected and pass the float on. Booleans are only ever accepted by fields
    that explicitly expect bool, since bool is a subclass of int.
    """
    types = expected_type if isinstance(expected_type, tuple) else (expected_type,)
    if isinstance(val, bool):
        return bool in types, val
    if isinstance(val, types):
        return True, val
    if float in types and isinstance(val, int):
        return True, float(val)
    return False, val


def _type_name(t):
    if isinstance(t, tuple):
        return " or ".join(x.__name__ if x is not type(None) else "null" for x in t)
    return t.__name__ if t is not type(None) else "null"
