# src/config.py
import re

# --- File Paths ---
INPUT_FILE = 'Data_raw.xlsx'  # Supports .xlsx or .csv
# Intermediate output (optional, for debugging)
PROCESSED_FILE = 'data_processed.csv' 
# Final output for model training
FINAL_FILE = 'final_data.csv'

# --- Filtering ---
# --- Filtering ---
# STRICT STRATEGY: Only keep ingredients that appear >= 26 times.
# This limits us to the "Top 8" most common ingredients.
MIN_INGREDIENT_COUNT = 26

# --- Chemical Groups (Feature Engineering) ---
# DISABLED for Current_Method2 (Strict Top 8 Only)
CHEMICAL_GROUPS = {}

# --- Columns ---
# Exact column names expected in the raw input (after lowercase conversion)
RAW_INGREDIENT_COL = 'all ingredients in cryoprotective solution'

# Columns to preserve from input to output
COLS_TO_KEEP = ['viability', 'recovery', 'doubling time', 'cooling rate']

# --- Parsing Rules ---
# Units that are considered valid
VALID_UNITS = {
    'M', 'mM', '%', 'µM', 'µg/mL', 'ng/mL', 'mg/ml', 'nM', 'mmol/L', 'mol/L', 'g/L'
}

# Text to ignore during parsing
UNITS_TO_IGNORE = ['wt', '(wt)', 'v/v', 'w/w']

# Regex pattern for "Value Unit Ingredient" (e.g., "10 % DMSO")
# Pre-compiled for performance
# Sort units by length (descending) to match 'mmol/L' before 'mol/L'
_sorted_units = sorted(list(VALID_UNITS), key=len, reverse=True)
_units_pattern = '|'.join(re.escape(u) for u in _sorted_units)

# Strict Regex: Requires space or clear separation
STRICT_REGEX = re.compile(r'^\s*([\d\.]+)\s*(' + _units_pattern + r')\s+(.*)\s*$', re.IGNORECASE)
# Fallback Regex: For tight strings like "10%DMSO"
LENIENT_REGEX = re.compile(r'^\s*([\d\.]+)\s*(' + _units_pattern + r')\s*(.*)\s*$', re.IGNORECASE)

# --- Synonyms & Normalization ---
# Map diverse names to a single canonical name
# Format: { 'canonical_name': ['synonym1', 'synonym2', ...] }
SYNONYM_GROUPS = {
    '1,2-propanediol': ['propylene glycol', 'proh'],
    'dmso': ['me2so', 'dimethyl sulfoxide'],
    'ectoin': ['ectoine'],
    'eg': ['ethylene glycol', 'ethyleneglycol'],
    'fbs': ['fcs', 'fetal bovine serum', 'fetal calf serum'],
    'hes': ['hydroxyethyl starch', 'hes450', 'hydroxychyl starch'],
    'hs': ['human serum'],
    'hsa': ['human albumin', 'human serum albumin', 'has'],
    'mc': ['methylcellulose'],
    'ha': ['hmw-ha'],
    'dextran': ['dextran-40']
}

# Invert dictionary for fast lookup: {'synonym1': 'canonical_name'}
SYNONYM_MAP = {}
for canonical, synonyms in SYNONYM_GROUPS.items():
    SYNONYM_MAP[canonical] = canonical
    for syn in synonyms:
        SYNONYM_MAP[syn] = canonical
