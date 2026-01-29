# src/config.py
import re

# --- File Paths ---
INPUT_FILE = 'Cryopreservative Data 2026.csv'  # Focus exclusively on 2026 data
PROCESSED_FILE = 'data_processed.csv' 
# Final output for model training (uses LLM-cleaned data)
FINAL_FILE = 'cleaned_data_2026.csv'

# --- Filtering ---
MAX_FEATURES = 15 
MIN_INGREDIENT_COUNT = 1 

CHEMICAL_GROUPS = {}

# Exact column names expected in the raw input
RAW_INGREDIENT_COL = 'all ingredients in cryoprotective solution'

# Columns to preserve
COLS_TO_KEEP = ['viability', 'recovery', 'doubling time', 'cooling rate']

# --- Parsing Rules ---
VALID_UNITS = {
    'M', 'mM', '%', 'µM', 'µg/mL', 'ng/mL', 'mg/ml', 'nM', 'mmol/L', 'mol/L', 'g/L'
}
UNITS_TO_IGNORE = ['wt', '(wt)', 'v/v', 'w/w']

_sorted_units = sorted(list(VALID_UNITS), key=len, reverse=True)
_units_pattern = '|'.join(re.escape(u) for u in _sorted_units)

STRICT_REGEX = re.compile(r'^\s*([\d\.]+)(?:\s*[±±]\s*[\d\.]+)?\s*(' + _units_pattern + r')\s+(.*)\s*$', re.IGNORECASE)
LENIENT_REGEX = re.compile(r'^\s*([\d\.]+)(?:\s*[±±]\s*[\d\.]+)?\s*(' + _units_pattern + r')\s*(.*)\s*$', re.IGNORECASE)

# ... (Synonyms kept for legacy compatibility but new logic uses internal map) ...
SYNONYM_GROUPS = {
    '1,2-propanediol': ['propylene glycol', 'proh'],
    'dmso': ['me2so', 'dimethyl sulfoxide'],
    'ectoin': ['ectoine'],
    'eg': ['ethylene glycol', 'ethyleneglycol'],
    'fbs': ['fcs', 'fetal bovine serum', 'fetal calf serum', 'fetal bovine serum culture medium'],
    'hes': ['hydroxyethyl starch', 'hes450', 'hydroxychyl starch'],
    'hs': ['human serum'],
    'hsa': ['human albumin', 'human serum albumin', 'has', 'human serum albumin(hsa)'],
    'mc': ['methylcellulose'],
    'ha': ['hmw-ha'],
    'dextran': ['dextran-40'],
    'stem-cellbanker': ['cb', 'stem-cellbankertm', 'stem-cellbanker tm', 'stem-cell-banker'],
    'dmem': ['admem', 'dulbecco\'s modified eagle medium'],
    'glucose': ['d-glucose'],
    'sucrose': ['saccharose']
}

SYNONYM_MAP = {}
for canonical, synonyms in SYNONYM_GROUPS.items():
    SYNONYM_MAP[canonical] = canonical
    for syn in synonyms:
        SYNONYM_MAP[syn] = canonical
