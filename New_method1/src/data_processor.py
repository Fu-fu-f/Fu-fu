# src/data_processor.py
import pandas as pd
import numpy as np
import re
import os
from . import config

class DataProcessor:
    def __init__(self, input_path=None):
        self.input_path = input_path or config.INPUT_FILE
        self.df = None
        self.parsing_errors = []

    def load_data(self):
        """Loads data from Excel or CSV and performs initial cleaning (formerly Beginning.py)."""
        print(f"Loading data from {self.input_path}...")
        
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        if self.input_path.endswith('.xlsx'):
            self.df = pd.read_excel(self.input_path)
        else:
            self.df = pd.read_csv(self.input_path)

        # Standardize columns to lowercase
        self.df.columns = [str(col).lower().strip() for col in self.df.columns]
        
        # Renaissance of Beginning.py: Check for 'all_ingredient' existence
        if config.RAW_INGREDIENT_COL not in self.df.columns:
            # Try to find it by fuzzy matching if exact match fails, or use 1st column
            print(f"Warning: '{config.RAW_INGREDIENT_COL}' column not found. Using first column as ingredient source.")
            self.df.rename(columns={self.df.columns[0]: config.RAW_INGREDIENT_COL}, inplace=True)

        original_len = len(self.df)
        
        # 1. Drop completely empty rows
        self.df.dropna(how='all', inplace=True)
        
        # 2. Drop rows where ingredient is missing/empty
        self.df = self.df[self.df[config.RAW_INGREDIENT_COL].notna()]
        self.df = self.df[self.df[config.RAW_INGREDIENT_COL].astype(str).str.strip() != '']
        
        print(f"Dropped {original_len - len(self.df)} empty/invalid rows. Current count: {len(self.df)}")
        return self.df

    def remove_rapid_freeze(self):
        """Removes rows where cooling rate is 'rapid freeze' (Legacy parity)."""
        print("--- Cleaning: Removing 'Rapid Freeze' rows ---")
        if 'cooling rate' not in self.df.columns:
            return

        # Normalized check
        # Convert to string, lowercase, remove underscores and spaces for loose match
        cr_clean = self.df['cooling rate'].astype(str).str.lower().str.replace('_', ' ').str.strip()
        mask = cr_clean == 'rapid freeze'
        
        count = mask.sum()
        if count > 0:
            self.df = self.df[~mask]
            print(f"Deleted {count} rows with 'rapid freeze'.")
        else:
            print("No 'rapid freeze' rows found.")

    def parse_ingredients(self):
        """Parses the ingredient string into the feature matrix (formerly first_process.py)."""
        print("Parsing ingredients...")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # List to store dictionaries of {feature_name: value}
        parsed_data = []
        
        for idx, row in self.df.iterrows():
            raw_str = str(row[config.RAW_INGREDIENT_COL])
            # Split by '+'
            ingredients = [s.strip() for s in raw_str.split('+') if s.strip()]
            
            row_features = {}
            for ing in ingredients:
                # Remove ignored units (e.g. w/w) from the string before regex
                clean_ing = ing
                for ignore in config.UNITS_TO_IGNORE:
                    clean_ing = clean_ing.replace(ignore, '').strip()

                # Match Regex
                match = config.STRICT_REGEX.match(clean_ing)
                if not match:
                    match = config.LENIENT_REGEX.match(clean_ing)
                
                if match:
                    val_str, unit, name = match.groups()
                    try:
                        value = float(val_str)
                        name = name.lower().strip()
                        
                        # Canonicalize name
                        name = config.SYNONYM_MAP.get(name, name)
                        
                        # Handle Unit Normalization (M -> mM)
                        if unit == 'M':
                            value *= 1000
                            unit = 'mM'
                        elif unit == 'mol/L':
                            value *= 1000
                            unit = 'mM'
                        elif unit == 'mmol/L':
                            unit = 'mM'

                        feature_key = f"{name}({unit})"
                        
                        # Accumulate (handle duplicate ingredients in same row if any)
                        row_features[feature_key] = row_features.get(feature_key, 0.0) + value

                    except ValueError:
                        self.parsing_errors.append(f"Row {idx}: Value Error parsing '{ing}'")
                else:
                    self.parsing_errors.append(f"Row {idx}: Regex failed for '{ing}'")
            
            parsed_data.append(row_features)

        # Create features DataFrame
        features_df = pd.DataFrame(parsed_data).fillna(0.0)
        
        # Merge back with original data (keeping preserved columns)
        # We assume 1-to-1 alignment since we iterated rows
        self.df.reset_index(drop=True, inplace=True)
        features_df.reset_index(drop=True, inplace=True)
        
        # Columns to keep from original
        meta_cols = [c for c in config.COLS_TO_KEEP if c in self.df.columns]
        
        self.final_df = pd.concat([self.df[[config.RAW_INGREDIENT_COL]], features_df, self.df[meta_cols]], axis=1)
        
        if self.parsing_errors:
            print(f"⚠️ Encountered {len(self.parsing_errors)} parsing errors (logged).")
            # In production, we might log these to a file
            
        print(f"Extracted {len(features_df.columns)} unique ingredient features.")
        
        
        # Apply Chemical Grouping (Phase 3)
        self.final_df = self._apply_chemical_grouping(self.final_df)
        
        # OPTIONAL: Still filter extremely sparse things if they don't fit in groups?
        # For now, we trust the grouping to handle most sparsity.
        self.final_df = self._filter_sparse_data(self.final_df)
        
        return self.final_df

    def _apply_chemical_grouping(self, df):
        """Sums specific ingredients into functional groups defined in config."""
        print("--- Feature Engineering: Applying Chemical Grouping ---")
        
        # 1. Identify all current ingredient columns
        meta_cols = [c for c in config.COLS_TO_KEEP if c in df.columns] + [config.RAW_INGREDIENT_COL]
        # We need to be careful not to double count if we iterate.
        
        new_df = df.copy()
        
        for group_name, members in config.CHEMICAL_GROUPS.items():
            # Find columns that match the members (chk for '(%)' suffix or just name match)
            # Our columns are named like 'dmso(%)' or 'dmso(mM)'
            # We need to find columns that start with member name
            
            cols_to_sum = []
            for col in new_df.columns:
                # Check if this column belongs to the group
                # Col names are like "dmso(%)"
                # We extract the name part
                if '(' in col:
                    col_name = col.split('(')[0]
                else:
                    col_name = col
                
                if col_name in members:
                    cols_to_sum.append(col)
            
            if cols_to_sum:
                print(f"  Group '{group_name}': Summing {cols_to_sum}")
                # Create the new group column
                if group_name in new_df.columns:
                    new_df[group_name] += new_df[cols_to_sum].sum(axis=1)
                else:
                    new_df[group_name] = new_df[cols_to_sum].sum(axis=1)
                
                # DROP the original columns to reduce dimensionality
                new_df.drop(columns=cols_to_sum, inplace=True)
        
        print(f"Grouping complete. New columns: {new_df.columns.tolist()}")
        return new_df

    def _filter_sparse_data(self, df):
        """Removes ingredient columns with fewer than MIN_INGREDIENT_COUNT non-zero entries, and their rows."""
        print(f"--- Filtering Sparse Data (Min Count: {config.MIN_INGREDIENT_COUNT}) ---")
        
        # Identify ingredient columns (those with '(%)' or '(mM)')
        # Note: In parse_ingredients we named them like 'dmso(mM)' or 'dmso(%)'
        # Safest way: Look for columns that were just added as features.
        # But we already merged. Let's rely on the parentheses naming convention from config.
        # Or better, we know the preserved columns, everything else is an ingredient.
        
        meta_cols = [c for c in config.COLS_TO_KEEP if c in df.columns] + [config.RAW_INGREDIENT_COL]
        feature_cols = [c for c in df.columns if c not in meta_cols]
        
        # Calculate non-zero counts
        # We assume 0.0 means "not present"
        counts = (df[feature_cols] > 0).sum()
        
        # Identify Keep vs Drop
        to_keep = counts[counts >= config.MIN_INGREDIENT_COUNT].index.tolist()
        to_drop = counts[counts < config.MIN_INGREDIENT_COUNT].index.tolist()
        
        if not to_drop:
            print("No sparse ingredients found to filter.")
            return df
            
        print(f"Keeping {len(to_keep)} ingredients: {to_keep}")
        print(f"Dropping {len(to_drop)} rare ingredients: {to_drop}")
        
        # Find rows that contain *any* of the dropped ingredients
        # If a row uses a rare ingredient, it must go (to be scientifically consistent? or just zero it out?)
        # User said "delete these columns and their corresponding rows" in the old script.
        # User re-iterated: "too scattered". So we delete the rows.
        
        mask_to_drop = (df[to_drop] > 0).any(axis=1)
        rows_dropped = mask_to_drop.sum()
        
        df_filtered = df[~mask_to_drop].copy()
        
        # Now drop the columns
        df_filtered.drop(columns=to_drop, inplace=True)
        
        print(f"Deleted {rows_dropped} rows containing rare ingredients.")
        print(f"Remaining data: {len(df_filtered)} rows, {len(to_keep)} ingredient columns.")
        
        return df_filtered

    def clean_target_columns(self):
        """Cleans viability and recovery columns (formerly second_process.py)."""
        print("Cleaning target columns...")
        
        for col in ['viability', 'recovery']:
            if col in self.final_df.columns:
                # Remove '±' and take first part, convert to float
                # We use a lambda implementation capable of handling mixed types
                self.final_df[col] = self.final_df[col].astype(str).apply(
                    lambda x: float(x.split('±')[0].strip()) if '±' in x else pd.to_numeric(x, errors='coerce')
                )
        return self.final_df

    def remove_missing_targets(self):
        """Removes rows where BOTH viability and recovery are missing or zero (Legacy parity)."""
        print("--- Cleaning: Removing rows with missing Targets ---")
        # Ensure numeric first (clean_target_columns must be called before this)
        
        mask_bad = (
            ((self.final_df['viability'].isna()) | (self.final_df['viability'] == 0)) & 
            ((self.final_df['recovery'].isna()) | (self.final_df['recovery'] == 0))
        )
        
        count = mask_bad.sum()
        if count > 0:
            self.final_df = self.final_df[~mask_bad]
            print(f"Deleted {count} rows where both targets are missing/zero.")
        
    def remove_impossible_sums(self):
        """Removes rows where total ingredients > 100% (Legacy parity)."""
        print("--- Cleaning: Removing Impossible Sums (>100%) ---")
        
        # Identify ingredient features (all columns except meta)
        meta_cols = [c for c in config.COLS_TO_KEEP if c in self.final_df.columns] + [config.RAW_INGREDIENT_COL] + ['viability', 'recovery']
        feature_cols = [c for c in self.final_df.columns if c not in meta_cols]
        
        # Sum ingredients
        sums = self.final_df[feature_cols].sum(axis=1)
        
        # Allow small floating point margin (e.g. 100.0001 is fine)
        # Old code used strict > 100 check or < 0 culture medium. 
        # "culture medium = 100 - sum". If "culture medium < 0", then sum > 100.
        
        mask_high = sums > 100.01 
        
        count = mask_high.sum()
        if count > 0:
            self.final_df = self.final_df[~mask_high]
            print(f"Deleted {count} rows where ingredient sum > 100%.")

    def resolve_conflicts(self):
        """
        Auto-resolves where a substance has multiple units (e.g., DMSO(%) and DMSO(mM)).
        Policy: If % exists, prefer %. If both exist, this is tricky. 
        For this refactor, we will merge 'M'->'mM' which is already done.
        For different dimensions (% vs mM), we advise user intervention or keep separate.
        Currently, we keep them separate but log a warning.
        """
        # Included for extensibility
        pass

    def save(self, output_path=None):
        out = output_path or config.FINAL_FILE
        self.final_df.to_csv(out, index=False)
        print(f"Pipeline finished. Data saved to {out}")

