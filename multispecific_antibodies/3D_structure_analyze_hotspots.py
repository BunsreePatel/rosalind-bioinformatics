import os
import re
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from Bio.PDB import PDBParser, ShrakeRupley, DSSP
from Bio.SeqUtils import seq1

warnings.filterwarnings("ignore")

"""
Fab Hotspot Analysis - Structure-Based Features
Source: Antibodies (Basel) 2019 Dec 3;8(4):55 - Table 2
Features: SASA (ShrakeRupley), B-factor, Secondary Structure (DSSP), CDR vs Framework (anarci)
"""

# --- PATHS ---
BASE_DIR = Path(r"C:\Users\meeko\rosalind-bioinformatics\multispecific_antibodies\Whole_mAb")
PDB_DIR = BASE_DIR / "PDB_ColabFold_Fab_Outputs"
OUTPUT_CSV = BASE_DIR / "Whole_mab_Structure_Based_Hotspots_Analysis.csv"
MASTER_CSV = BASE_DIR / "TheraSAbDab_SeqStruc_07Dec2025.csv"

# --- MOTIFS ---
HOTSPOT_MOTIFS = {
    'N_Glycosylation':              r'N[^P][ST]',   # NxS/T sequon - glycosylation & deamidation risk
    'Asn_Deamidation':              r'NG',           # Asn-Gly - highest risk deamidation
    'Asp_Isomerization':            r'DG',           # Asp-Gly - backbone isomerization
    'Gln_Deamidation':              r'Q[NGS]',       # Gln deamidation (slower than Asn)
    'Cys_Oxidation':                r'C',            # Cysteinylation / disulfide risk
    'His_Oxidation':                r'H',            # His crosslinking / oxidation
    'Met_Oxidation':                r'M',            # Met oxidation
    'Trp_Oxidation':                r'W',            # Trp aromaticity loss / oxidation
    'Asp_Pro_Amide_Hydrolysis':     r'DP',           # Asp-Pro amide bond hydrolysis
    'N_Terminal_PyroGlu':           r'^[EQ]',        # N-terminal Glu/Gln pyroglutamate formation
}

# --- STRUCTURE LOADING ---
def load_pdb_structure(pdb_path: Path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, pdb_path)
    return structure[0]

# --- SEQUENCE EXTRACTION ---
def get_chain_residues(structure):
    """Returns dict of chain_id -> list of "(residue, 1-letter-code, seq_index)"""
    from Bio.PDB.Polypeptide import is_aa
    chains = {}
    for chain in structure:
        residues = [r for r in chain if is_aa(r, standard=True)]
        seq = "".join([seq1(r.get_resname()) for r in residues])
        chains[chain.id] = {"residues": residues, "sequence": seq}
    return chains

# --- SASA via SHRAKERUPLEY (SURFACE EXPOSURE OF EACH MOTIF HIT, FILTERS BURIED VS EXPOSED HITS) ---
def compute_sasa(structure):
    """Returns SASA dict keyed by (chain_id, res_id)"""
    sr = ShrakeRupley()
    sr.compute(structure, level="R")
    sasa = {}
    for chain in structure:
        for res in chain:
            sasa[(chain.id, res.get_id())] = round(res.sasa, 3)
    return sasa

# --- B-FACTOR via PDBPARSER (LOCAL FLEXIBILITY AT MOTIF SITE) ---
def get_bfactor(residue):
    """Mean B-factor across all atoms in residue"""
    bfactors = [atom.get_bfactor() for atom in residue]
    return round(np.mean(bfactors), 3) if bfactors else None

# --- SECONDARY STRUCTURE AT MOTIF via DSSP (WHETHER MOTIF SITS IN LOOP, HELIX, SHEET) ---
DSSP_MAP = {
    'H': 'Helix', 'G': 'Helix', 'I': 'Helix',
    'E': 'Sheet', 'B': 'Sheet',
    'T': 'Loop',  'S': 'Loop',  '-': 'Loop', ' ': 'Loop'
}

def compute_dssp(structure, pdb_path: Path):
    """Returns per-residue secondary structure dict keyed by (chain_id, res_id)"""
    dssp = DSSP(structure, pdb_path, dssp='mkdssp')
    secondarystructure = {}
    for key in dssp.keys():
        chain_id, res_id = key
        record = dssp[key]
        raw_secondarystructure = record[2]
        secondarystructure[(chain_id, res_id)] = DSSP_MAP.get(raw_secondarystructure, 'Loop')
    return secondarystructure

# --- CDR vs FRAMEWORK LOCATION via ANARCI (WHETHER HIT IS IN CDR = HIGHER RISK)
def get_cdr_annotations(chains: dict, scheme: str = 'chothia'):
    """
    Uses anarci to number each chain and annotate CDR vs framework.
    Returns dict: chain_id -> list of (seq_index, region)
    where region is 'CDR1','CDR2','CDR3','FR1','FR2','FR3','FR4'
    """
    try:
        from anarci import anarci
    except ImportError:
        raise ImportError("anarci not installed. Run: pip install anarci")

    CDR_RANGES_CHOTHIA = {
        'H': {'CDR1': (26, 35), 'CDR2': (52, 56), 'CDR3': (95, 102)},
        'L': {'CDR1': (24, 34), 'CDR2': (50, 56), 'CDR3': (89, 97)},
    }

    annotations = {}
    for chain_id, data in chains.items():
        seq = data['sequence']
        results, _, _ = anarci(
            [(chain_id, seq)], scheme=scheme, output=False
        )
        if not results or results[0] is None:
            annotations[chain_id] = ['Unknown'] * len(seq)
            continue

        numbered = results[0][0][0]  # list of ((pos, insert), aa)
        chain_type = results[0][0][1]  # 'H' or 'L'
        cdr_ranges = CDR_RANGES_CHOTHIA.get(chain_type, {})

        region_list = []
        for (pos, insert), aa in numbered:
            if aa == '-':
                continue
            region = 'Framework'
            for cdr_name, (start, end) in cdr_ranges.items():
                if start <= pos <= end:
                    region = cdr_name
                    break
            region_list.append(region)

        annotations[chain_id] = region_list
    return annotations

# --- MOTIF SCANNING ---
def scan_motifs(sequence: str, motifs: dict):
    """Returns list of (motif_name, start, end, matched_seq)"""
    hits = []
    for name, pattern in motifs.items():
        for m in re.finditer(pattern, sequence):
            hits.append((name, m.start(), m.end(), m.group()))
    return hits

# --- MAIN: COMBINE ALL CALCULATIONS TOGETHER ---
def analyze_hotspots(pdb_path: Path, sasa_threshold: float = 20.0):
    """
    Run full structure-based hotspot analysis.
    Returns list of dicts, one per motif hit.
    """
    structure = load_pdb_structure(pdb_path)
    chains = get_chain_residues(structure)
    sasa_map = compute_sasa(structure)
    secondarystructure_map = compute_dssp(structure, pdb_path)
    cdr_map = get_cdr_annotations(chains)

    results = []

    for chain_id, data in chains.items():
        residues = data['residues']
        sequence = data['sequence']
        cdr_annotations = cdr_map.get(chain_id, ['Unknown'] * len(residues))

        hits = scan_motifs(sequence, HOTSPOT_MOTIFS)

        for motif_name, start, end, matched_seq in hits:
            hit_residues = residues[start:end]

            # FEATURE 1: mean SASA across motif span
            sasa_vals = [
                sasa_map.get((chain_id, r.get_id()), 0.0)
                for r in hit_residues
            ]
            mean_sasa = round(np.mean(sasa_vals), 3) if sasa_vals else None
            surface_exposed = mean_sasa >= sasa_threshold if mean_sasa is not None else None

            # FEATURE 2: mean B-factor across motif span
            mean_bfactor = round(
                np.mean([get_bfactor(r) for r in hit_residues if get_bfactor(r) is not None]), 3
            )

            # FEATURE 3: secondary structure at first residue of motif
            anchor_res_id = hit_residues[0].get_id() if hit_residues else None
            secondary_structure = secondarystructure_map.get((chain_id, anchor_res_id), 'Unknown')

            # FEATURE 4: CDR vs framework at motif start position
            cdr_location = cdr_annotations[start] if start < len(cdr_annotations) else 'Unknown'

            results.append({
                'chain':              chain_id,
                'motif':              motif_name,
                'position_start':     start,
                'position_end':       end,
                'matched_sequence':   matched_seq,
                'mean_sasa':          mean_sasa,
                'surface_exposed':    surface_exposed,
                'mean_bfactor':       mean_bfactor,
                'secondary_structure':secondary_structure,
                'cdr_location':       cdr_location,
            })

    return results

def run_hotspot_analysis():
    if not PDB_DIR.exists():
        print(f"ERROR: PDB directory not found at {PDB_DIR}")
        return

    df_master = pd.read_csv(MASTER_CSV)
    df_master['key'] = df_master['Therapeutic'].str.lower().str.replace(" ", "").str.strip()

    pdb_files = list(PDB_DIR.glob("*.pdb"))
    print(f"Found {len(pdb_files)} PDB files in {PDB_DIR}")

    master_features = []

    for pdb_path in tqdm(pdb_files, desc="Processing PDBs"):
        try:
            hits = analyze_hotspots(pdb_path)
            for hit in hits:
                therapeutic_name = pdb_path.stem.replace("_fab", "")
                hit["Therapeutic"] = therapeutic_name
                hit["key"] = therapeutic_name.lower().replace("_fab", "").replace(" ", "").strip()
                master_features.append(hit)
        except Exception as e:
            print(f"Error processing {pdb_path.name}: {e}")

    if not master_features:
        print("No features extracted, exiting.")
        return

    df = pd.DataFrame(master_features)

    df['key'] = df['Therapeutic'].str.lower().str.replace("_fab", "").str.replace(" ", "").str.strip()
    df_master['key'] = df_master['Therapeutic'].str.lower().str.replace(" ", "").str.strip()

    df = pd.merge(df, df_master[['key', 'CH1 Isotype', 'VD LC']], on='key', how='left')

    df.drop(columns=['key'], inplace=True)

    cols = ['Therapeutic', 'CH1 Isotype', 'VD LC', 'motif', 'chain', 'position_start', 'position_end',
            'matched_sequence', 'mean_sasa', 'surface_exposed', 'mean_bfactor',
            'secondary_structure', 'cdr_location'] + \
           [c for c in df.columns if c not in ['Therapeutic', 'CH1 Isotype', 'VD LC', 'motif', 'chain',
            'position_start', 'position_end', 'matched_sequence', 'mean_sasa',
            'surface_exposed', 'mean_bfactor', 'secondary_structure', 'cdr_location']]
    df = df[cols]

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n=== SUCCESS: Hotspot analysis extracted for {len(df)} motif hits across {len(pdb_files)} mAbs ===")
    print(f"File saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_hotspot_analysis()