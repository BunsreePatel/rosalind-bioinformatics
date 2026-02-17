import os
import warnings
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

warnings.filterwarnings("ignore")

# --- PATHS (Targeting your verified folder) ---
BASE_DIR = Path(r"C:\Users\bunsr\rosalind-bioinformatics\multispecific_antibodies\Whole_mAb")
PDB_DIR = BASE_DIR / "PDB_ColabFold_Fab_Outputs"
MASTER_CSV = BASE_DIR / "TheraSAbDab_SeqStruc_07Dec2025.csv"
OUTPUT_CSV = BASE_DIR / "Whole_mAb_3D_Physics_Features_Filtered.csv"

"""
THEORY & BACKGROUND: 3D BIOPHYSICAL PROFILING (VALIDATED METRICS)

1. SOLVENT ACCESSIBLE SURFACE AREA (SASA) & THE SHRAKE-RUPLEY ALGORITHM
   - Theory: SASA measures the surface area of a biomolecule accessible to solvent. 
     A probe sphere (radius ~1.4Å, approximating water) rolls over the Van der Waals 
     surface of the protein.
   - Algorithm: Shrake-Rupley (1973) generates a mesh of points on each atom's surface 
     and tests if points remain solvent-accessible or are buried by neighbors.
   - Significance: SASA distinguishes buried vs. surface-exposed residues in the 3D 
     folded context. Surface-exposed hydrophobic residues create aggregation hotspots.

2. HYDROPHOBIC SURFACE PATCHES & AGGREGATION PROPENSITY
   - Theory: Aggregation is driven by localized hydrophobic patches, not bulk 
     hydrophobicity. Contiguous surface regions of hydrophobic residues (ALA, VAL, 
     ILE, LEU, MET, PHE, TYR, TRP) create "sticky" zones that promote intermolecular 
     association.
   - Metrics Calculated:
     * Max_Hydrophobic_Patch: Largest contiguous hydrophobic surface area (5-residue 
       sliding window) - correlates with experimental aggregation propensity
   - Literature Basis: Sharma et al. (2014), Jain et al. (2017) - patch size matters 
     more than total surface hydrophobicity for aggregation

3. NET CHARGE & CHARGE SYMMETRY
   - Theory: At pH 7.4, net charge determines electrostatic repulsion between antibody 
     molecules (DLVO theory). Charge asymmetry between chains creates dipole moments 
     that can promote self-association.
   - Metrics Calculated:
     * Net_Charge: Total ionizable residues (ARG, LYS, HIS positive; ASP, GLU negative)
   - Literature Basis: Therapeutic Antibody Profiler (TAP) guidelines - charge asymmetry 
     thresholds correlate with viscosity and clearance
"""

# --- BIOPHYSICAL SCALES ---
HYDROPHOBIC_RESIDUES = ['ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP']
CHARGE_SCALE = {
    'ARG': 1.0, 'LYS': 1.0, 'HIS': 0.1,  # Positive
    'ASP': -1.0, 'GLU': -1.0,            # Negative
}
PATCH_WINDOW = 5  # Sliding window for hydrophobic patches (examines 5 consecutive residues at a time, if all 5 residues are hydrophobic, the segment is considered a "patch")
"""The literature (Sharma 2014, Jain 2017) often uses 4-6 residues for hydrophobic patch detection.
5 is a reasonable compromise: small enough to catch meaningful patches, large enough to ignore tiny “random” clusters."""

def run_3D_physics_analysis():
    if not PDB_DIR.exists():
        print(f"ERROR: PDB directory not found at {PDB_DIR}")
        return
    
    # Read MASTER_CSV
    df_master = pd.read_csv(MASTER_CSV)
    df_master['key'] = df_master['Therapeutic'].str.lower().str.strip()

    # Map available PDBs (lowercase for easy matching)
    pdb_map = {}
    for f in PDB_DIR.glob("*.pdb"):
      key = f.stem.replace("_fab", "").lower() # remove _fab in .pdb filename
      pdb_map[key] = f

    print(f"Found {len(pdb_map)} PDB files ready for analysis.")

    # Setup tools
    parser = PDBParser(QUIET=True)
    sr = ShrakeRupley()
    master_features = []

    # Process every PDB found
    print(f"--- EXTRACTING PHYSICS DATA ---")
    for ab_name, pdb_path in tqdm(pdb_map.items()):
        try:
            struct = parser.get_structure(ab_name, str(pdb_path))
            sr.compute(struct, level="R") # Residue-level SASA
            
            fab_total_sasa, fab_hydro_sasa, fab_total_charge = 0, 0, 0
            chain_stats = {}
            chain_residues = {}

            # Analyze every chain (A, B)
            for model in struct:
                for chain in model:
                    c_id = chain.get_id()
                    c_sasa, c_hydro, c_charge = 0, 0, 0
                    max_patch = 0
                    
                    residues =list(chain.get_residues())
                    c_residues = len(residues)
                    chain_residues[c_id] = c_residues 

                    for res in residues:
                        res_name = res.get_resname()
                        res_sasa = getattr(res, 'sasa', 0)
                        c_sasa += res_sasa
                        c_charge += CHARGE_SCALE.get(res_name, 0)
                        if res_name in HYDROPHOBIC_RESIDUES:          
                            c_hydro += res_sasa

                    # Hydrophobic patch detection
                    for i in range(len(residues) - PATCH_WINDOW + 1):
                        window = residues[i:i+PATCH_WINDOW]
                        if all(r.get_resname() in HYDROPHOBIC_RESIDUES for r in window):
                            patch_sasa = sum(getattr(r, 'sasa', 0) for r in window)
                            max_patch = max(max_patch, patch_sasa)

                    # Store individual chain features
                    chain_stats[f"Ch_{c_id}_Max_Patch"] = round(max_patch, 2)
                    chain_stats[f"Ch_{c_id}_SASA"] = round(c_sasa, 2)
                    chain_stats[f"Ch_{c_id}_Charge"] = round(c_charge, 1)
                    
                    fab_total_sasa += c_sasa
                    fab_hydro_sasa += c_hydro
                    fab_total_charge += c_charge

            # Chain residue summary
            total_residues = sum(chain_residues.values())
            chA_length = chain_residues.get('A', 0)
            chB_length = chain_residues.get('B', 0)
            total_length = total_residues

            # Match antibody names
            ab_name_clean = ab_name.replace('_fab', '').capitalize()

            # Compile the data for this mAb
            entry = {
                "Therapeutic": ab_name.capitalize(),
                "Fab_Total_SASA": round(fab_total_sasa, 2),
                "Fab_Hydro_SASA": round(fab_hydro_sasa, 2),
                "Fab_Net_Charge": round(fab_total_charge, 1),
                "Ch_A_Length": chA_length,
                "Ch_B_Length": chB_length,
                "Fab_Total_AAs": total_length
            }
            entry.update(chain_stats)
            master_features.append(entry)

        except Exception as e:
            print(f"Error processing {ab_name}: {e}")

    if master_features:
        df = pd.DataFrame(master_features)

        # create matching keys (strip _fab, lowercase, remove spaces)
        df['key'] = df['Therapeutic'].str.lower().str.replace("_fab", "").str.replace(" ", "").str.strip()
        df_master['key'] = df_master['Therapeutic'].str.lower().str.replace(" ", "").str.strip()

        # merge on the cleaned 'key' column
        df = pd.merge(df, df_master[['key', 'CH1 Isotype', 'VD LC']], on='key', how='left')

        # drop the temporary key column
        df.drop(columns=['key'], inplace=True)

        # reorder columns
        cols = ['Therapeutic', 'CH1 Isotype', 'VD LC'] + [c for c in df.columns if c not in ['Therapeutic', 'CH1 Isotype', 'VD LC']]
        df = df[cols]

        # save final CSV
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n=== SUCCESS: Physics extracted for {len(df)} mAbs ===")
        print(f"File saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    run_3D_physics_analysis()