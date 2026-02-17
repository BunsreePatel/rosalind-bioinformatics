import os
import warnings
import pandas as pd
import string
from pathlib import Path
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

warnings.filterwarnings("ignore")

# --- PATHS ---
BASE_DIR = Path(r"C:\Users\bunsr\rosalind-bioinformatics\multispecific_antibodies\Whole_mAb")
PDB_DIR = BASE_DIR / "PDB_ColabFold_Fab_Outputs"
MASTER_CSV = BASE_DIR / "TheraSAbDab_SeqStruc_07Dec2025.csv"
OUTPUT_CSV = BASE_DIR / "Whole_mab_Structure_Based_Viscosity_Risk_Profile.csv"

if not BASE_DIR.exists():
    raise FileNotFoundError(f"No valid ROOT_DIR found at {BASE_DIR}")

"""
THEORY & BACKGROUND: STRUCTURE-BASED IN SILICO ANTIBODY VISCOSITY RISK PROFILING
===============================================

1. SOLVENT ACCESSIBLE SURFACE AREA (SASA) & THE SHRAKE-RUPLEY ALGORITHM
  - Theory: SASA measures the surface area of a biomolecule accessible to solvent. 
    A probe sphere (radius ~1.4Å, approximating water) rolls over the Van der Waals 
    surface of the protein.
  - Algorithm: Shrake-Rupley (1973) generates a mesh of points on each atom's surface 
    and tests if points remain solvent-accessible or are buried by neighbors. Surface
    exposure is calculated per atom and aggregated per residue.
  - Significance: SASA identifies surface-exposed residues in their true 3D context.
    Viscosity-driving interactions originate from exposed surface regions — not
    buried residues. All downstream patch calculations (charge and hydrophobic)
    depend on accurate SASA determination.

  - References:
    Shrake, A., & Rupley, J.A. (1973) J. Mol. Biol. 79(2), 351-371. PMID: 4760134.
    Lee, B., & Richards, F.M. (1971) J. Mol. Biol. 55(3), 379-400. PMID: 5551392.
  
2. NEGATIVE SURFACE CHARGE PATCHES & ELECTROSTATIC SELF-ASSOCIATION
  - Theory: Surface-exposed acidic residues (ASP, GLU) create local negative charge patches that can promote self-association of antibodies, especially at low ionic strength. The magnitude, density, and spatial clustering of negative charges are key predictors of electrostatic-driven viscosity.
  - Calculation: Sum per-residue charges using a pH-dependent scale (e.g., +1 for ARG/LYS, -1 for ASP/GLU, HIS partially protonated). Compute contiguous negative patches along the Fab surface (using SASA > threshold) with a sliding window to estimate patch size and density.
  - Significance: Larger contiguous negative patches correlate with higher electrostatic repulsion when isolated, but if unevenly distributed, they can create “sticky” regions that drive intermolecular interactions, increasing solution viscosity.

  - References:
    Liu, J., Nguyen, M.D.H., Andya, J.D., Shire, S.J. (2005) J. Pharm. Sci. 94(9), 1928-1940. PMID: 16052543.
    Yadav, S., Shire, S.J., Kalonia, D.S. (2012) J. Pharm. Sci. 101(3), 998-1011. PMID: 22113861.
    
3. HYDROPHOBIC SURFACE PATCHES & INTERMOLECULAR ASSOCIATION
  - Theory: Surface-exposed hydrophobic residues (ALA, VAL, ILE, LEU, MET, PHE, TYR, TRP) form contiguous patches that promote nonpolar interactions between antibody molecules. Aggregation propensity and viscosity rise with the size of these hydrophobic clusters.
  - Calculation: Use SASA values to identify surface-exposed hydrophobic residues. Apply a sliding window (e.g., PATCH_WINDOW = 5 residues) to sum SASA within local patches, reporting the largest contiguous hydrophobic area per chain or Fab.
  - Significance: Hydrophobic patch size, not total hydrophobicity, is a strong predictor of solution viscosity and aggregation risk. Critical for identifying “sticky” zones on the Fab surface.
 
  - References:
    Yadav, S., Shire, S.J., Kalonia, D.S. (2010) J. Pharm. Sci. 99(12), 4812-4829. PMID: 20821382.
    Hung, J.J., Dear, B.J., Mitra, N., et al. (2018) Pharm. Res. 35(4), 74. PMID: 29713822.
  
4. COMBINED STRUCTURE-BASED VISCOSITY METRICS
  - Theory: Viscosity arises from the interplay of electrostatic and hydrophobic interactions on the antibody surface. Key determinants include:
    * Total exposed negative charge
    * Total exposed hydrophobic surface
    * Maximal hydrophobic patch SASA
    * Maximal negative patch size
    * Net charge asymmetry between chains
  - Usage: Combine per-chain or per-Fab metrics into a predictive score or classification (High / Moderate / Low viscosity risk). These metrics allow concentration-dependent viscosity predictions when coupled with experimental scaling factors.

  - References:
    Tomar, D.S., Kumar, S., Balasubramanian, S., et al. (2016) mAbs 8(2), 216-228. PMID: 26736022.
    Hung, J.J., Zeno, W.F., et al. (2019) J. Phys. Chem. B 123(1), 10818-10827.

5. CONCENTRATION-DEPENDENT VISCOSITY MODELING
  - Theory: Solution viscosity increases with protein concentration due to more frequent intermolecular interactions. For antibodies, both electrostatic (charge patches) and hydrophobic (sticky patches) surfaces contribute to non-ideal solution behavior.
  - Modeling Approach: 
    * Use per-Fab structural features (SASA, hydrophobic patch, negative patch, net charge) as input predictors.
    * Apply scaling laws or semi-empirical relationships (e.g., Einstein or Krieger-Dougherty equations) to estimate viscosity η as a function of concentration C:
        η(C) = η_0 * (1 + k_visc * C + …)
        where η_0 is solvent viscosity, and k_visc is an effective interaction parameter derived from surface patch metrics.
    * Alternatively, machine learning or regression models can be trained on experimental viscosity data using these structural features.
  - Significance: Provides a predictive map of viscosity across therapeutic-relevant concentrations (typically 50-200 mg/mL). Enables early identification of high-risk antibodies that may require formulation adjustments (e.g., ionic strength, pH, excipients).
  - Notes:
    * Accurate SASA and patch calculations are critical—buried residues do not contribute.
    * Hydrophobic patches tend to dominate at high concentration, while charge patches dominate at low ionic strength.
    * Multiplying patch effects by concentration allows relative ranking of viscosity risk across different mAbs.
 
  - References:
    Krieger, I.M., & Dougherty, T.J. (1959) Trans. Soc. Rheol. 3, 137-152.
    Roberts, C.J. (2014) Curr. Opin. Biotechnol. 30, 211-217. PMID: 25173826.
    Yadav, S., Shire, S.J., Kalonia, D.S. (2012) J. Pharm. Sci. 101(3), 998-1011. PMID: 22113861.  
    
"""

# --- BIOPHYSICAL SCALES ---
STANDARD_AAS = sorted("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC_RESIDUES = ['ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP']
CHARGE_SCALE = {
    'ARG': 1.0, 'LYS': 1.0, 'HIS': 0.1,  # Positive
    'ASP': -1.0, 'GLU': -1.0,            # Negative
}
PATCH_WINDOW = 5  # Sliding window for hydrophobic patches (examines 5 consecutive residues at a time, if all 5 residues are hydrophobic, the segment is considered a "patch")
"""The literature (Sharma 2014, Jain 2017) often uses 4-6 residues for hydrophobic patch detection.
5 is a reasonable compromise: small enough to catch meaningful patches, large enough to ignore tiny “random” clusters."""

# --- PDB STRUCTURE LOADING & SASA COMPUTATION ---
def load_pdb_structure(pdb_path: Path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, pdb_path)
    return structure

def compute_sasa(structure):
    sr = ShrakeRupley()
    sr.compute(structure, level="R")  # level="R" attaches .sasa to residues
    return structure

def extract_chain_features(structure):
    chain_features = {}
    chain_residues = {}

    for model in structure:
        for chain in model:
            c_id = chain.get_id()
            residues = list(chain.get_residues())

            total_sasa = sum(getattr(res, 'sasa', 0) for res in residues)
            hydro_sasa = sum(getattr(res, 'sasa', 0) for res in residues if res.get_resname() in HYDROPHOBIC_RESIDUES)
            net_charge = sum(CHARGE_SCALE.get(res.get_resname(), 0) for res in residues)

            # Max contiguous hydrophobic patch
            max_patch_sasa = 0
            for i in range(len(residues) - PATCH_WINDOW + 1):
                window = residues[i:i + PATCH_WINDOW]
                if all(r.get_resname() in HYDROPHOBIC_RESIDUES for r in window):
                    patch_sasa = sum(getattr(r, 'sasa', 0) for r in window)
                    max_patch_sasa = max(max_patch_sasa, patch_sasa)

            chain_features[c_id] = {
                "Total_SASA": round(total_sasa, 2),
                "Hydro_SASA": round(hydro_sasa, 2),
                "Net_Charge": round(net_charge, 1),
                "Max_Hydro_Patch": round(max_patch_sasa, 2),
            }

    return chain_features

def aggregate_fab_features(chain_features):

    fab_total_sasa = sum(ch["Total_SASA"] for ch in chain_features.values())
    fab_hydro_sasa = sum(ch["Hydro_SASA"] for ch in chain_features.values())
    fab_net_charge = sum(ch["Net_Charge"] for ch in chain_features.values())
    fab_max_hydro_patch = max(ch["Max_Hydro_Patch"] for ch in chain_features.values())
    total_residues = sum(ch["Num_Residues"] for ch in chain_features.values())
    
    fab_features = {
        "Fab_Total_SASA": round(fab_total_sasa, 2),
        "Fab_Hydro_SASA": round(fab_hydro_sasa, 2),
        "Fab_Net_Charge": round(fab_net_charge, 1),
        "Fab_Max_Hydro_Patch": round(fab_max_hydro_patch, 2),
        "ChA_Length": chain_features.get("A", {}).get("Num_Residues", 0),
        "ChB_Length": chain_features.get("B", {}).get("Num_Residues", 0),
        "Fab_Total_Residues": total_residues,        
    }
    
    # Also include individual chain features if desired
    for c_id, feats in chain_features.items():
        for key, val in feats.items():
            fab_features[f"Ch_{c_id}_{key}"] = val
    
    return fab_features

def run_3D_viscosity_analysis():
    if not PDB_DIR.exists():
        print(f"ERROR: PDB directory not found at {PDB_DIR}")
        return
    
    df_master = pd.read_csv(MASTER_CSV)
    df_master['key'] = df_master['Therapeutic'].str.lower().str.replace(" ", "").str.strip()

    pdb_files = list(PDB_DIR.glob("*.pdb"))
    print(f"Found {len(pdb_files)} PDB files in {PDB_DIR}")

    master_features= []

    for pdb_path in tqdm(pdb_files, desc="Processing PDBs"):
        try:
            structure = load_pdb_structure(pdb_path)
            structure = compute_sasa(structure)
            chain_feats = extract_chain_features(structure)
            fab_feats = aggregate_fab_features(chain_feats)

            therapeutic_name = pdb_path.stem.replace("_fab", "")
            fab_feats["Therapeutic"] = therapeutic_name
            fab_feats["key"] = therapeutic_name.lower().replace("_fab", "").replace(" ", "").strip()

            master_features.append(fab_feats)
        except Exception as e:
            print(f"Error processing {pdb_path.name}: {e}")

    if not master_features:
        print("No features extracted, exiting.")
        return
    
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
    print(f"\n=== SUCCESS: Viscosity risk profile extracted for {len(df)} mAbs ===")
    print(f"File saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_3D_viscosity_analysis()