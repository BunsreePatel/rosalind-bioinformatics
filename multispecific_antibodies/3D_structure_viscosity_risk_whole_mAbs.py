import os
import warnings
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage
from collections import Counter

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
1. Viscosity (η) in protein solutions increases with concentration due to intermolecular forces. 
For monoclonal antibodies (mAbs), high viscosity (>20 cP at 150 mg/mL) is a major developability issue, often linked to self-association or aggregation.
Structure-based prediction focuses on surface features that mediate these interactions, as buried residues don't contribute. 
The tiers build on each other: Tier 1 is fast/computational but approximate; Tier 2 is more accurate but computationally intensive; Tier 3 is the most predictive but data-dependent.
    - Tier 1 -


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

# --- BIOPHYSICAL SCALES AND CONSTANTS ---
STANDARD_AAS = sorted("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC_RESIDUES = ['ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP']
POSITIVE_RESIDUES = ['ARG', 'LYS', 'HIS']
NEGATIVE_RESIDUES = ['ASP', 'GLU']
CHARGE_SCALE = {
    'ARG': 1.0, 'LYS': 1.0, 'HIS': 0.1,  # Positive
    'ASP': -1.0, 'GLU': -1.0,            # Negative
}
SASA_THRESHOLD = 5.0  # Å² Minimum SASA for surface exposure **NEED TO LOOK THIS UP IN LITERATURE
CLUSTER_DISTANCE = 10.0  # Å for spatial patch clustering **NEED TO LOOK THIS UP IN LITERATURE

def load_pdb_structure(pdb_path: Path):
    # Load a .pdb structure file using Bio.PDB.
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, pdb_path)
    return structure

def compute_sasa(structure):
    # Compute Solvent Accessible Surface Area (SASA) for the structure
    sr = ShrakeRupley()
    sr.compute(structure, level="R")  # level="R" attaches .sasa to residues
    return structure

def get_residue_center(residue):
    # Get the center of mass of a residue.
    atoms = list(residue.get_atoms())
    if not atoms:
        return None
    coords = np.array([atom.get_coord() for atom in atoms])
    return np.mean(coords, axis=0)

def calculate_spatial_hydrophobic_patches(residues, distance_cutoff=CLUSTER_DISTANCE, sasa_threshold=SASA_THRESHOLD):
    # Calculate hydrophobic patches using 3D spatial clustering; returns the maximum patch SASA.
    surface_hydros = [(i, getattr(res, 'sasa', 0), get_residue_center(res)) 
                     for i, res in enumerate(residues) 
                     if res.get_resname() in HYDROPHOBIC_RESIDUES and getattr(res, 'sasa', 0) > sasa_threshold and get_residue_center(res) is not None]
    
    if len(surface_hydros) < 2:
        return 0.0
    
    # Extract 3D centers and cluster
    centers = np.array([h[2] for h in surface_hydros])
    
    # Find clusters using distance cutoff
    if len(centers) > 1:
        Z = linkage(centers, method='single')
        clusters = fcluster(Z, distance_cutoff, criterion='distance')
    else:
        clusters = [1]
    
    # Calculate max patch SASA
    cluster_sasas = {}
    for idx, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_sasas:
            cluster_sasas[cluster_id] = 0
        cluster_sasas[cluster_id] += surface_hydros[idx][1]
    
    return max(cluster_sasas.values()) if cluster_sasas else 0.0

def calculate_spatial_pos_charge_patches(residues, distance_cutoff=CLUSTER_DISTANCE, sasa_threshold=SASA_THRESHOLD):
    # Calculate charge patches using 3D spatial clustering; returns the maximum positive patch size (number of residues).
    surface_pos = [(i, res.get_resname(), get_residue_center(res)) 
                      for i, res in enumerate(residues) 
                      if res.get_resname() in POSITIVE_RESIDUES and getattr(res, 'sasa', 0) > sasa_threshold and get_residue_center(res) is not None]
    
    if len(surface_pos) < 2:
        return 0

    centers = np.array([c[2] for c in surface_pos])
    #CHECK THIS FORMULA [1] OR [2]??
    if len(centers) > 1:
        Z = linkage(centers, method='single')
        clusters = fcluster(Z, distance_cutoff, criterion='distance')
    else:
        clusters = [1]
    
    cluster_sizes = Counter(clusters)
    return max(cluster_sizes.values()) if cluster_sizes else 0


def calculate_spatial_neg_charge_patches(residues, distance_cutoff=CLUSTER_DISTANCE, sasa_threshold=SASA_THRESHOLD):
    # Calculate charge patches using 3D spatial clustering; returns the maximum negative patch size (number of residues).
    surface_neg = [(i, res.get_resname(), get_residue_center(res)) 
                      for i, res in enumerate(residues) 
                      if res.get_resname() in NEGATIVE_RESIDUES and getattr(res, 'sasa', 0) > sasa_threshold and get_residue_center(res) is not None]
    
    if len(surface_neg) < 2:
        return 0

    centers = np.array([c[2] for c in surface_neg])
    #CHECK THIS FORMULA [1] OR [2]??
    if len(centers) > 1:
        Z = linkage(centers, method='single')
        clusters = fcluster(Z, distance_cutoff, criterion='distance')
    else:
        clusters = [1]
    
    cluster_sizes = Counter(clusters)
    return max(cluster_sizes.values()) if cluster_sizes else 0

def calculate_charge_dipole_moment(residues, distance_cutoff=CLUSTER_DISTANCE, sasa_threshold=SASA_THRESHOLD):
    charges = []
    positions = []
    for res in residues:
        charge = CHARGE_SCALE.get(res.get_resname(), 0)
        sasa = getattr(res, 'sasa', 0)
        if charge != 0 and sasa > SASA_THRESHOLD:
            center = get_residue_center(res)
            if center is not None:
                charges.append(charge)
                positions.append(center)
    
    if not charges:
        return 0.0
    
    charges = np.array(charges)
    positions = np.array(positions)
    
    total_charge = sum(charges)
    if total_charge == 0:
        return 0.0
    
    com = np.average(positions, weights=np.abs(charges))
    dipole = sum(charges[i] * (positions[i] - com) for i in range(len(charges)))
    return np.linalg.norm(dipole)

def identify_cdr_regions(chain_residues, chain_id):
    cdrs = []
    for res in chain_residues:
        res_id = res.get_id()[1]
        if chain_id.upper() in ['H', 'A']:
            if 26 <= res_id <= 32 or 52 <= res_id <= 56 or 95 <= res_id <= 102:
                cdrs.append(res)
        elif chain_id.upper() in ['L', 'B']:
            if 24 <= res_id <= 34 or 50 <= res_id <= 56 or 89 <= res_id <= 97:
                cdrs.append(res)
    return cdrs

def calculate_viscosity_risk_score(fab_features):
    weight_hydro_patch = 0.3
    weight_neg_patch = 0.2
    weight_dipole = 0.2
    weight_cdr_hydro = 0.15
    weight_cdr_charge = 0.15
    
    hydro_patch_norm = fab_features['Fab_Max_Hydro_Patch'] / 1000
    neg_patch_norm = fab_features['Fab_Max_Neg_Patch'] / 20
    dipole_norm = fab_features['Fab_Charge_Dipole'] / 100
    cdr_hydro_norm = fab_features['CDR_Hydro_SASA'] / 2000
    cdr_charge_norm = abs(fab_features['CDR_Net_Charge']) / 10
    
    score = (weight_hydro_patch * hydro_patch_norm +
             weight_neg_patch * neg_patch_norm +
             weight_dipole * dipole_norm +
             weight_cdr_hydro * cdr_hydro_norm +
             weight_cdr_charge * cdr_charge_norm)
    
    #***NEED TO CHECK THESE THRESHOLD, COULD BE ARBITRARY
    risk = "High" if score > 0.6 else "Moderate" if score > 0.3 else "Low"
    return round(score, 3), risk

def estimate_viscosity_vs_concentration(fab_features, concentrations=None):
    # Estimate viscosity vs. concentration using Ross-Minton-inspired model; returns dict of η/η₀ at given concentrations (default: 50, 100, 150 mg/mL). 
    # Based on Yadav et al. (2012): k_visc derived from surface features.
    if concentrations is None:
        concentrations = [50, 100, 150]     # mg/mL
    
    # Derive k_visc from features(empirical_weights)
    k_visc = (0.01 * fab_features['Fab_Max_Hydro_Patch'] / 1000 +  # Hydrophobic drive
              0.005 * fab_features['Fab_Max_Neg_Patch'] / 20 +     # Electrostatic
              0.002 * fab_features['Fab_Charge_Dipole'] / 100)     # Anisotropy
    
    # Ross-Minton approximation for dilute: η/η₀ = 1 + k_visc * C
    viscosity_ratios = {c: 1 + k_visc * c for c in concentrations}

    # Flag high risk if η/η₀ > 2 at 150 mg/mL (common threshold)
    high_risk_conc = [c for c, ratio in viscosity_ratios.items() if ratio > 2]
    #***CHECK THE THRESHOLD
    
    return {
        "Viscosity_Ratios": viscosity_ratios,
        "High_Risk_Concentrations": high_risk_conc,
        "k_visc": round(k_visc, 4)
    }

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
            
            max_hydro_patch = calculate_spatial_hydrophobic_patches(residues)
            max_pos_patch = calculate_spatial_pos_charge_patches(residues)
            max_neg_patch = calculate_spatial_neg_charge_patches(residues)
            dipole = calculate_charge_dipole_moment(residues)
            
            cdr_residues = identify_cdr_regions(residues, c_id)
            cdr_hydro_sasa = sum(getattr(res, 'sasa', 0) for res in cdr_residues if res.get_resname() in HYDROPHOBIC_RESIDUES)
            cdr_net_charge = sum(CHARGE_SCALE.get(res.get_resname(), 0) for res in cdr_residues)
            
            chain_features[c_id] = {
                "Total_SASA": round(total_sasa, 2),
                "Hydro_SASA": round(hydro_sasa, 2),
                "Net_Charge": round(net_charge, 1),
                "Max_Hydro_Patch": round(max_hydro_patch, 2),
                "Max_Pos_Patch": max_pos_patch,
                "Max_Neg_Patch": max_neg_patch,
                "Charge_Dipole": round(dipole, 2),
                "CDR_Hydro_SASA": round(cdr_hydro_sasa, 2),
                "CDR_Net_Charge": round(cdr_net_charge, 1),
                "Num_Residues": len(residues),
            }

    return chain_features

def aggregate_fab_features(chain_features):
    fab_total_sasa = sum(ch["Total_SASA"] for ch in chain_features.values())
    fab_hydro_sasa = sum(ch["Hydro_SASA"] for ch in chain_features.values())
    fab_net_charge = sum(ch["Net_Charge"] for ch in chain_features.values())
    fab_max_hydro_patch = max(ch["Max_Hydro_Patch"] for ch in chain_features.values())
    fab_max_pos_patch = max(ch["Max_Pos_Patch"] for ch in chain_features.values())
    fab_max_neg_patch = max(ch["Max_Neg_Patch"] for ch in chain_features.values())
    fab_dipole = max(ch["Charge_Dipole"] for ch in chain_features.values())
    cdr_hydro_sasa = sum(ch["CDR_Hydro_SASA"] for ch in chain_features.values())
    cdr_net_charge = sum(ch["CDR_Net_Charge"] for ch in chain_features.values())
    total_residues = sum(ch["Num_Residues"] for ch in chain_features.values())
    
    fab_features = {
        "Fab_Total_SASA": round(fab_total_sasa, 2),
        "Fab_Hydro_SASA": round(fab_hydro_sasa, 2),
        "Fab_Net_Charge": round(fab_net_charge, 1),
        "Fab_Max_Hydro_Patch": round(fab_max_hydro_patch, 2),
        "Fab_Max_Pos_Patch": fab_max_pos_patch,
        "Fab_Max_Neg_Patch": fab_max_neg_patch,
        "Fab_Charge_Dipole": round(fab_dipole, 2),
        "CDR_Hydro_SASA": round(cdr_hydro_sasa, 2),
        "CDR_Net_Charge": round(cdr_net_charge, 1),
        "ChA_Length": chain_features.get("A", {}).get("Num_Residues", 0),
        "ChB_Length": chain_features.get("B", {}).get("Num_Residues", 0),
        "Fab_Total_Residues": total_residues,
    }

    score, risk = calculate_viscosity_risk_score(fab_features)
    fab_features["Viscosity_Risk_Score"] = score
    fab_features["Viscosity_Risk_Category"] = risk
    
    conc_model = estimate_viscosity_vs_concentration(fab_features)
    fab_features.update(conc_model)

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
    cols = ['Therapeutic', 'CH1 Isotype', 'VD LC', 'Viscosity_Risk_Score', 'Viscosity_Risk_Category'] + [c for c in df.columns if c not in ['Therapeutic', 'CH1 Isotype', 'VD LC', 'Viscosity_Risk_Score', 'Viscosity_Risk_Category']]
    df = df[cols]

    # save final CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n=== SUCCESS: Viscosity risk profile extracted for {len(df)} mAbs ===")
    print(f"File saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_3D_viscosity_analysis()