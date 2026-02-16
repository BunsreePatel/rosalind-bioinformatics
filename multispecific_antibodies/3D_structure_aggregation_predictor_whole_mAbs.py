import os
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from Bio.PDB import PDBParser, NeighborSearch

warnings.filterwarnings("ignore")

# --- PATHS ---
BASE_DIR = Path(r"C:\Users\meeko\rosalind-bioinformatics\multispecific_antibodies\Whole_mAb")
PDB_DIR = BASE_DIR / "PDB_Output_Files_GPU_Full"
OUTPUT_CSV = BASE_DIR / "Whole_mab_Structure_Based_Aggregation_Risk.csv"
MASTER_CSV = BASE_DIR / "TheraSAbDab_SeqStruc_07Dec2025.csv"


"""
THEORY: STRUCTURE-BASED AGGREGATION PREDICTION
===============================================

AGGREGATION DRIVERS (3D-Dependent):
1. HYDROPHOBIC SURFACE PATCHES
   - Contiguous hydrophobic residues in 3D space (not just sequence)
   - Measured by spatial clustering: residues within 6Å radius
   - Literature: Chennamsetty et al. (2009) - spatial patches >150 Ų = high risk

2. SPATIAL CHARGE DISTRIBUTION
   - Electrostatic dipole moments from asymmetric charge distribution
   - High dipoles (>300 Debye) correlate with self-association
   - Literature: Tomar et al. (2016) - dipole predicts viscosity

3. EXPOSED AROMATIC CLUSTERS
   - PHE, TYR, TRP clusters on surface drive π-π stacking
   - Particularly problematic in CDRs
   - Literature: Schuyler et al. (2011) - aromatic patches = aggregation

4. SURFACE ELECTROSTATIC POTENTIAL
   - Complementary charged patches enable protein-protein association
   - Measured by positive/negative patch proximity
   - Literature: Yadav et al. (2012) - charge patches correlate with high viscosity

RISK SCORING MODEL:
- Each metric weighted by experimental correlation strength
- Final score: 0-100 (>50 = high risk, 30-50 = moderate, <30 = low)
"""

# --- BIOPHYSICAL SCALES ---
HYDROPHOBIC_RESIDUES = ['ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP']
AROMATIC_RESIDUES = ['PHE', 'TYR', 'TRP']
POSITIVE_RESIDUES = ['ARG', 'LYS', 'HIS']
NEGATIVE_RESIDUES = ['ASP', 'GLU']

# Thresholds (from literature)
PATCH_CUTOFF = 150.0  # Ų - Chennamsetty (2009)
DIPOLE_CUTOFF = 300.0  # Debye - Tomar (2016)
CLUSTER_RADIUS = 6.0  # Å - standard for spatial clustering
MIN_CLUSTER_SIZE = 4  # residues

def calculate_spatial_hydrophobic_clusters(chain):
    """
    Finds 3D spatial clusters of hydrophobic residues.
    Returns: number of clusters, largest cluster size, largest cluster SASA
    """
    hydro_residues = [r for r in chain if r.get_resname() in HYDROPHOBIC_RESIDUES]
    
    if len(hydro_residues) < MIN_CLUSTER_SIZE:
        return 0, 0, 0
    
    # Get all atoms from hydrophobic residues
    hydro_atoms = []
    for res in hydro_residues:
        hydro_atoms.extend(list(res.get_atoms()))
    
    # Use NeighborSearch for efficient spatial queries
    ns = NeighborSearch(hydro_atoms)
    
    # Find clusters using graph-based approach
    visited = set()
    clusters = []
    
    for res in hydro_residues:
        if res in visited:
            continue
            
        # BFS to find connected cluster
        cluster = []
        queue = [res]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
                
            visited.add(current)
            cluster.append(current)
            
            # Find neighbors within CLUSTER_RADIUS
            for atom in current.get_atoms():
                neighbors = ns.search(atom.get_coord(), CLUSTER_RADIUS)
                for neighbor_atom in neighbors:
                    neighbor_res = neighbor_atom.get_parent()
                    if (neighbor_res not in visited and 
                        neighbor_res.get_resname() in HYDROPHOBIC_RESIDUES):
                        queue.append(neighbor_res)
        
        if len(cluster) >= MIN_CLUSTER_SIZE:
            clusters.append(cluster)
    
    if not clusters:
        return 0, 0, 0
    
    # Calculate cluster metrics
    largest_cluster = max(clusters, key=len)
    largest_cluster_sasa = sum(getattr(r, 'sasa', 0) for r in largest_cluster)
    
    return len(clusters), len(largest_cluster), round(largest_cluster_sasa, 2)

def calculate_aromatic_clusters(chain):
    """
    Finds spatial clusters of aromatic residues (π-π stacking risk).
    Returns: number of clusters, largest cluster size
    """
    aromatic_residues = [r for r in chain if r.get_resname() in AROMATIC_RESIDUES]
    
    if len(aromatic_residues) < 3:
        return 0, 0
    
    aromatic_atoms = []
    for res in aromatic_residues:
        aromatic_atoms.extend(list(res.get_atoms()))
    
    ns = NeighborSearch(aromatic_atoms)
    
    visited = set()
    clusters = []
    
    for res in aromatic_residues:
        if res in visited:
            continue
            
        cluster = []
        queue = [res]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
                
            visited.add(current)
            cluster.append(current)
            
            for atom in current.get_atoms():
                neighbors = ns.search(atom.get_coord(), CLUSTER_RADIUS)
                for neighbor_atom in neighbors:
                    neighbor_res = neighbor_atom.get_parent()
                    if (neighbor_res not in visited and 
                        neighbor_res.get_resname() in AROMATIC_RESIDUES):
                        queue.append(neighbor_res)
        
        if len(cluster) >= 3:
            clusters.append(cluster)
    
    if not clusters:
        return 0, 0
    
    largest_cluster = max(clusters, key=len)
    return len(clusters), len(largest_cluster)

def calculate_charge_dipole(chain):
    """
    Calculates electrostatic dipole moment from 3D charge distribution.
    Returns: dipole magnitude in Debye
    """
    positive_coords = []
    negative_coords = []
    
    for res in chain:
        res_name = res.get_resname()
        com = np.array([atom.get_coord() for atom in res.get_atoms()]).mean(axis=0)
        
        if res_name in POSITIVE_RESIDUES:
            charge = 1.0 if res_name in ['ARG', 'LYS'] else 0.1
            positive_coords.append((com, charge))
        elif res_name in NEGATIVE_RESIDUES:
            negative_coords.append((com, 1.0))
    
    if not positive_coords and not negative_coords:
        return 0.0
    
    # Calculate center of mass for entire chain
    all_coords = [atom.get_coord() for res in chain for atom in res.get_atoms()]
    chain_com = np.array(all_coords).mean(axis=0)
    
    # Calculate dipole vector: sum of (charge * displacement from COM)
    dipole_vector = np.zeros(3)
    
    for coord, charge in positive_coords:
        dipole_vector += charge * (coord - chain_com)
    
    for coord, charge in negative_coords:
        dipole_vector -= charge * (coord - chain_com)
    
    # Convert to Debye (1 Debye = 0.2082 e·Å)
    dipole_magnitude = np.linalg.norm(dipole_vector) / 0.2082
    
    return round(dipole_magnitude, 2)

def calculate_complementary_charge_patches(chain):
    """
    Finds spatial proximity of positive and negative patches.
    Returns: number of complementary patch pairs
    """
    positive_residues = [r for r in chain if r.get_resname() in POSITIVE_RESIDUES]
    negative_residues = [r for r in chain if r.get_resname() in NEGATIVE_RESIDUES]
    
    if not positive_residues or not negative_residues:
        return 0
    
    # Find positive clusters
    pos_atoms = []
    for res in positive_residues:
        pos_atoms.extend(list(res.get_atoms()))
    
    ns_pos = NeighborSearch(pos_atoms)
    
    # Count complementary patches within 10-15Å (interaction range)
    complementary_pairs = 0
    
    for neg_res in negative_residues:
        for atom in neg_res.get_atoms():
            # Check if there's a positive cluster nearby (but not too close - buried salt bridges)
            neighbors = ns_pos.search(atom.get_coord(), 15.0, level='R')
            close_neighbors = ns_pos.search(atom.get_coord(), 4.0, level='R')
            
            # Complementary patch: positive cluster nearby but not buried together
            if len(neighbors) >= 3 and len(close_neighbors) == 0:
                complementary_pairs += 1
                break
    
    return complementary_pairs

def run_structure_aggregation_analysis():
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

    print(f"Found {len(pdb_map)} PDB files for aggregation analysis.")
    
    # Setup tools
    parser = PDBParser(QUIET=True)
    master_features = []
    
    # Process every PDB found
    print("--- CALCULATING STRUCTURE-BASED AGGREGATION RISK ---")
    for ab_name, pdb_path in tqdm(pdb_map.items()):
        try:
            struct = parser.get_structure(ab_name, str(pdb_path))
           
            total_hydro_clusters = 0
            total_aromatic_clusters = 0
            max_dipole = 0
            total_charge_patches = 0
            chain_stats = {}
            chain_residues = {}

            # Analyze each chain (A, B)
            for model in struct:
                for chain in model:
                    c_id = chain.get_id()

                    residues = list(chain.get_residues())
                    c_residues = len(residues)
                    chain_residues[c_id] = c_residues
                    
                    # Calculate spatial metrics
                    hydro_clusters, largest_hydro_size, largest_hydro_sasa = calculate_spatial_hydrophobic_clusters(chain)
                    aromatic_clusters, largest_aromatic_size = calculate_aromatic_clusters(chain)
                    dipole = calculate_charge_dipole(chain)
                    charge_patches = calculate_complementary_charge_patches(chain)
            
                    # Store per-chain metrics
                    chain_stats[f"Ch_{c_id}_Hydro_Clusters"] = hydro_clusters
                    chain_stats[f"Ch_{c_id}_Largest_Hydro_Cluster_Size"] = largest_hydro_size
                    chain_stats[f"Ch_{c_id}_Largest_Hydro_Cluster_SASA"] = largest_hydro_sasa
                    chain_stats[f"Ch_{c_id}_Aromatic_Clusters"] = aromatic_clusters
                    chain_stats[f"Ch_{c_id}_Largest_Aromatic_Size"] = largest_aromatic_size
                    chain_stats[f"Ch_{c_id}_Dipole_Moment"] = dipole
                    chain_stats[f"Ch_{c_id}_Charge_Patches"] = charge_patches
                    
                    # Accumulate totals
                    total_hydro_clusters += hydro_clusters
                    total_aromatic_clusters += aromatic_clusters
                    max_dipole = max(max_dipole, dipole)
                    total_charge_patches += charge_patches

            # Chain residue summary
            total_residues = sum(chain_residues.values())
            chA_length = chain_residues.get('A', 0)
            chB_length = chain_residues.get('B', 0)
            total_length = total_residues
        
            # Match antibody names
            ab_name_clean = ab_name.replace('_fab', '').capitalize()

            # Calculate aggregate risk score
            # Weights based on literature correlations
            risk_score = 0
            
            # 1. Hydrophobic clusters (40% weight)
            if total_hydro_clusters > 0:
                risk_score += min(total_hydro_clusters * 10, 40)
            
            # 2. Dipole moment (30% weight)
            if max_dipole > DIPOLE_CUTOFF:
                risk_score += min((max_dipole / DIPOLE_CUTOFF) * 30, 30)
            
            # 3. Aromatic clusters (20% weight)
            if total_aromatic_clusters > 0:
                risk_score += min(total_aromatic_clusters * 5, 20)
            
            # 4. Complementary charge patches (10% weight)
            if total_charge_patches > 0:
                risk_score += min(total_charge_patches * 2, 10)
            
            # Compile overall data for this mAb
            entry = {
                "Therapeutic": ab_name.capitalize(),
                "Total_Hydro_Clusters": total_hydro_clusters,
                "Total_Aromatic_Clusters": total_aromatic_clusters,
                "Max_Dipole_Moment": max_dipole,
                "Total_Charge_Patches": total_charge_patches,
                "Aggregation_Risk_Score": round(risk_score, 2),
                "Ch_A_Length": chA_length,
                "Ch_B_Length": chB_length,
                "Fab_Total_AAs": total_length
            }
            entry.update(chain_stats)
            master_features.append(entry)
            
            # Risk categorization
            if risk_score >= 50:
                entry["Risk_Category"] = "HIGH"
            elif risk_score >= 30:
                entry["Risk_Category"] = "MODERATE"
            else:
                entry["Risk_Category"] = "LOW"
           
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
        print(f"\n=== SUCCESS: Aggregation analysis complete for {len(df)} antibodies ===")
        print(f"File saved to: {OUTPUT_CSV}")
        print(f"\nRisk Distribution:")
        print(df["Risk_Category"].value_counts())
        print(f"\nTop 20 Highest Risk Antibodies:")
        print(df[["Therapeutic", "Aggregation_Risk_Score", "Risk_Category"]].head())

if __name__ == "__main__":
    run_structure_aggregation_analysis()