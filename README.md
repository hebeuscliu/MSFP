# MSFP

Code and trained models for:

**MSFP: Multi-Scale Feature Fusion for SND1 Inhibitor Discovery in Breast Cancer**

MSFP combines a pretrained MTSSMol graph encoder with ECFP4, MACCS, and six
physicochemical descriptors through a fusion MLP (with SE-Block), evaluated under a
scaffold-disjoint (leakage-free) split. The deployed ensemble reaches validation AUC 0.81
and, after screening and docking against SND1, yields enrichment factors up to 5.9-fold;
twelve docking hits are advanced to 100 ns MD and MM-GBSA.

## Repository contents

```
code/                 MSFP fusion model definition (Python)
  models/fusion_model.py   FusionModel (MTSSMol + ECFP4 + MACCS + 6 physchem + fusion MLP/SE-Block);
                           load trained checkpoints via build_ablation()

checkpoints/          10 trained fusion_full_rep*.pth -> deployed soft-voting ensemble

docking_md/           virtual screening -> docking -> MD -> MM-GBSA
  dock.py, prep_ligands.py, analyze_docking.py, select_md_candidates.py   (AutoDock Vina)
  setup_md_systems.py, md/*.sh, analyze_md.py                              (AMBER: tleap/antechamber/pmemd/cpptraj)
  prep_mmpbsa.sh, run_mmpbsa_full.sh, summarize_md_energy.py               (MMPBSA.py)
  receptor/  results/   receptor structure and result tables (ligands/data regeneratable)
```

## Data

The 178 experimentally confirmed positive SND1 inhibitors used for training (SMILES, also
in Supporting Information Table S5) are deposited on Zenodo:
<https://doi.org/10.5281/zenodo.21776509>.


## Dependencies

Python 3.10+, PyTorch, RDKit, scikit-learn, pandas, numpy; AutoDock Vina (docking);
AMBER (tleap, antechamber, pmemd, cpptraj, MMPBSA.py) for MD and MM-GBSA.

## License

Code: MIT. Trained checkpoints: CC-BY 4.0.
