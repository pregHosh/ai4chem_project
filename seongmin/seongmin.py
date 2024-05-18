from rdkit import Chem
#import io
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from utils import plot_3d, plot_2d


df = pd.read_csv('../data/iflp_dataset_ed_idxs.csv')
#df.head()

#PandasTools.AddMoleculeColumnToFrame(df, smilesCol='smiles', molCol='Molecule')
#df.head()

#Check if mol are valid
def mol_valid(molecule):
    val = True
    flag = Chem.SanitizeMol(molecule)
    if flag == Chem.SanitizeFlags.SANITIZE_NONE:
        val = True
    else:
        val = False
    return val

df['sanitize_flag']=df['mol'].apply(mol_valid)


# Plot the FEHA and FEPA
