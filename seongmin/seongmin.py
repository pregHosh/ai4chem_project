pip install rdkit
from rdkit import Chem
import pandas as pd
from rdkit.Chem import PandasTools

df = pd.read_csv("../data/iflp_smiles.csv")

PandasTools.AddMoleculeColumnToFrame(df, smilesCol='smiles', molCol='Molecule')
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

df.head()