from rdkit import Chem
#import io
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import seaborn as sns
import matplotlib.pyplot as plt




df = pd.read_csv('../data/iflp_dataset_ed_idxs.csv')
df['mol']=df['smiles'].apply(Chem.MolFromSmiles)


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


# Plot FEHA and FEPA
sns.scatterplot(data=df, x='FEPA', y='FEHA')
plt.title('FEPA vs FEHA')
plt.xlabel('FEPA')
plt.ylabel('FEHA')
plt.show()

# Add FEPA/FEHA coulumn

df['FEPA/FEHA']=df['FEPA']/df['FEHA']
print(df.head())

