# Import necessary libraries
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import io
import time

# Record the start time
start_time = time.time()

# Define a function that visualizes molecules from their SMILES string
def smiles_to_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol, size=(500, 500))
    return img

# Read dataset of FEHA-, and FEPA-values into a DataFrame
df_iflp_dataset_ed_idxs = pd.read_csv('../data/iflp_dataset_ed_idxs.csv')

# Make a nice human-readable version of the dataset
df_dataset_visualized = df_iflp_dataset_ed_idxs.drop(columns=['d','b_idx','n_idx']) # Drop unnecessary columns.
df_dataset_visualized = df_dataset_visualized.rename(columns={'smiles': 'molecules'}) # Rename SMILES-column to molecules.
df_dataset_visualized['molecules'] = df_dataset_visualized['molecules'].apply(smiles_to_image) # Use previously defined function to visualize molecules from their SMILES string.
for idx, img in enumerate(df_dataset_visualized['molecules']): 
      img.save(f'molecules_{idx}.png')

# Check, if that worked
#print(df_iflp_dataset_ed_idxs.head())
print(df_dataset_visualized.head())

# Record the end time
end_time = time.time()

# Calculate the duration
duration = end_time - start_time

# Display the duration
print(f"Execution time: {duration} seconds")