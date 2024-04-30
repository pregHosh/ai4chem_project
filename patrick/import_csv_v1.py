# Import of necessary libraries
import pandas as pd

# Read dataset of FEHA-, and FEPA-values into a DataFrame
df = pd.read_csv('../data/iflp_dataset_ed_idxs.csv')

# Check, if that worked
print(df.head())