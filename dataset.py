from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.IFLP_dataset")
@utils.copy_args(
    data.MoleculeDataset.load_csv, ignore=("smiles_field", "target_fields")
)
class IFLP_dataset_smiles(data.MoleculeDataset):
    """
    A dataset class for IFLP molecules based on SMILES representations.

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    def __init__(self, path, verbose=1, **kwargs):
        self.load_csv(
            path, smiles_field="smiles", verbose=verbose, target_fields=None, **kwargs
        )


@R.register("datasets.IFLP_dataset_properties")
@utils.copy_args(
    data.MoleculeDataset.load_csv, ignore=("smiles_field", "target_fields")
)
class IFLP_dataset_properties(data.MoleculeDataset):
    """
    A dataset class for IFLP molecules with geometrical and chemical properties.

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    def __init__(self, path, verbose=1, **kwargs):
        target_fields = ["d", "phi", "FEPA", "FEHA"]
        self.load_csv(
            path,
            smiles_field="smiles",
            verbose=verbose,
            target_fields=target_fields,
            **kwargs
        )


if __name__ == "__main__":
    dataset = IFLP_dataset_smiles(
        "iflp_4test.csv",
        kekulize=True,
        atom_feature="default",
        bond_feature="default",
    )
    dataset = IFLP_dataset_properties(
        "iflp_dataset_full.csv",
        kekulize=True,
        atom_feature="default",
        bond_feature="default",
    )
