import argparse
import os
import pickle

import numpy as np
import pandas as pd
from torchdrug import data, models, tasks
from tqdm import tqdm
from rdkit import Chem
from navicatGA.timeout import timeout
import torch

def make_predictive_task(
    path: str,
    task_names: list,
    eval=True,
    mutual_info_weight: float = 0.0,
    frozen_core: bool = False,
):
    print("\nReading hyperparameters for predictive model...\n")
    hyperparam_predictive_model_path = os.path.join(path, "hyperparams.pkl")
    with open(hyperparam_predictive_model_path, "rb") as file:
        params = pickle.load(file)
    for key, value in params.items():
        print(f"{key}: {value}")
    if params["architecture"] == "rgcn":
        model = models.RGCN(
            input_dim=params["node_feature_dim"],
            num_relation=params["num_bond_type"],
            hidden_dims=params["hidden_dims"],
            short_cut=False,
            batch_norm=True,
            readout=params["readout"],
        )
    elif params["architecture"] == "gin":
        model = models.GIN(
            input_dim=params["node_feature_dim"],
            hidden_dims=params["hidden_dims"],
            edge_input_dim=params["edge_feature_dim"],
            short_cut=False,
            batch_norm=True,
            readout=params["readout"],
        )
    elif params["architecture"] == "gat" or params["architecture"] == "gan":
        model = models.GAT(
            input_dim=params["node_feature_dim"],
            edge_input_dim=params["edge_feature_dim"],
            hidden_dims=params["hidden_dims"],
            batch_norm=True,
            num_head=params["num_head"],
            readout=params["readout"],
        )
    else:
        raise ValueError(
            f"Invalid architecture {params['architecture']} (allowed: rgcn, gin, gat, gan)"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(f"{path}/chem_predictive.pkl", map_location=device)["model"]
    local = False
    if eval:
        std_mean = [checkpoint["std"], checkpoint["mean"]]
    else:
        std_mean = None

    if mutual_info_weight == 0:
        print("------Use a normal fine-tuning approach------")
        if params["idx_pos"] is None:
            task = tasks.PropertyPrediction(
                model,
                task=task_names,
                criterion="mse",
                metric=("mae", "mae"),
                num_class=len(task_names),
                mlp_batch_norm=True,
                normalization=True,
                mlp_dropout=0,
                std_mean=std_mean,
                num_mlp_layer=params["num_mlp_layer"],
            )
        else:
            print("Using local prediction")
            task = tasks.PropertyPrediction_local(
                model,
                task=task_names,
                criterion="mse",
                metric=("mae", "mae"),
                num_mlp_layer=params["num_mlp_layer"],
                num_class=len(task_names),
                mlp_batch_norm=True,
                normalization=True,
                mlp_dropout=0,
                std_mean=std_mean,
                idx_pos=params["idx_pos"],
            )
            local = True

        task.load_state_dict(checkpoint, strict=False)
    else:
        print("------Use teacher-student model------")
        task = tasks.ProperyPrediction_KnowledgeDistillation(
            model,
            task=task_names,
            criterion="mse",
            metric=("mae"),
            mlp_batch_norm=True,
            normalization=True,
            mlp_dropout=0,
            num_mlp_layer=params["num_mlp_layer"],
            mutual_info_weight=mutual_info_weight,
            activation="relu",
        )
        task.load_model(checkpoint)

    if frozen_core:
        print("------Use fixed feature extractor approach------")
        for param in task.model.parameters():
            param.requires_grad = False

    return task, params, local


def prediction(smiles, task, params, crds_locator, local=False):

    if local:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        crd_all = crds_locator(mol)
        if -1 in crd_all:
            raise ValueError("Cannot identify the node indices")

        graphs = {
            "graph": data.Molecule.from_smiles(
                smiles,
                kekulize=params["kekulize"],
                atom_feature=params["node_feature"],
                bond_feature=params["edge_feature"],
                with_hydrogen=params["with_hydrogen"],
            ),
        }
        for i, value in enumerate(crd_all):
            graphs[f"idx_{i}"] = value
    else:
        graphs = {
            "graph": data.Molecule.from_smiles(
                smiles,
                kekulize=params["kekulize"],
                atom_feature=params["node_feature"],
                bond_feature=params["edge_feature"],
                with_hydrogen=params["with_hydrogen"],
            )
        }

    preds = task.predict(graphs, evaluate=True).detach().numpy()[0]

    return preds



def main():

    parser = argparse.ArgumentParser(
        description="""Properties (FMO) prediction with trained model.
        """
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        type=str,
        default="pretrain_gm",
        help="Input path for the CSV containing smiles, gap, HOMO, and LUMO ",
    )
    parser.add_argument(
        "-p",
        "--predictive",
        dest="predictive",
        type=str,
        default="predictive",
        help="Input path for the predictive model",
    )
    parser.add_argument(
        "-t",
        "--task_names",
        dest="task_names",
        nargs="+",
        default=["gap", "HOMO", "LUMO"],
        help="Specify the task names [must be in the same order as in the predictive model]",
    )

    args = parser.parse_args()
    input_path = args.input
    predictive_model_path = args.predictive
    task_names = args.task_names

    df_test = pd.read_csv(input_path)
    prop_dft = np.array([df_test[task].to_numpy() for task in task_names]).transpose()
    smiles_data = pd.read_csv(input_path)
    task_pred, params_pred, local = make_predictive_task(
        predictive_model_path, task_names
    )
    if "smiles" not in smiles_data.columns:
        print("No 'smiles' column found in the CSV file.")
        return
    smiles_generated = smiles_data["smiles"].values
    print(f"Got {len(smiles_generated)} SMILES.")

    if len(smiles_generated) == 0:
        print("No valid SMILES generated.")
        return

    props = []
    fail_eval_smiles = []
    pass_eval_smiles = []
    for smiles in tqdm(smiles_generated, total=len(smiles_generated)):
        try:
            with timeout(seconds=86):
                prop_preds = prediction(smiles, task_pred, params_pred, None, local)
                props.append(prop_preds)
                pass_eval_smiles.append(smiles)
        except Exception as e:
            print(f"Fail to pred for {smiles} due to {e}")
            fail_eval_smiles.append(smiles)
        
    preds = np.array(props)
    for i, task_name in enumerate(task_names):
        print(f"{task_name}: {preds[:, i]}\n")


if __name__ == "__main__":
    main()
