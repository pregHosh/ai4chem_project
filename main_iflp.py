import os
import pickle
import sys
from typing import  List, Optional, Union

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch import optim
from torchdrug import core, models, tasks
from iflp.dataset import IFLP_dataset_properties as dataset_properties
from config import inverse_design_home_config

cs = ConfigStore.instance()
cs.store(name="iflp_config", node=inverse_design_home_config)


class iflp_proppred:

    def __init__(
        self,
        filename: str,
        node_feature: str,
        edge_feature: str,
        with_hydrogen: bool,
        kekulize: bool,
        gnn_choice: str,
        num_epochs: int,
        batch_size: int,
        optimizer_choice: str,
        lr: float,
        weight_decay: float,
        val_interval: int,
        verb: int,
        output_dir: str,
        project_wandb: Union[str, None] = None,
        name_wandb: Union[str, None] = None,
        logger: str = "logging",
    ) -> None:

        self.filename = filename
        self.node_feature = node_feature
        self.edge_feature = edge_feature
        self.with_hydrogen = with_hydrogen
        self.kekulize = kekulize
        self.gnn_choice = gnn_choice
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer_choice = optimizer_choice
        self.lr = lr
        self.weight_decay = weight_decay

        self.val_interval = val_interval
        self.verb = verb
        self.output_dir = output_dir
        self.project_wandb = project_wandb
        self.name_wandb = name_wandb
        self.logger = logger if logger in ["wandb", "logging"] else "logging"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        if self.verb == 0:
            self.log_interval = 50
        elif self.verb == 1:
            self.log_interval = 10
        else:
            self.log_interval = 1

    def predictive_model(
        self,
        hidden_dims: list,
        num_head: int,
        readout: str,
        num_mlp_layer: int,
        mlp_dropout: int,
        factor: float,
        patience: int,
        min_lr: float,
        train_ratio: float = 0.8,
        task_name: Optional[List[str]] = None,
    ):
        dataset, model = self.get_dataset_model(
            with_hydrogen=self.with_hydrogen,
            kekulize=self.kekulize,
            input_dim_type="node_feat",
            edge_input_dim_type="edge_feat",
            hidden_dims=hidden_dims,
            readout=readout,
            num_head=num_head,
            mode="supervise",
        )
        params = {
            "architecture": self.gnn_choice,
            "node_feature_dim": dataset.node_feature_dim,
            "edge_feature_dim": dataset.edge_feature_dim,
            "kekulize": self.kekulize,
            "with_hydrogen": self.with_hydrogen,
            "num_bond_type": dataset.num_bond_type,
            "hidden_dims": hidden_dims,
            "num_head": num_head,
            "readout": readout,
            "num_mlp_layer": num_mlp_layer,
            "mlp_dropout": mlp_dropout,
            "node_feature": self.node_feature,
            "edge_feature": self.node_feature,
        }
        task = tasks.PropertyPrediction(
            model,
            task=task_name,
            criterion="mse",
            metric=("mae"),
            num_mlp_layer=num_mlp_layer,
            mlp_batch_norm=True,
            mlp_dropout=mlp_dropout,
            normalization=True,
        )

        optimizer = self.get_optimizer(task)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )
        save_hyp_dir = os.path.join(self.output_dir, "hyperparams.pkl")

        with open(save_hyp_dir, "wb") as f:
            pickle.dump(params, f)

        test_ratio = (1 - train_ratio) / 2
        lengths = [
            int(train_ratio * len(dataset)),
            int(test_ratio * len(dataset)),
        ]
        lengths += [len(dataset) - sum(lengths)]
        train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths)
        print(
            f"Total: {len(dataset)}, Train: {len(train_set)}, Valid: {len(valid_set)}, Test: {len(test_set)}"
        )
        solver = core.Engine(
            task,
            train_set,
            valid_set,
            test_set,
            optimizer,
            scheduler=scheduler,
            batch_size=self.batch_size,
            logger=self.logger,
            log_interval=self.log_interval,
            name_wandb=self.name_wandb,
            project_wandb=self.project_wandb,
            dir_wandb=self.output_dir,
        )

        best_val_loss = 20180322
        preds_test_best = None
        targets_test_best = None
        for i in range(self.num_epochs):
            solver.train(num_epoch=1)
            if train_ratio < 1.0:
                if i % self.val_interval == 0 or i == self.num_epochs - 1:
                    _, preds, targets = solver.evaluate("valid")
                    _, preds_test, targets_test = solver.evaluate("test")
                    preds = torch.cat(preds, dim=0)
                    targets = torch.cat(targets, dim=0)
                    mae_per_property = torch.mean(torch.abs(preds - targets), dim=0)
                    val_loss = torch.mean(mae_per_property)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        preds_test_best = preds_test
                        targets_test_best = targets_test

        y_preds = torch.cat(preds_test_best, dim=0)
        y_trues = torch.cat(targets_test_best, dim=0)
        np.save(
            os.path.join(self.output_dir, "y_preds.npy"),
            y_preds.detach().cpu().numpy(),
        )
        np.save(
            os.path.join(self.output_dir, "y_trues.npy"),
            y_trues.detach().cpu().numpy(),
        )
        return best_val_loss

    def fine_tune_predictive_model(
        self,
        model_path,
        readout: str,
        num_mlp_layer: int,
        mlp_dropout: int,
        factor: float,
        patience: int,
        min_lr: float,
        train_ratio: float = 0.8,
        task_name: Optional[List[str]] = None,
    ):
        hyperparam_path = os.path.join(model_path, "hyperparams.pkl")

        with open(hyperparam_path, "rb") as file:
            params = pickle.load(file)
        print(
            "------------Reading the model with the following hyperparameters-------------:"
        )
        for key, value in params.items():
            print(f"{key}: {value}")

        model_architecture_path = os.path.join(model_path, "chem_reprlearn.pkl")
        if torch.cuda.is_available():
            chk_point = torch.load(model_architecture_path)["model"]
        else:
            chk_point = torch.load(
                model_architecture_path, map_location=torch.device("cpu")
            )["model"]

        dataset = dataset_properties(
            self.filename,
            kekulize=params["kekulize"],
            atom_feature=params["node_feature"],
            bond_feature=params["edge_feature"],
            with_hydrogen=params["with_hydrogen"],
        )
        if params["architecture"] == "gin":
            model = models.GIN(
                input_dim=params["node_feature_dim"],
                hidden_dims=params["hidden_dims"],
                edge_input_dim=params["edge_feature_dim"],
                short_cut=False,
                batch_norm=True,
                readout=readout,
            )
        elif params["architecture"] == "rgcn":
            model = models.RGCN(
                input_dim=params["node_feature_dim"],
                num_relation=params["num_bond_type"],
                hidden_dims=params["hidden_dims"],
                batch_norm=True,
                short_cut=False,
                readout=readout,
            )
        elif params["architecture"] == "gan":
            model = models.GAT(
                input_dim=params["node_feature_dim"],
                edge_input_dim=params["edge_feature_dim"],
                hidden_dims=params["hidden_dims"],
                batch_norm=True,
                num_head=params["num_head"],
                readout=readout,
            )

        task = tasks.PropertyPrediction(
            model,
            task=task_name,
            criterion="mse",
            metric=("mae"),
            num_mlp_layer=num_mlp_layer,
            mlp_batch_norm=True,
            mlp_dropout=mlp_dropout,
            normalization=True,
        )
        params["idx_pos"] = None
        task.load_state_dict(chk_point, strict=False)
        save_hyp_dir = os.path.join(self.output_dir, "hyperparams.pkl")
        params["num_mlp_layer"] = num_mlp_layer
        params["mlp_dropout"] = mlp_dropout

        with open(save_hyp_dir, "wb") as f:
            pickle.dump(params, f)

        optimizer = self.get_optimizer(task)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )
        test_ratio = (1 - train_ratio) / 2
        lengths = [
            int(train_ratio * len(dataset)),
            int(test_ratio * len(dataset)),
        ]
        lengths += [len(dataset) - sum(lengths)]
        train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths)
        print(
            f"Total: {len(dataset)}, Train: {len(train_set)}, Valid: {len(valid_set)}, Test: {len(test_set)}"
        )
        solver = core.Engine(
            task,
            train_set,
            valid_set,
            test_set,
            optimizer,
            scheduler=scheduler,
            batch_size=self.batch_size,
            logger=self.logger,
            log_interval=self.log_interval,
            name_wandb=self.name_wandb,
            project_wandb=self.project_wandb,
            dir_wandb=self.output_dir,
        )

        best_val_loss = 20180322
        preds_test_best = None
        targets_test_best = None
        for i in range(self.num_epochs):
            solver.train(num_epoch=1)
            if train_ratio < 1.0:
                if i % self.val_interval == 0 or i == self.num_epochs - 1:
                    _, preds, targets = solver.evaluate("valid")
                    _, preds_test, targets_test = solver.evaluate("test")
                    preds = torch.cat(preds, dim=0)
                    targets = torch.cat(targets, dim=0)
                    mae_per_property = torch.mean(torch.abs(preds - targets), dim=0)
                    val_loss = torch.mean(mae_per_property)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        preds_test_best = preds_test
                        targets_test_best = targets_test

        y_preds = torch.cat(preds_test_best, dim=0)
        y_trues = torch.cat(targets_test_best, dim=0)
        np.save(
            os.path.join(self.output_dir, "y_preds.npy"),
            y_preds.detach().cpu().numpy(),
        )
        np.save(
            os.path.join(self.output_dir, "y_trues.npy"),
            y_trues.detach().cpu().numpy(),
        )
        return best_val_loss
    
    def get_dataset_model(
        self,
        with_hydrogen: bool,
        kekulize: bool,
        input_dim_type: str,
        edge_input_dim_type: Optional[str],
        hidden_dims: List[int],
        readout: str,
        num_head,

    ):
        """
        Get the dataset and model

        Args:
            with_hydrogen (bool): Whether to include hydrogen atoms in the dataset.
            kekulize (bool): Whether to kekulize the molecule structures.
            input_dim_type (str): symbol or node_feat
            edge_input_dim_type (str): edge_feat or None
            hidden_dims (List[int]): List of hidden dimensions for the GNN model.
            readout (str): Readout method for the GNN model.
            num_head (int): Number of attention heads for the GAT model.
            mode (str): Mode of operation, either "unsupervise" or "supervise".
            mix (str): large scale dataset to mix with.
            Available to just unsupervise mode as only smiles are considered.
            Just ZINC250k + FORM for now


        Returns:
            tuple: A tuple containing the dataset and model objects.
        """

        dataset = dataset_properties(
            self.filename,
            kekulize=kekulize,
            atom_feature=self.node_feature,
            bond_feature=self.edge_feature,
            with_hydrogen=with_hydrogen,
        )
        if input_dim_type == "symbol":
            input_dim = dataset.num_atom_type
        elif input_dim_type == "node_feat":
            input_dim = dataset.node_feature_dim

        if edge_input_dim_type == "edge_feat":
            edge_input_dim = dataset.edge_feature_dim
        else:
            edge_input_dim = None
        if self.gnn_choice == "rgcn":
            model = models.RGCN(
                input_dim=input_dim,
                edge_input_dim=edge_input_dim,
                num_relation=dataset.num_bond_type,
                hidden_dims=hidden_dims,
                batch_norm=True,
                short_cut=False,
                readout=readout,
            )
        elif self.gnn_choice == "gin":
            model = models.GIN(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                edge_input_dim=dataset.edge_feature_dim,
                batch_norm=True,
                short_cut=False,
                readout=readout,
            )
        elif self.gnn_choice == "gan":
            model = models.GAT(
                input_dim=input_dim,
                edge_input_dim=dataset.edge_feature_dim,
                hidden_dims=hidden_dims,
                batch_norm=True,
                num_head=num_head,
                readout=readout,
            )
        else:
            raise ValueError("Invalid GNN choice (gin, rgcn, or gan)")
        return dataset, model

    def get_optimizer(self, task, foreach=False):

        if self.optimizer_choice == "adam":
            optimizer = optim.Adam(
                task.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                foreach=foreach,
            )
        elif self.optimizer_choice == "amsgrad":
            optimizer = optim.Adam(
                task.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                amsgrad=True,
                foreach=foreach,
            )
        elif self.optimizer_choice == "adamw":
            optimizer = optim.AdamW(
                task.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                foreach=foreach,
            )
        elif self.optimizer_choice == "radam":
            optimizer = optim.RAdam(
                task.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                foreach=foreach,
            )
        else:
            raise ValueError(
                f"Invalid optimizer {self.optimizer_choice} (allowed: adam, adamw, radam)"
            )
        return optimizer


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: iflp_proppred):

    if cfg.seed:
        import random

        print(f"Setting seed to {cfg.seed}")
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)  # For multi-GPU.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

    if cfg.output_dir == "default":
        output_dir = cfg.mode
        print("Setting output_dir to", output_dir)
    else:
        output_dir = cfg.output_dir
    print("Training run with...\n")
    print(OmegaConf.to_yaml(cfg))
    inverse_platform = iflp_proppred(
        filename=cfg.paths.filename,
        node_feature=cfg.mod_params.node_feature,
        edge_feature=cfg.mod_params.edge_feature,
        with_hydrogen=cfg.mod_params.with_hydrogen,
        kekulize=cfg.mod_params.kekulize,
        gnn_choice=cfg.mod_params.gnn_choice,
        num_epochs=cfg.train_params.num_epochs,
        batch_size=cfg.train_params.batch_size,
        optimizer_choice=cfg.train_params.optimizer_choice,
        lr=cfg.train_params.lr,
        weight_decay=cfg.train_params.weight_decay,
        val_interval=cfg.train_params.val_interval,
        verb=cfg.verb,
        output_dir=output_dir,
        project_wandb=cfg.project_wandb,
        name_wandb=cfg.name_wandb,
        logger=cfg.logger,
    )
    hidden_dims = [cfg.mod_params.hidden_size] * cfg.mod_params.num_layers_gnn
    
    if cfg.mode == "predictive":
        inverse_platform.predictive_model(
            hidden_dims=hidden_dims,
            num_head=cfg.mod_params.num_head,
            readout=cfg.mod_params.readout,
            num_mlp_layer=cfg.mod_params.num_mlp_layer,
            mlp_dropout=cfg.mod_params.mlp_dropout,
            factor=cfg.train_params.factor,
            patience=cfg.train_params.patience,
            min_lr=cfg.train_params.minlr,
            train_ratio=cfg.train_params.train_ratio,
            task_name=cfg.task_name,
        )

    elif cfg.mode == "ft_predictive":
        if not (os.path.exists(cfg.paths.model_path)):
            sys.exit("Error: Model path does not exist.")
        else:
            print(
                f"\n====Fine-tuning the from {cfg.paths.model_path} to predict the properties====\n"
            )
        inverse_platform.fine_tune_predictive_model(
            model_path=cfg.paths.model_path,
            readout=cfg.mod_params.readout,
            num_mlp_layer=cfg.mod_params.num_mlp_layer,
            mlp_dropout=cfg.mod_params.mlp_dropout,
            factor=cfg.train_params.factor,
            patience=cfg.train_params.patience,
            min_lr=cfg.train_params.minlr,
            train_ratio=cfg.train_params.train_ratio,
            task_name=cfg.task_name,
        )
        
if __name__ == "__main__":
    main()
