from dataclasses import dataclass
from typing import List

seed: int
mode: str
logger: str
output_dir: str
verb: int
task_name: List[str]
project_wandb: str
name_wandb: str


@dataclass
class Paths:
    filename: str
    model_path: str

@dataclass
class Train_params:
    batch_size: int
    num_epochs: int
    optimizer_choice: str
    lr: float
    weight_decay: float
    factor: float
    patience: int
    minlr: float
    clip_value: float
    early_stopping: int
    train_ratio: float
    cv = bool
    nfold = int
    val_interval = int
    early_stop = int


@dataclass
class Model_params:
    node_feature: str
    edge_feature: str
    kekulize: bool
    with_hydrogen: bool
    gnn_choice: str
    hidden_size: int
    num_head: int
    num_layers_gnn: int
    num_mlp_layer: int
    mlp_dropout: float
    readout: str
    ssl_strategy: str
    mask_rate: float
    k: int
    r1: int
    r2: int
    num_negative: int

@dataclass
class Dataset:
    dataset_choice: str
    atom_feature: str
    bond_feature: str
    kekulize: bool
    with_hydrogen: bool
    atom_types: List[int]
    bond_types: List[int]


@dataclass
class inverse_design_home_config:
    train_params: Train_params
    mod_params: Model_params

