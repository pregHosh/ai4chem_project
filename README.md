# Building GNN prediction model for chemical descriptors to facailitate IFLP design


The trained models can be found in "model" folder.

## Installation

```
conda env create -f ai4chem.yml
```
## Usages 

To view all configurations, run the following command:
```bash
python main_iflp.py --help
```

### Train the GNN model from scratch
```bash
python main_iflp.py mode=predictive
```

### Fine-tune from a pre-trained GNN model
```bash
python main_iflp.py mode=ft_predictive paths.model_path=[pretrained_model_path]
```

### Predict the chemical descriptors
```bash
python eval_iflp.py --p [pretrained_model_path] --i [input_file_path] --t FEPA FEHA
```

### Analyze the trained model
```bash
python utilis/model_analysis.py --d [pretrained_model_path] --t FEPA FEHA
```


