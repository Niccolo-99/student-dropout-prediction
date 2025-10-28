# Student Dropout Prediction con UniTS

This repository contains the code used in the study "Leveraging Time-Series Modeling for Enhanced Student Dropout Prediction: A Comparative Study". The pipeline combines the UniTS model with online learning datasets to classify at-risk students and to reuse UniTS public checkpoints in prompt-tuning and supervised fine-tuning scenarios.

## Project structure

- `run.py`: entry point exposing all CLI/Debug options for UniTS training, prompt tuning, and testing via PyTorch and Weights & Biases.
- `exp/exp_sup.py`: implements the multi-task training loop (loader balancing, metric computation, DDP handling).
- `data_provider/`: YAML configurations listing datasets and tasks; only_dropout_datasets_task.yaml maps Unitelma, KDDCup15, and XuetangX to their respective processed folders.
- `download_data_all.sh`: script to automatically download public archives (TimesNet/UCR/UAE and several UEA datasets) required by general multi-task experiments.
- `Unitelma_dataset_builder.py`, `KDDcup15_dataset_builder.py`, `XuetangX_dataset_builder.py`: scripts to convert raw datasets into UEA multivariate format compatible with UniTS.
- `checkpoints/`: directory where you place pre-trained weights (e.g., `units_x64_supervised_checkpoint.pth`).

## Requirements

1. Python environment: Python 3.9+ with PyTorch (CUDA >= 11.7 recommended) and an NVIDIA GPU to run UniTS efficiently.
2. Dependencies: install the packages listed in requirements.txt (plus PyTorch/torchvision/torchaudio compatible with your GPU).

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
# install PyTorch separately based on your GPU: https://pytorch.org/get-started/locally/
```

If you use Weights & Biases, log in (wandb login) before starting experiments.

## Data preparation

### 1. UniTS public datasets (TimesNet/UEA/UCR)

Run the included script that downloads all secondary archives required by the multi-task configurations:

```bash
bash download_data_all.sh
```

The script creates the `dataset/` folder and fills subfolders such as `dataset/electricity`, `dataset/UAE`, `dataset/UCR`, etc.

### 2. Dropout datasets

Download the raw packages from the official links:

| Dataset | Link sorgente |
| ------- | ------------- |
| XuetangX | http://moocdata.cn/data/user-activity |
| KDDCup 2015 | http://moocdata.cn/data/user-activity |
| Unitelma Sapienza | https://figshare.com/articles/dataset/UnitelmaSapienza_1_0_zip/14554137?file=27923373 |

Organizzare gli archivi come segue all'interno di `dataset/`:

```
dataset/
├── XuetangX/XuetangX_raw/...
├── KDDcup15/KDDCup15_raw/...
└── Unitelma/Unitelma_raw/...
```

Tip: keep the original CSVs in the _raw folders so you can rebuild the preprocessed versions at any time.

### 3. Conversion to UEA format

The included Python scripts generate `.ts` files with UEA headers starting from the raw CSVs. 

Each script reads all required CSVs, builds day-by-day trajectories, and writes `*_TRAIN.ts` and `*_TEST.ts` files to the output folder along with JSON metadata.
The YAML configurations point exactly to these destinations (`dataset/Unitelma/unitelma_processed`, `dataset\KDDcup15\kddcup15_processed`, `dataset\XuetangX\XuetangX_processed`).

### 4. UniTS checkpoints

Download original paper, code and weights from:
- https://github.com/mims-harvard/UniTS
- https://github.com/mims-harvard/UniTS/releases/tag/ckpt

And place them in `checkpoints/` with the same names used in the Code configurations (e.g., `units_x64_supervised_checkpoint.pth`, `units_x128_pretrain_checkpoint.pth`).

## Running experiments

### Command-line usage

`run.py` lets you launch training, prompt tuning, and testing by toggling flags; for example:

```bash
# Supervised fine-tuning on the dropout-only datasets (prompts disabled)
python run.py \
  --is_training 1 \
  --model_id Finetune_sup_dropout_d64_5ep \
  --model UniTS \
  --task_data_config_path data_provider/only_dropout_datasets_task.yaml \
  --pretrained_weight checkpoints/units_x64_supervised_checkpoint.pth \
  --train_epochs 5 \
  --learning_rate 5e-5 \
  --batch_size 32 \
  --acc_it 1 \
  --clip_grad 100 \
  --d_model 64 \
  --e_layers 3 \
  --patch_len 16 --stride 16 --prompt_num 10 \
  --project_name tsfm-multitask --debug disabled
```

The parameters replicate one of the ready-made configurations in `launch.json`; you can adapt `--task_data_config_path` to switch from general multi-task scenarios to dropout-specific scenarios.

For evaluation only (no training), set `--is_training 0` and provide the checkpoint via `--pretrained_weight`. The code automatically initializes distributed mode (even on a single GPU) and starts the WandB logger only on the main process.

### Debugging with VS Code (`launch.json`)

The `launch.json` file offers preconfigured profiles for the most common scenarios:

- **BaseHP_supervised_all_datasets_d64**: supervised training on all general multi-task datasets.
- **BaseHP_supervised_only_dropout_d64**: supervised training on the dropout-only datasets.
- **Ptune_pretrain_* / Ptune_supervised_***: prompt tuning starting from x128 pretrain or x64 supervised checkpoints.
- **Finetune_sup_*_5ep**: supervised fine-tuning for a few epochs using x64 weights.
- **Test_sup_d64**: evaluation of a supervised checkpoint.

Select a configuration in the Run & Debug panel of VS Code and start in Debug mode; the environment sets `CUDA_VISIBLE_DEVICES=0` and, for prompt-tuning profiles, enables `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` to ensure full checkpoint loading.

## Task configurations

- `data_provider/only_dropout_datasets_task.yaml`: uses only the Unitelma, KDDCup15, and XuetangX datasets converted to UEA format.
- `data_provider/multi_task.yaml`: includes forecasting, classification, and other UniTS benchmark tasks (requires all datasets downloaded with `download_data_all.sh`).

You can duplicate a YAML and edit the paths to introduce new datasets or change sequence lengths.
