# RuTaBERT
Model for solving the problem of Column Type Annotation with BERT, trained on [russian corpus](https://github.com/STI-Team/RuTaBERT-Dataset).

## Project structure
```
📦RuTaBERT
 ┣ 📂checkpoints
 ┃ ┗ Saved PyTorch models `.pt` 
 ┣ 📂data
 ┃ ┣ 📂inference
 ┃ ┃ ┗ Tabels to inference `.csv`
 ┃ ┣ 📂test
 ┃ ┃ ┗ Test dataset files `.csv`
 ┃ ┣ 📂train
 ┃ ┃ ┗ Train dataset files `.csv`
 ┃ ┗  Directory for storing dataset files.
 ┣ 📂dataset
 ┃ ┗  Dataset wrapper classes, dataloaders
 ┣ 📂logs
 ┃ ┗ Log files (train / test / error)
 ┣ 📂model
 ┃ ┗ Model and metrics
 ┣ 📂trainer
 ┃ ┗ Trainer
 ┣ 📂utils
 ┃ ┗ Helper functions
 ┗ Entry points (train.py, test.py, inference.py), configuration, building files.
```

## Configuration
The model configuration can be found in the file `config.json`.

The configuratoin argument parameters are listed below:

| argument    | description  |
|-------------|-------------|
|   num_labels   | Number of labels used for classification |
|   num_gpu   | Number of GPUs to use |
|   save_period_in_epochs   | Number characterizing with what periodicity the checkpoint is saved (in epochs) |
|   metrics   | The classification metrics used are  |
|  pretrained_model_name    | BERT shortcut name from HuggingFace  |
|  table_serialization_type    | Method of serializing a table into a sequence |
|  batch_size    | Batch size |
|  num_epochs    | Number of training epochs |
|  random_seed    | Random seed |
|  logs_dir    | Directory for logging |
|  train_log_filename    | File name for train logging  |
|  test_log_filename    | File name for test logging |
|  start_from_checkpoint    | Flag to start training from checkpoint |
|  checkpoint_dir    | Directory for storing checkpoints of model |
|  checkpoint_name    | File name of a checkpoint (model state) |
|  inference_model_name    | File name of a model for inference |
|  inference_dir    | Directory for storing inference tables `.csv` |
|  dataloader.valid_split    | Amount of validation subset split |
|  dataloader.num_workers    | Number of dataloader workers |
|  dataset.num_rows    | Number of readable rows in the dataset, if `null` read all rows in files |
|  dataset.data_dir    | Directory for storing train/test/inference files |
|  dataset.train_path    | Directory for storing train dataset files `.csv` |
|  dataset.test_path    | Direcotry for storing test dataset files `.csv` |

We recomend to change ONLY theese parameters:
- `num_gpu` - Any positive ingeter number + {0}. `0` stand for training / testing on CPU.
- `save_period_in_epochs` - Any positive integer number, measures in epochs.
- `table_serialization_type` - "column_wise" or "table_wise".
- `pretrained_model_name` - BERT shorcut names from Huggingface PyTorch pretrained models.
- `batch_size` - Any positive integer number.
- `num_epochs` - Any positive integer number.
- `random_seed` - Any integer number.
- `start_from_checkpoint` - "true" or "false".
- `checkpoint_name` - Any name of model, saved in `checkpoint` directory.
- `inference_model_name` - Any name of model, saved in `checkpoint` directory. But we recommend to use the best models: [model_best_f1_weighted.pt, model_best_f1_macro.pt, model_best_f1_micro.pt].
- `dataloader.valid_split` - Real number within range [0.0, 1.0] (0.0 stands for 0 % of train subset, 0.5 stands for 50 % of train subset). Or positive integer number (Denoting a fixed number of a validation subset).
- `dataset.num_rows` - "null" stands for read all lines in dataset files. Positive integer means the number of lines to read in the files of the dataset.


## Dataset files
Before training / testing the model you need to:
1. Download [dataset repository](https://github.com/STI-Team/RuTaBERT-Dataset) in the same directory as RuTaBERT, example dir strucutre:
```
├── src
│  ├── RuTaBERT
│  ├── RuTaBERT-Dataset
│  │  ├── move_dataset.sh
```
3. Run script `move_dataset.sh` from dataset reporitory, to move dataset files into RuTaBERT `data` directory:
```bash
RuTaBERT-Dataset$ ./move_dataset.sh
```
3. configure `config.json` file before training.


---

## Training / Testing
RuTaBERT supports training / testing locally and inside Docker container. Also supports [slurm](https://slurm.schedmd.com/overview.html) workload manager.

### Locally
1. Create virtual environment:
```bash
RuTaBERT$ virtualenv venv
```
or
```bash
RuTaBERT$ python -m virtualenv venv
```

2. Install requirements and start train and test.
```bash
RuTaBERT$ source venv/bin/activate &&\
    pip install -r requirements.txt &&\
    python3 train.py 2> logs/error_train.log &&\
    python3 test.py 2> logs/error_test.log
```

3. Models will be saved in `checkpoint` directory.
4. Output will be in `logs/` directory (`training_results.csv`, `train.log`, `test.log`, `error_train.log`, `error_test.log`).

### Docker
Requirements:
- [Docker installation guide (ubuntu)](https://docs.docker.com/engine/install/ubuntu/)
- NVIDIA driver
- [NVIDIA Container Toolkit installation guide (ubuntu)](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

1. Make sure all dependencies are installed.
2. Build image:
```bash
RuTaBERT$ sudo docker build -t rutabert .
```

3. Run image
```bash
RuTaBERT$ sudo docker run -d --runtime=nvidia --gpus=all \
    --mount source=rutabert_logs,target=/app/rutabert/logs \
    --mount source=rutabert_checkpoints,target=/app/rutabert/checkpoints \
    rutabert
```

4. Move models and logs from container after training / testing.
```bash
RuTaBERT$ sudo cp -r /var/lib/docker/volumes/rutabert_checkpoints/_data ./checkpoints
```

```bash
RuTaBERT$ sudo cp -r /var/lib/docker/volumes/rutabert_logs/_data ./logs
```

5. *Don't forget to remove volumes after training! Docker wont do it for you.*
6. Models will be saved in `checkpoint` directory.
7. Output will be in `logs/` directory (`training_results.csv`, `train.log`, `test.log`, `error_train.log`, `error_test.log`).

### Slurm
1. Run slurm script:
```bash
RuTaBERT$ sbatch run.slurm
```
2. Check job status:
```bash
RuTaBERT$ squeue
```
3. Models will be saved in `checkpoint` directory.
4. Output will be in `logs/` directory (`train.log`, `test.log`, `error_train.log`, `error_test.log`).

## Testing
1. Make sure data placed in `data/test` directory.
2. Configure which model to test in `config.json`.
3. Run:
```bash
RuTaBERT$ source venv/bin/activate &&\
    pip install -r requirements.txt &&\
    python3 test.py 2> logs/error_test.log
```
4. Output will be in `logs/` directory (`test.log`, `error_test.log`).

## Inference
1. Make sure data placed in `data/inference` directory.
2. Configure which model to inference in `config.json`
3. Run:
```bash
RuTaBERT$ source venv/bin/activate &&\
    pip install -r requirements.txt &&\
    python3 inference.py
```
4. Labels will be in `data/inference/result.csv`
