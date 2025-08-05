# 🧠 Tumor_Detection

This project performs tumor segmentation in prostate WSIs using supervised and semi-supervised learning. Follow the steps below to set up the environment, prepare the data, train the model, and evaluate the results.

---

## ⚙️ Environment Setup

Install [uv](https://github.com/astral-sh/uv) and create a reproducible Python environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
uv init
uv venv
source .venv/bin/activate
uv pip install -r requirements.lock.txt
```

---

## 📁 Dataset Preparation

Download the dataset:

- Training WSIs
- Validation WSIs

Extract patches from WSIs:

```bash
python src/dataset/create_pathches.py
```

> ⚠️ Edit the script to specify:
> - Path to dataset
> - Path to store extracted patches

Update the config:

- Open `src/config.py`
- Set paths for extracted image and mask patches
- Adjust model hyperparameters as needed

---

Training

Supervised training 
run src/train.py 

run tensorboard to visulise loss and class wise iou for train and val set. 
tensorboard --logdir=runs/finetune_20250805-184611 --port=6006


Semi superwise fine tunning :

use extra 30 wsi (does not have mask)
run file inside src/dataset 
1. src/dataset/extract_patches_unlabled.py # extract the relevent patches 
2. src/dataset/hard_mining.py # get hard samples from patches
3. src/generate_pseudo_lables.py # generate pseudo lables using best trained model
4. src/train_with_pseudo.py  # train with real training + pseudo label dataset

run tensorboard

Evaluate the model 
download trained model from shared gdrive link 

run src/evaluate_wsi_level.py # verify Validation_path and save_dir path in config.py file 
this gives patch and wsi level each class iou. 
