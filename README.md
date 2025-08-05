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

## 🧪 Supervised Training

Train the model using labeled data:

```bash
python src/train.py
```

Visualize training with TensorBoard:

```bash
tensorboard --logdir=runs/finetune_20250805-184611 --port=6006
```

---

## 🤖 Semi-Supervised Fine-Tuning

Use extra 30 WSIs (without masks) to enhance performance:

### 1. Extract relevant patches:

```bash
python src/dataset/extract_patches_unlabled.py
```

### 2. Mine hard examples:

```bash
python src/dataset/hard_mining.py
```

### 3. Generate pseudo-labels using best supervised model:

```bash
python src/generate_pseudo_lables.py
```

### 4. Train with real + pseudo-labeled dataset:

```bash
python src/train_with_pseudo.py
```

Visualize training with TensorBoard:

```bash
tensorboard --logdir=runs --port=6006
```

---

## 🧾 Evaluation

Download the trained model from the provided Google Drive link.

Evaluate on validation set:

```bash
python src/evaluate_wsi_level.py
```

> ✅ Make sure these are correctly set in `src/config.py`:
> - `Validation_path`
> - `save_dir`

This will output patch-level and WSI-level IoU scores for each class.

## Visulize and save output 
```bash
notebook/visulize.ipynb
```


---

