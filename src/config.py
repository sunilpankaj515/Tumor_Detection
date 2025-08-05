import torch 

CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Data
    "train_img_dir": "patches/train/images",
    "train_mask_dir": "patches/train/masks",
    "val_img_dir": "patches/val/images",
    "val_mask_dir": "patches/val/masks",
    "label_cache": "patches/patch_labels.pkl",

    "pseudo_img_dir": "patches/pseudo/mined_img",
    "pseudo_mask_dir": "patches/pseudo/mined_masks",
    "best_checkpoint_path": "models/finetune_20250805-184611/best_model.pth",
    
    #val 
    "Validation_path":  "data/Validation", 
    "Save_dir": "data/Validation/pred_mask", 
    #"models/run_20250805-105346/best_model.pth", #"models/run_20250804-193621/best_model.pth", #"models/run_20250804-171951/best_model.pth",

    # Model
    "num_classes": 4,

    # Training hyperparameters
    "batch_size": 64, #32
    "learning_rate": 1e-3,
    "num_epochs": 50,
    "early_stopping_patience": 10,

    # Logging
    "log_dir": "runs/",

    # Output
    "checkpoint_path": "models/"
}
