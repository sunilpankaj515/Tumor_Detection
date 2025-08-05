import numpy as np 

#  4-class color mapping
PALETTE = {
    (0, 0, 0): 0,          # Background - Black
    (0, 0, 255): 1,        # Stroma - Blue
    (0, 255, 0): 2,        # Benign - Green
    (255, 255, 0): 3       # Tumor - Yellow
}

PALETTE_RGB = {
    0: (0, 0, 0),         # Background
    1: (0, 0, 255),       # Stroma
    2: (0, 255, 0),       # Benign
    3: (255, 255, 0)      # Tumor
}


def rgb_to_label(mask_rgb):
    """
    Convert a 3-channel RGB mask into a 2D mask of class indices.
    """
    h, w, _ = mask_rgb.shape
    mask_label = np.zeros((h, w), dtype=np.uint8)

    for color, cls_id in PALETTE.items():
        matches = np.all(mask_rgb == np.array(color), axis=-1)
        mask_label[matches] = cls_id

    return mask_label

def label_to_rgb(mask):
    """
    Convert a [H, W] mask with integer class labels into an RGB image [H, W, 3].
    """
    mask = mask.astype(np.uint8)  # Ensure mask is uint8
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for cls_id, color in PALETTE_RGB.items():
        cls_mask = (mask == cls_id)
        rgb_mask[cls_mask] = np.array(color, dtype=np.uint8)

    return rgb_mask