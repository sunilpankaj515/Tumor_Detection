import torch
import segmentation_models_pytorch as smp

def unet():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",  # Set to None to avoid overwriting with ImageNet weights
        in_channels=3,
        classes=4
    )
    # Load custom pretrained weights (e.g., from tumor segmentation)
    #model = torch.load("/home/ubuntu/project/Tumor_Detection/models/run_20250805-121642/best_model.pth", weights_only=False)

    return model


def deeplabv3():
    return smp.DeepLabV3(
        encoder_name="resnet34",           # Or try "resnet101", "timm-efficientnet-b4", etc.
        encoder_weights="imagenet",        # Pretrained
        in_channels=3,
        classes=4
    )

def deeplabv3_plus():
    return smp.DeepLabV3Plus(
        encoder_name="resnet34",           # Or try "resnet101", "timm-efficientnet-b4", etc.
        encoder_weights="imagenet",        # Pretrained
        in_channels=3,
        classes=4
    )

def get_segformer(num_classes=4, in_channels=3):
    model = smp.Segformer(
        encoder_name="resnet50",                # Backbone from TIMM (e.g., mit_b0 to mit_b5)
        encoder_weights="imagenet",           # Pretrained on ImageNet
        decoder_segmentation_channels=256,    # Number of channels in decoder
        in_channels=in_channels,              # RGB = 3
        classes=num_classes,                  # Your target classes
        upsampling=4,                         # Output stride adjustment
        activation=None                       # Keep raw logits
    )
    return model

def get_model():
    return unet()
