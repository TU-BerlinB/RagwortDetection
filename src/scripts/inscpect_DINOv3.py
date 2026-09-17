import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"
IMAGE_PATH = "data/dino_dataset/ragwort/jkk_jkk0001_jpg.rf.7927955a370eff71ccacce7dd950e3c1.jpg"
REPORT_PATH = "outputs/dinov3_inspection.json"


def tensor_info(tensor):
    """Return basic information about a tensor."""
    if tensor is None:
        return None

    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def inspect_model(model, processor, image):
    print("\n" + "=" * 60)
    print("DINOv3 INSPECTION")
    print("=" * 60)

    # ---------------------------------------------------------
    # Model information
    # ---------------------------------------------------------

    print("\n[MODEL]")
    print(f"Model: {MODEL_NAME}")

    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    config = model.config

    print("\n[CONFIG]")

    config_attributes = [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "hidden_act",
        "layer_norm_eps",
        "image_size",
        "patch_size",
        "num_channels",
    ]

    config_data = {}

    for attribute in config_attributes:
        value = getattr(config, attribute, None)
        config_data[attribute] = value
        print(f"{attribute}: {value}")

    # ---------------------------------------------------------
    # Processor information
    # ---------------------------------------------------------

    print("\n[PROCESSOR]")

    print(f"Processor type: {type(processor).__name__}")

    processor_data = {}

    for attribute in [
        "size",
        "crop_size",
        "shortest_edge",
        "do_resize",
        "do_center_crop",
        "image_mean",
        "image_std",
    ]:
        value = getattr(processor, attribute, None)

        processor_data[attribute] = value

        print(f"{attribute}: {value}")

    # ---------------------------------------------------------
    # Image
    # ---------------------------------------------------------

    print("\n[INPUT IMAGE]")

    print(f"Path: {IMAGE_PATH}")
    print(f"Original image size: {image.size}")
    print(f"Image mode: {image.mode}")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    print(f"Pixel values shape: {list(inputs['pixel_values'].shape)}")

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------

    device = next(model.parameters()).device

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    print("\n[FORWARD PASS]")

    with torch.no_grad():
        outputs = model(**inputs)

    print(f"Output type: {type(outputs).__name__}")

    # ---------------------------------------------------------
    # Output structure
    # ---------------------------------------------------------

    print("\n[OUTPUT]")

    output_keys = []

    if hasattr(outputs, "keys"):
        output_keys = list(outputs.keys())

    print(f"Available outputs: {output_keys}")

    output_data = {}

    for key in output_keys:
        value = getattr(outputs, key, None)

        if torch.is_tensor(value):
            print(f"{key}: {list(value.shape)}")

            output_data[key] = tensor_info(value)

        elif value is not None:
            print(f"{key}: {type(value).__name__}")

    # ---------------------------------------------------------
    # Last hidden state
    # ---------------------------------------------------------

    print("\n[LAST HIDDEN STATE]")

    last_hidden_state = getattr(outputs, "last_hidden_state", None)

    if last_hidden_state is not None:

        print(f"Shape: {list(last_hidden_state.shape)}")

        batch_size = last_hidden_state.shape[0]
        num_tokens = last_hidden_state.shape[1]
        embedding_dim = last_hidden_state.shape[2]

        print(f"Batch size: {batch_size}")
        print(f"Number of tokens: {num_tokens}")
        print(f"Embedding dimension: {embedding_dim}")

    else:
        print("last_hidden_state not available.")

        batch_size = None
        num_tokens = None
        embedding_dim = None

    # ---------------------------------------------------------
    # Token analysis
    # ---------------------------------------------------------

    print("\n[TOKEN ANALYSIS]")

    token_data = {}

    if last_hidden_state is not None:

        image_size = config_data.get("image_size")
        patch_size = config_data.get("patch_size")

        token_data["total_tokens"] = num_tokens
        token_data["embedding_dimension"] = embedding_dim

        if image_size is not None and patch_size is not None:

            patches_h = image_size // patch_size
            patches_w = image_size // patch_size
            expected_patches = patches_h * patches_w

            print(f"Expected patch grid: {patches_h} × {patches_w}")
            print(f"Expected number of patches: {expected_patches}")

            token_data["patch_grid"] = [patches_h, patches_w]
            token_data["expected_patches"] = expected_patches

            special_tokens = num_tokens - expected_patches

            print(f"Special tokens: {special_tokens}")

            token_data["special_tokens"] = special_tokens

            if special_tokens > 0:
                print(
                    "The output contains special tokens in addition "
                    "to patch tokens."
                )

        else:
            print(
                "Could not determine patch grid because "
                "image_size or patch_size is unavailable."
            )

    # ---------------------------------------------------------
    # Patch features
    # ---------------------------------------------------------

    print("\n[PATCH FEATURES]")

    patch_features = None
    patch_data = {}

    if last_hidden_state is not None:

        image_size = config_data.get("image_size")
        patch_size = config_data.get("patch_size")

        if image_size is not None and patch_size is not None:

            patches_h = image_size // patch_size
            patches_w = image_size // patch_size
            expected_patches = patches_h * patches_w

            if num_tokens >= expected_patches:

                patch_features = last_hidden_state[:, -expected_patches:, :]

                print(
                    f"Patch features shape: "
                    f"{list(patch_features.shape)}"
                )

                patch_data["shape"] = list(patch_features.shape)

                # Reshape:
                # [B, N, C]
                # ->
                # [B, C, H, W]

                spatial_features = patch_features.reshape(
                    batch_size,
                    patches_h,
                    patches_w,
                    embedding_dim,
                )

                spatial_features = spatial_features.permute(
                    0,
                    3,
                    1,
                    2,
                )

                print(
                    f"Spatial feature map shape: "
                    f"{list(spatial_features.shape)}"
                )

                patch_data["spatial_shape"] = list(
                    spatial_features.shape
                )

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------

    report = {
        "model": MODEL_NAME,
        "parameters": {
            "total": total_parameters,
            "trainable": trainable_parameters,
        },
        "config": config_data,
        "processor": processor_data,
        "input": {
            "image_path": IMAGE_PATH,
            "original_size": list(image.size),
            "mode": image.mode,
            "pixel_values_shape": list(
                inputs["pixel_values"].shape
            ),
        },
        "outputs": output_data,
        "tokens": token_data,
        "patch_features": patch_data,
    }

    report_path = Path(REPORT_PATH)
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(report_path, "w") as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    print("\n" + "=" * 60)
    print("REPORT SAVED")
    print("=" * 60)
    print(REPORT_PATH)


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    print(f"Loading model: {MODEL_NAME}")

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME
    ).to(device)

    model.eval()

    image = Image.open(IMAGE_PATH).convert("RGB")

    inspect_model(
        model=model,
        processor=processor,
        image=image,
    )


if __name__ == "__main__":
    main()