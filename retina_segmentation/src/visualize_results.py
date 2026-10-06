import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from retina_segmentation.src.model import DeepVesselNet
from retina_segmentation.src.data_loader import get_dataloaders


def visualize_predictions():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    test_img_dir = os.path.join(BASE_DIR, "Data", "test", "image")
    test_mask_dir = os.path.join(BASE_DIR, "Data", "test", "mask")
    checkpoint_path = os.path.join(BASE_DIR, "outputs", "checkpoints", "best_model.pth")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepVesselNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # DataLoader
    _, test_loader = get_dataloaders(
        train_img_dir=test_img_dir, train_mask_dir=test_mask_dir,
        test_img_dir=test_img_dir, test_mask_dir=test_mask_dir,
        batch_size=1, img_size=512
    )

    save_dir = os.path.join(BASE_DIR, "outputs", "visuals")
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            pred_mask = (outputs.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

            # Convert for plotting
            img_np = images.squeeze().permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            gt_mask_np = masks.squeeze().cpu().numpy() * 255

            # Plot side-by-side
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_np)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(gt_mask_np, cmap="gray")
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            axes[2].imshow(pred_mask, cmap="gray")
            axes[2].set_title("Predicted Mask")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"comparison_{idx}.png"), dpi=300)
            plt.close()

            if idx < 3:  # Show first 3 examples interactively
                plt.imshow(pred_mask, cmap="gray")
                plt.show()


def main():
    visualize_predictions()


if __name__ == "__main__":
    main()
