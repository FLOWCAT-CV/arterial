import os
import numpy as np
import torch
import nibabel as nib
import cc3d
import cv2
import torchio as tio  # Required for resampling operations

from utils import (
    actualizar_json_con_predicciones,
    restore_centroids_to_original_origin,
    resample_image,
    crop_or_pad_image,
    change_origin_preprocess,
)
from model import load_trained_model_seg
from io.load_and_save_operations import load_nifti, save_nifti

class SingleCTADataset:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        cta_path = os.path.join(self.folder_path, "cta.nii.gz")
        preprocessed_path = os.path.join(self.folder_path, "cta_preprocessed.nii.gz")

        og_img = nib.load(cta_path)
        tio_object = tio.ScalarImage(cta_path)

        resampled = resample_image(tio_object, (0.8, 0.8, 0.8))
        cropped_or_padded = crop_or_pad_image(resampled, (320, 320, 480))
        final_image = change_origin_preprocess(cropped_or_padded, (0, 0, 0))

        final_image.save(preprocessed_path)
        img = nib.load(preprocessed_path)

        volume = np.clip(img.get_fdata().astype(np.float32), 0, 700) / 700
        volume = torch.tensor(
            np.expand_dims(np.transpose(volume, (2, 0, 1)), axis=0),
            dtype=torch.float32
        )

        return {
            "volume": volume,
            "affine": og_img.affine,
            "folder": self.folder_path
        }

class LandmarkAutomator:
    def __init__(self, model_path: str, device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = load_trained_model_seg(model_path, self.device)
        self.model.eval()

    def _prepare_input(self, folder_path: str):
        dataset = SingleCTADataset(folder_path)
        sample = dataset[0]
        volume = sample["volume"].unsqueeze(0).to(self.device)
        return volume, sample

    def _postprocess_and_save(self, preds, sample, output_folder, save_mask=True, save_json=True):
        preds = preds.cpu().numpy()[0]  # (5, D, H, W)
        combined = np.zeros(preds.shape[1:], dtype=np.uint8)
        confidence_map = np.zeros(preds.shape[1:], dtype=np.float32)

        for c in range(1, preds.shape[0]):
            mask = (preds[c] > -1) & ((preds[c] > confidence_map) | (combined == 0))
            confidence_map[mask] = preds[c][mask]
            combined[mask] = c

        centroids_voxel = np.zeros((4, 3), dtype=np.float32)
        all_largest_components = []
        for label in range(1, 5):
            binary_mask = (combined == label).astype(np.uint8)
            kernel = np.ones((2, 2), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

            labels_cc = cc3d.largest_k(binary_mask, k=1, connectivity=26)
            all_largest_components.append(labels_cc)
            stats = cc3d.statistics(labels_cc)
            centroid = stats["centroids"][1]
            centroids_voxel[label - 1] = [centroid[1], centroid[2], centroid[0]]

        combined_largest_components = np.zeros(preds.shape[1:], dtype=np.uint8)
        for i, mask in enumerate(all_largest_components):
            combined_largest_components[mask] = i + 1

        centroids_mm = centroids_voxel * 0.8
        affine = sample["affine"]
        orientation = nib.aff2axcodes(affine)

        if orientation == ('L', 'A', 'S'):
            centroids_mm[:, 1] = -centroids_mm[:, 1]

        affine_path = os.path.join(sample["folder"], "affine_before_origin_change.txt")
        centroids_mm = restore_centroids_to_original_origin(centroids_mm, affine_path, orientation)

        output_folder_img = os.path.join(output_folder, os.path.basename(sample["folder"]))
        os.makedirs(output_folder_img, exist_ok=True)

        if save_json:
            output_json_path = os.path.join(output_folder_img, "F_o.json")
            input_json_path = os.path.join(sample["folder"], "F.json")
            actualizar_json_con_predicciones(centroids_mm, output_json_path, input_json_path)

        if save_mask:
            combined_img = np.transpose(combined_largest_components, (1, 2, 0))
            save_nifti(nib.Nifti1Image(combined_img, affine), os.path.join(output_folder_img, "pred_mask.nii.gz"))

    def infer_folder(self, folder_path: str, output_folder: str, save_mask=True, save_json=True):
        volume, sample = self._prepare_input(folder_path)
        with torch.no_grad():
            preds = self.model(volume)
        self._postprocess_and_save(preds, sample, output_folder, save_mask=save_mask, save_json=save_json)
        print(f"Inference done for: {folder_path}")
