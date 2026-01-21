# -*- coding: utf-8 -*-
"""
Custom Data Generators for MIFOCAT Cardiac Segmentation

Supports:
1. Simple .npy loading (legacy)
2. Fold-aware loading for k-fold cross-validation

@author: ramad
Modified for k-fold CV support
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Optional, Generator
import numpy as np


def load_img(img_dir: str, img_list: List[str]) -> np.ndarray:
    """
    Load .npy images from directory.
    
    Args:
        img_dir: Directory path containing .npy files
        img_list: List of image filenames to load
        
    Returns:
        Stacked numpy array of shape (N, *image_shape)
    """
    images = []
    for i, image_name in enumerate(img_list):    
        if image_name.split('.')[-1] == 'npy':
            image = np.load(os.path.join(img_dir, image_name))
            images.append(image)
    
    return np.array(images)


def imageLoader(img_dir: str, img_list: List[str], 
                mask_dir: str, mask_list: List[str], 
                batch_size: int) -> Generator:
    """
    Legacy infinite batch generator for Keras training.
    
    Args:
        img_dir: Directory containing image .npy files
        img_list: List of image filenames
        mask_dir: Directory containing mask .npy files
        mask_list: List of mask filenames
        batch_size: Batch size for yielding
        
    Yields:
        Tuple (X, Y) of batch-sized numpy arrays
    """
    L = len(img_list)
    
    # Keras needs infinite generator
    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            
            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])

            yield (X, Y)
            
            batch_start += batch_size   
            batch_end += batch_size


class FoldAwareDataLoader:
    """
    Fold-aware data loader for k-fold cross-validation.
    
    Filters patient-level data based on fold metadata, ensuring all slices
    of a patient remain in the same split (train/val/test).
    """
    
    def __init__(self, base_dir: str, fold_metadata_path: str):
        """
        Initialize fold-aware loader.
        
        Args:
            base_dir: Base directory containing 2D sliced patient data
                     (assumes structure: base_dir/Pasien <id>/images/)
            fold_metadata_path: Path to kfold_metadata.json from split_data.py
        """
        self.base_dir = Path(base_dir)
        self.fold_metadata_path = Path(fold_metadata_path)
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> dict:
        """Load k-fold metadata JSON."""
        if not self.fold_metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.fold_metadata_path}")
        
        with open(self.fold_metadata_path, 'r') as f:
            return json.load(f)
    
    def get_fold_patients(self, fold_id: int, split: str = 'train') -> List[str]:
        """
        Get list of patient IDs for a specific fold and split.
        
        Args:
            fold_id: Fold index (0 to n_splits-1)
            split: 'train', 'val', or 'test'
            
        Returns:
            List of patient ID strings
        """
        if fold_id >= len(self.metadata['folds']):
            raise ValueError(f"fold_id {fold_id} exceeds n_splits={self.metadata['n_splits']}")
        
        fold_info = self.metadata['folds'][fold_id]
        
        if split not in fold_info:
            raise ValueError(f"split '{split}' not in fold_info. Available: {fold_info.keys()}")
        
        return fold_info[split]
    
    def get_file_list(self, patient_ids: List[str], 
                     image_subdir: str = 'images') -> Tuple[List[str], str]:
        """
        Get list of image filenames for given patients.
        
        Assumes folder structure: base_dir/Pasien <id>/<image_subdir>/*.npy
        
        Args:
            patient_ids: List of patient IDs
            image_subdir: Subdirectory name within patient folder (default 'images')
            
        Returns:
            Tuple of (file_list, directory_path) for use with load_img()
        """
        file_list = []
        
        for pid in patient_ids:
            patient_dir = self.base_dir / f"Pasien {pid}" / image_subdir
            
            if not patient_dir.exists():
                print(f"[FoldAwareDataLoader] Warning: {patient_dir} not found, skipping patient {pid}")
                continue
            
            # Get all .npy files in this patient's image directory
            npy_files = sorted([f.name for f in patient_dir.glob("*.npy")])
            file_list.extend(npy_files)
        
        # Return first patient's directory (assumes all are same structure)
        if patient_ids:
            first_patient_dir = self.base_dir / f"Pasien {patient_ids[0]}" / image_subdir
            return file_list, str(first_patient_dir)
        else:
            raise ValueError("No valid patient IDs provided")
    
    def get_generators(self, fold_id: int, batch_size: int = 8,
                      image_subdir: str = 'images',
                      mask_subdir: str = 'masks') -> Tuple[Generator, Generator, int, int]:
        """
        Get train and validation generators for a specific fold.
        
        Args:
            fold_id: Fold index
            batch_size: Batch size for generators
            image_subdir: Subdirectory containing images
            mask_subdir: Subdirectory containing masks
            
        Returns:
            Tuple of (train_generator, val_generator, train_steps, val_steps)
        """
        # Get patient lists
        train_patients = self.get_fold_patients(fold_id, 'train')
        val_patients = self.get_fold_patients(fold_id, 'val')
        
        print(f"[FoldAwareDataLoader] Fold {fold_id}: "
              f"{len(train_patients)} train patients, {len(val_patients)} val patients")
        
        # Get file lists
        train_files, train_img_dir = self.get_file_list(train_patients, image_subdir)
        val_files, val_img_dir = self.get_file_list(val_patients, image_subdir)
        
        # Construct mask paths (assumes Pasien <id>/masks/ structure)
        # This is a simplified assumption; adjust based on your actual structure
        train_mask_dir = str(Path(train_img_dir).parent / mask_subdir)
        val_mask_dir = str(Path(val_img_dir).parent / mask_subdir)
        
        print(f"[FoldAwareDataLoader] Train files: {len(train_files)}, Val files: {len(val_files)}")
        
        # Create generators
        train_gen = imageLoader(
            train_img_dir, train_files,
            train_mask_dir, train_files,  # Assume same filenames for masks
            batch_size
        )
        val_gen = imageLoader(
            val_img_dir, val_files,
            val_mask_dir, val_files,
            batch_size
        )
        
        train_steps = max(1, len(train_files) // batch_size)
        val_steps = max(1, len(val_files) // batch_size)
        
        return train_gen, val_gen, train_steps, val_steps