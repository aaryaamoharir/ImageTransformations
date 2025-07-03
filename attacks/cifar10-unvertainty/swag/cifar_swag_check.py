import tensorflow_datasets as tfds
import torch
import torchvision.transforms as T
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import pathlib 
import logging
from typing import Union, Dict, List
from torch.utils.data import DataLoader
import torch.nn.functional as F

from swag_wrapper import SWAGWrapper
from model_wrapper import DistributionalWrapper
from metric import AverageMeter # Although AverageMeter is not directly used in this script, it's from the provided utils.
from metric import entropy
from context import DefaultContext
from collate import fast_collate



def load_cifar10c():
    ds = tfds.load(
        #'cifar10_corrupted',
        'cifar10',
        split='test',
        shuffle_files=False,
        as_supervised=True,
    )
    return ds
def preprocess_tf_to_torch(image, label):
    # image: tf.uint8 Tensor [32,32,3]
    #img = image.numpy()
    if hasattr(image, 'numpy'): # Check if the object has a .numpy() method (typical for EagerTensors)
        image = image.numpy()

    # As a fallback, ensure it's a NumPy array (e.g., if it's another tensor type or not recognized by previous check)
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    img = T.ToTensor()(image)               # scales to [0,1]
    img = T.Normalize((0.4914, 0.4822, 0.4465),
                      (0.2023, 0.1994, 0.2010))(img)
    return img, int(label)

def create_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    is_training_dataset: bool,
    mean: List[float], # Not strictly needed here, but kept for consistency with original signature
    std: List[float],  # Not strictly needed here, but kept for consistency with original signature
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    device: torch.device,
):
    """Creates a PyTorch DataLoader."""
    collate_fn = fast_collate

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_training_dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training_dataset,
        persistent_workers=persistent_workers,
    )
    return loader

class TFDatasetToTorchIterable:
    """Helper class to convert TensorFlow Datasets to PyTorch IterableDataset."""
    def __init__(self, tf_dataset, preprocess_fn, transform):
        self.tf_dataset = tf_dataset
        self.preprocess_fn = preprocess_fn
        self.transform = transform
        self.dataset_list = []
        print("Preprocessing TF data...")
        for img_tf, lbl_tf in tqdm(tf_dataset):
            img, lbl = preprocess_fn(img_tf, lbl_tf)
            self.dataset_list.append((img, lbl))

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        img_tensor, label_int = self.dataset_list[idx]
        return img_tensor, label_int
# --- End Helper Functions ---


def main():
    # load dataset from python package 
    ds = load_cifar10c()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load pretrained cifar 10 model 
    model_one = torch.hub.load(
        'chenyaofo/pytorch-cifar-models',
        'cifar10_resnet56',
        pretrained=True
    ).eval().to('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_dir = pathlib.Path('swag_checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    model_weights_path = checkpoint_dir / 'cifar10_resnet56_pretrained.pt'
    checkpoint_data = {
        'state_dict': model_one.state_dict(),
    }
    torch.save(checkpoint_data, model_weights_path)

    print(f"Model weights saved to {model_weights_path}")
    
    ds_eval = load_cifar10c()
    cifar10_test_iterable = TFDatasetToTorchIterable(ds_eval, preprocess_tf_to_torch, T.Compose([]))
    test_loader = create_loader(
        dataset=cifar10_test_iterable,
        batch_size=128,
        is_training_dataset=False,
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        device=torch.device(device)
    )

     # 2.2 Train DataLoader (used for SWAG statistics collection and BatchNorm re-estimation)
    train_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    cifar10_train_dataset = tfds.as_numpy(tfds.load('cifar10', split='train', as_supervised=True, shuffle_files=False))

    cifar10_train_iterable = TFDatasetToTorchIterable(cifar10_train_dataset, preprocess_tf_to_torch, train_transform)
    train_loader = create_loader(
        dataset=cifar10_train_iterable,
        batch_size=128,
        is_training_dataset=True,
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        device=torch.device(device)
    )

    all_preds, all_labels = [], []
    
    print("\n--- Verifying direct loading of pretrained weights ---")
    model = torch.hub.load(
        'chenyaofo/pytorch-cifar-models',
        'cifar10_resnet56',
        pretrained=False # Start with random weights
    ).eval().to('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_weights_path, map_location="cpu", weights_only=True)
    state_dict_from_checkpoint = checkpoint["state_dict"]

    # Load the state_dict onto the new model instance
    model.load_state_dict(state_dict_from_checkpoint, strict=True)
    print("State dict loaded successfully onto direct_load_test_model.")





    #get the prediction of each image + the actual label and store it 
    for img_tf, label in tfds.as_numpy(ds):
        img, lbl = preprocess_tf_to_torch(img_tf, label)
        img = img.unsqueeze(0).to(next(model.parameters()).device)
        with torch.no_grad():
            logits = model(img)
            pred = logits.argmax(dim=1).cpu().item()

        all_preds.append(pred)
        all_labels.append(int(lbl))

    # final metrics 
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print("CIFAR‑10‑C Evaluation with ResNet56:")
    print(f"  → Accuracy : {acc:.4f}")
    print(f"  → Precision: {prec:.4f}")
    print(f"  → Recall   : {rec:.4f}")
    print(f"  → F1‑Score : {f1:.4f}")


    base_model_for_swag = torch.hub.load(
        'chenyaofo/pytorch-cifar-models',
        'cifar10_resnet56',
        pretrained=False # SWAGWrapper will load weights from file
    ).eval().to(device)

    # Instantiate SWAGWrapper.
    # It will load the weights from 'model_weights_path' onto 'base_model_for_swag'.
    swag_model = SWAGWrapper(
        model=base_model_for_swag,
        weight_path=str(model_weights_path), # Pass the path to your saved pretrained weights
        use_low_rank_cov=True,
        max_rank=20
    )
    swag_model.to(device)

    print("\n--- Verifying initial accuracy of swag_model.model ---")
    swag_model.eval() # Set swag_model to eval mode for this check

    temp_preds, temp_labels = [], []
    for images, labels in tqdm(test_loader, desc="Initial SWAG model check"):
        images = images.to(device)
        images = images.to(torch.float32)

        with torch.no_grad():
            logits = swag_model.model(images) # Evaluate swag_model.model
            temp_pred = logits.argmax(dim=1).cpu()

        temp_preds.extend(temp_pred.tolist())
        temp_labels.extend(labels.tolist())

    initial_swag_model_accuracy = accuracy_score(temp_labels, temp_preds)
    print(f"Initial swag_model.model Accuracy (before SWAG collection): {initial_swag_model_accuracy:.4f}")
    swag_model.train() # Set swag_model back to train mode for statistics collection


if __name__ == "__main__":
    main()
