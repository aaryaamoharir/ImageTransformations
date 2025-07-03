import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import pathlib
from typing import Union, Dict, List
import tensorflow_datasets as tfds
import torch
import torchvision.transforms as T
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import logging 
#import the code taken from the github
from swag_wrapper import SWAGWrapper # Assuming swag_wrapper.py is in the same directory
from model_wrapper import DistributionalWrapper # Assuming model_wrapper.py is in the same directory
from metric import AverageMeter # Assuming utils_metrics.py is in the same directory
from metric import entropy # Assuming utils_metrics.py is in the same directory
from context import DefaultContext # Assuming utils_context.py is in the same directory
from collate import fast_collate # Assuming utils_collate.py is in the same directory
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_cifar10c():
    ds = tfds.load(
        'cifar10',
        split='test',
        shuffle_files=False,
        as_supervised=True,
    )
    return ds

def preprocess_tf_to_torch(image, label):
    if not isinstance(image, np.ndarray):
        image = image.numpy()
    img = T.ToTensor()(image)
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
    collate_fn = fast_collate # Always use fast_collate in this simplified version

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

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #loads and saves the pretrained model 
    model = torch.hub.load(
        'chenyaofo/pytorch-cifar-models',
        'cifar10_resnet56',
        pretrained=True
    ).eval().to(device)

    checkpoint_dir = pathlib.Path('swag_checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    model_weights_path = checkpoint_dir / 'cifar10_resnet56_pretrained.pt'
    checkpoint_data = {
        'state_dict': model.state_dict(),
    }
    torch.save(checkpoint_data, model_weights_path)
    print(f"Model weights saved to {model_weights_path}")

    # 2. Prepare CIFAR-10 Training Dataset Loader for SWAG sampling
    train_transform = T.Compose([ # This transform is currently not used due to preprocess_tf_to_torch
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    cifar10_train_dataset = tfds.as_numpy(tfds.load('cifar10', split='train', as_supervised=True, shuffle_files=False))

    class TFDatasetToTorchIterable:
        def __init__(self, tf_dataset, preprocess_fn, transform):
            self.tf_dataset = tf_dataset
            self.preprocess_fn = preprocess_fn
            self.transform = transform # This is currently unused in __getitem__
            self.dataset_list = []
            print("Preprocessing TF training data for SWAG...") # This specific print belongs to the train iterable
            for img_tf, lbl_tf in tqdm(tf_dataset):
                img, lbl = preprocess_fn(img_tf, lbl_tf)
                self.dataset_list.append((img, lbl))

        def __len__(self):
            return len(self.dataset_list)

        def __getitem__(self, idx):
            img_tensor, label_int = self.dataset_list[idx]
            return img_tensor, label_int

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

     # 3. Instantiate SWAGWrapper
    base_model_for_swag = torch.hub.load(
        'chenyaofo/pytorch-cifar-models',
        'cifar10_resnet56',
        pretrained=False # SWAGWrapper loads the weights
    ).eval().to(device) # Keep in eval mode initially

    swag_model = SWAGWrapper(
        model=base_model_for_swag,
        weight_path=str(model_weights_path), # This path should point to your *fully trained* model
        use_low_rank_cov=True,
        max_rank=20
    )
    swag_model.to(device)

    print("Collecting SWAG statistics")
    #swag collection statsitics happen in train mode ig 
    swag_model.train()
    num_swag_updates = 0
    min_updates_for_low_rank_cov = swag_model._max_rank + 1

    SWAG_COLLECTION_EPOCHS = 50 

    for epoch in range(SWAG_COLLECTION_EPOCHS):
        #print staetment for debugging 
        print(f"--- SWAG Collection Epoch {epoch + 1}/{SWAG_COLLECTION_EPOCHS} ---")
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"SWAG Epoch {epoch+1}")):
            images = images.to(device)
            labels = labels.to(device)
            images = images.to(torch.float32)

            with torch.no_grad(): # Ensure no gradients are computed to prevent accidental training
                _ = swag_model.model(images) # This is just to run data through for BatchNorm updates if needed

            swag_model.update_stats() # This is where SWAG collects its mean and covariance
            num_swag_updates += 1

    # Ensure enough updates if using low-rank covariance
    if swag_model._use_low_rank_cov and num_swag_updates < min_updates_for_low_rank_cov:
        raise ValueError(f"Not enough SWAG updates for low-rank covariance. Need at least {min_updates_for_low_rank_cov}, but only collected {num_swag_updates}.")

    swag_model.eval() # Set model back to eval mode after stats collection
    print(f"SWAG statistics collected from {num_swag_updates} updates.")


    print("Collecting SWAG samples for uncertainty approximation (this can take a while)...")
    swag_model.get_mc_samples(
        train_loader=train_loader, # Still needs the train_loader for BatchNorm re-estimation
        num_mc_samples=30, # Number of weight snapshots to collect for SWAG approximation
        channels_last=False
    )
    print("SWAG samples collected.")

    swag_model.eval() # Set to evaluation mode
    all_preds, all_labels = [], []
    all_uncertainties = {
        'entropies_of_bma': [],
        'one_minus_max_probs_of_bma': [],
        'jensen_shannon_divergences': [],
        'expected_variances_of_probs': []
    }

    # Evaluation Dataset Loader
    ds_eval = load_cifar10c()
    # Using the same TFDatasetToTorchIterable for test data for consistency
    cifar10_test_iterable = TFDatasetToTorchIterable(ds_eval, preprocess_tf_to_torch, T.Compose([])) # No extra transform needed if preprocess handles it

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

    num_inference_mc_samples = 10 # Number of MC samples to draw during inference for prediction

    print("Starting evaluation with SWAG...")
    for images, labels in tqdm(test_loader, desc="Evaluating with SWAG"):
        images = images.to(device)
        images = images.to(torch.float32)
        labels = labels.to('cpu')

        with torch.no_grad():
            inference_output = swag_model(images, num_mc_samples=num_inference_mc_samples, amp_autocast=DefaultContext())
            sampled_logits = inference_output['logit']

            avg_probs = F.softmax(sampled_logits, dim=-1).mean(dim=1)
            pred = avg_probs.argmax(dim=1).cpu()

            # Uncertainty calculations (as before)
            entropies_of_bma = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1)
            one_minus_max_probs_of_bma = 1 - avg_probs.max(dim=-1)[0]

            log_probs_individual = F.log_softmax(sampled_logits, dim=-1)
            probs_individual = log_probs_individual.exp()
            expected_entropy = -torch.sum(probs_individual * log_probs_individual, dim=-1).mean(dim=-1)
            jensen_shannon_divergence = entropies_of_bma - expected_entropy
            expected_variances_of_probs = torch.var(probs_individual, dim=1).mean(dim=-1)

        all_preds.extend(pred.tolist())
        all_labels.extend(labels.tolist())
        all_uncertainties['entropies_of_bma'].extend(entropies_of_bma.tolist())
        all_uncertainties['one_minus_max_probs_of_bma'].extend(one_minus_max_probs_of_bma.tolist())
        all_uncertainties['jensen_shannon_divergences'].extend(jensen_shannon_divergence.tolist())
        all_uncertainties['expected_variances_of_probs'].extend(expected_variances_of_probs.tolist())

    print("\nSWAG Evaluation complete.")
    print(f"Total predictions: {len(all_preds)}")
    print(f"Average Entropy of BMA: {np.mean(all_uncertainties['entropies_of_bma']):.4f}")
    print(f"Average 1 - Max Prob of BMA: {np.mean(all_uncertainties['one_minus_max_probs_of_bma']):.4f}")
    print(f"Average Jensen-Shannon Divergence: {np.mean(all_uncertainties['jensen_shannon_divergences']):.4f}")
    print(f"Average Expected Variance of Probs: {np.mean(all_uncertainties['expected_variances_of_probs']):.4f}")
    overall_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Overall Accuracy: {overall_accuracy:.4f}")

if __name__ == "__main__":
    main()
