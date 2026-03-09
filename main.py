"""
ResNet Fine-Tuning
Progressive fine-tuning of ResNet-50 on a custom dataset with layer unfreezing strategy.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
import warnings
warnings.filterwarnings('ignore')

CLASSES = ['airplane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
NUM_CLASSES = len(CLASSES)


def create_dataset(n_per_class=100, img_size=64, seed=42):
    """Synthetic dataset."""
    torch.manual_seed(seed)
    X, y = [], []
    for cls_id in range(NUM_CLASSES):
        for _ in range(n_per_class):
            img = torch.rand(3, img_size, img_size)
            # Class-specific bias
            img[cls_id % 3] += 0.3
            X.append(img.clamp(0, 1))
            y.append(cls_id)
    X = torch.stack(X)
    y = torch.LongTensor(y)
    idx = torch.randperm(len(X))
    return X[idx], y[idx]


def get_resnet50(num_classes, strategy='head_only'):
    """Load ResNet-50 with different fine-tuning strategies."""
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    except:
        model = models.resnet50(weights=None)

    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    if strategy == 'head_only':
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad_(False)
    elif strategy == 'last_block':
        for name, param in model.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad_(False)
    elif strategy == 'progressive':
        # Start frozen, will unfreeze progressively
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad_(False)
    # full: all params trainable

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Strategy '{strategy}': {trainable:,} / {total:,} trainable params ({trainable/total:.1%})")
    return model


def unfreeze_layers(model, layers_to_unfreeze):
    """Progressively unfreeze layers."""
    for name, param in model.named_parameters():
        for layer_name in layers_to_unfreeze:
            if layer_name in name:
                param.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  After unfreezing {layers_to_unfreeze}: {trainable:,} trainable params")


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.item()
        correct  += (out.argmax(1) == y).sum().item()
        total    += len(y)
    return loss_sum/len(loader), correct/total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out  = model(X)
        loss_sum += criterion(out, y).item()
        correct  += (out.argmax(1) == y).sum().item()
        total    += len(y)
    return loss_sum/len(loader), correct/total


def progressive_fine_tuning(model, tr_ld, te_ld, device, phases):
    """Multi-phase progressive fine-tuning."""
    criterion = nn.CrossEntropyLoss()
    all_accs  = []

    for phase_name, layers_to_unfreeze, n_epochs, lr in phases:
        print(f"\n  Phase: {phase_name}")
        unfreeze_layers(model, layers_to_unfreeze)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        for epoch in range(1, n_epochs + 1):
            tr_loss, tr_acc = train_epoch(model, tr_ld, optimizer, criterion, device)
            te_loss, te_acc = eval_epoch(model,  te_ld, criterion, device)
            scheduler.step()
            all_accs.append({'phase': phase_name, 'tr': tr_acc, 'te': te_acc})
            if epoch == n_epochs:
                print(f"    Final: Train={tr_acc:.4f}, Test={te_acc:.4f}")
    return all_accs


def compare_strategies(X_tr, y_tr, X_te, y_te, device):
    """Compare different fine-tuning strategies."""
    tr_ld = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
    te_ld = DataLoader(TensorDataset(X_te, y_te), batch_size=32)
    criterion = nn.CrossEntropyLoss()
    results   = {}

    for strategy in ['head_only', 'last_block', 'full']:
        print(f"\n--- Strategy: {strategy} ---")
        model = get_resnet50(NUM_CLASSES, strategy).to(device)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        accs = []
        for epoch in range(15):
            train_epoch(model, tr_ld, optimizer, criterion, device)
            _, te_acc = eval_epoch(model, te_ld, criterion, device)
            scheduler.step()
            accs.append(te_acc)
        results[strategy] = accs
        print(f"  Final test accuracy: {accs[-1]:.4f}")
    return results


def plot_strategy_comparison(results, save_path='strategy_comparison.png'):
    plt.figure(figsize=(9, 5))
    for name, accs in results.items():
        plt.plot(accs, lw=2, label=name)
    plt.xlabel('Epoch'); plt.ylabel('Test Accuracy')
    plt.title('ResNet-50 Fine-Tuning Strategy Comparison')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Strategy comparison saved to {save_path}")


def main():
    print("=" * 60)
    print("RESNET-50 FINE-TUNING")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    X, y = create_dataset(n_per_class=100)
    split = int(len(X) * 0.8)
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    X_norm = (X - mean) / std
    X_tr, y_tr = X_norm[:split], y[:split]
    X_te, y_te = X_norm[split:], y[split:]

    # Compare strategies
    print("\n--- Comparing Fine-Tuning Strategies ---")
    results = compare_strategies(X_tr, y_tr, X_te, y_te, device)
    plot_strategy_comparison(results)

    # Progressive fine-tuning demo
    print("\n--- Progressive Fine-Tuning ---")
    prog_model = get_resnet50(NUM_CLASSES, 'progressive').to(device)
    tr_ld = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
    te_ld = DataLoader(TensorDataset(X_te, y_te), batch_size=32)

    phases = [
        ('Head only',    ['fc'],              5, 1e-3),
        ('+ Layer4',     ['layer4', 'fc'],    5, 5e-4),
        ('+ Layer3',     ['layer3', 'layer4', 'fc'], 5, 1e-4),
    ]
    all_accs = progressive_fine_tuning(prog_model, tr_ld, te_ld, device, phases)

    # Plot progressive fine-tuning
    te_accs = [a['te'] for a in all_accs]
    phases_x = [a['phase'] for a in all_accs]
    plt.figure(figsize=(10, 5))
    plt.plot(te_accs, 'g-', lw=2)
    # Phase boundaries
    phase_counts = {}
    for p in phases_x:
        phase_counts[p] = phase_counts.get(p, 0) + 1
    cumsum = 0
    for phase_name, _, n_ep, _ in phases:
        plt.axvline(cumsum, color='r', linestyle='--', alpha=0.5)
        plt.text(cumsum + 0.5, max(te_accs) * 0.5, phase_name, fontsize=8, rotation=90)
        cumsum += n_ep
    plt.xlabel('Epoch'); plt.ylabel('Test Accuracy')
    plt.title('Progressive Fine-Tuning Accuracy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('progressive_finetuning.png', dpi=150)
    plt.close()

    torch.save(prog_model.state_dict(), 'resnet50_finetuned.pth')
    print("\nModel saved to resnet50_finetuned.pth")
    print("\n✓ ResNet Fine-Tuning complete!")


if __name__ == '__main__':
    main()
