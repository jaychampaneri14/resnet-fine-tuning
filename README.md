# ResNet-50 Fine-Tuning

Compares three fine-tuning strategies for ResNet-50 and demonstrates progressive layer unfreezing.

## Strategies Compared
1. **Head Only** — train only the new FC head (frozen backbone)
2. **Last Block** — unfreeze `layer4` + head
3. **Full** — all layers trainable (with low LR)
4. **Progressive** — gradually unfreeze: head → layer4 → layer3

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Output
- `strategy_comparison.png` — test accuracy across epochs per strategy
- `progressive_finetuning.png` — accuracy during progressive unfreezing
- `resnet50_finetuned.pth` — final model weights

## Key Insight
Progressive fine-tuning often outperforms full fine-tuning from the start, especially with small datasets.
