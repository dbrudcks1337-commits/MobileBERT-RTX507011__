import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

epoch_results = [
    (102455.0518, 0.7800, 0.7712),  # Epoch 1
    (0.7496,      0.8709, 0.8150),  # Epoch 2
    (0.3339,      0.9256, 0.8275),  # Epoch 3
    (0.2541,      0.9650, 0.8512),  # Epoch 4
    (0.1807,      0.9766, 0.8550),  # Epoch 5
    (0.4280,      0.9850, 0.8562),  # Epoch 6
    (0.1099,      0.9906, 0.8575),  # Epoch 7
    (0.0861,      0.9916, 0.8550),  # Epoch 8
]

def make_training_chart():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    epochs     = list(range(1, len(epoch_results) + 1))
    train_loss = [r[0] for r in epoch_results]
    train_acc  = [r[1] * 100 for r in epoch_results]
    valid_acc  = [r[2] * 100 for r in epoch_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('MobileBERT Fine-tuning Results', fontsize=13, fontweight='bold')

    # Loss (Epoch 1 제외)
    axes[0].plot(epochs[1:], train_loss[1:], 'o-', color='#2196F3', linewidth=2, markersize=7, label='Train Loss')
    axes[0].set_title(f'Training Loss (Epoch 2~{len(epochs)})', fontsize=11)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_xticks(epochs[1:])
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, train_acc, 'o-', color='#4CAF50', linewidth=2, markersize=7, label='Train Acc')
    axes[1].plot(epochs, valid_acc, 's--', color='#FF9800', linewidth=2, markersize=7, label='Valid Acc')
    axes[1].set_title('Training / Validation Accuracy', fontsize=11)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_xticks(epochs)
    axes[1].set_ylim(50, 100)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(current_dir, "chart4_training.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 저장 완료: chart4_training.png")


if __name__ == "__main__":
    make_training_chart()
