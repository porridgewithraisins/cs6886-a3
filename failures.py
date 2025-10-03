import torch
import numpy as np
import wandb
import cv2
from main import MobileNetV2_CIFAR, get_loaders, device

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
STD = torch.tensor([0.2470, 0.2430, 0.2610])


def unnormalize(img):
    if torch.is_tensor(img):
        img = img.cpu()
        img = img * STD[:, None, None] + MEAN[:, None, None]
        img = (img * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return img


def text_to_image(text, font_size=20, size=(64, 64)):
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, font_size / 40.0, 1)
    org = ((size[0] - w) // 2, (size[1] + h) // 2)
    cv2.putText(img, text, org, font, font_size / 40.0, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def log_confusion_matrix_with_images(x_test, y_test, y_pred, class_names):
    num_classes = len(class_names)

    y_test = np.array(y_test).astype(int)
    y_pred = np.array(y_pred).astype(int)

    correct_counts = {i: 0 for i in range(num_classes)}
    total_counts = {i: 0 for i in range(num_classes)}

    misclassified = {
        (i, j): [] for i in range(num_classes) for j in range(num_classes) if i != j
    }
    misclassified_counts = {key: 0 for key in misclassified}

    for img, true, pred in zip(x_test, y_test, y_pred):
        total_counts[true] += 1
        if true == pred:
            correct_counts[true] += 1
        else:
            if len(misclassified[(true, pred)]) == 0:
                misclassified[(true, pred)].append(wandb.Image(unnormalize(img)))
            misclassified_counts[(true, pred)] += 1

    table_data = []
    for true_label in range(num_classes):
        row = [class_names[true_label]]
        for pred_label in range(num_classes):
            if true_label == pred_label:
                total = total_counts[true_label]
                correct = correct_counts[true_label]
                fraction_text = f"{correct}/{total}" if total > 0 else "N/A"
                cell = [wandb.Image(text_to_image(fraction_text))]
            else:
                images = misclassified.get((true_label, pred_label), [])
                count = misclassified_counts.get((true_label, pred_label), 0)
                count_text_image = wandb.Image(text_to_image(str(count)))
                cell = images + [count_text_image] if images else [count_text_image]
            row.append(cell)
        table_data.append(row)

    table = wandb.Table(columns=["True \\ Pred"] + class_names, data=table_data)
    wandb.log({"Confusion Matrix": table})


def main():
    wandb.init(name="failure_modes")

    _, _, test_loader = get_loaders(batch_size=128)

    model = MobileNetV2_CIFAR(num_classes=10).to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    all_imgs, all_labels, all_preds = [], [], []
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x.to(device))
            pred = out.argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += y.size(0)

            all_imgs.extend(x)
            all_labels.extend(y)
            all_preds.extend(pred)

    acc = 100.0 * correct / total
    print(f"Final Test Accuracy: {acc:.2f}%")
    wandb.log({"test_accuracy": acc})

    log_confusion_matrix_with_images(all_imgs, all_labels, all_preds, CLASSES)

    wandb.finish()


if __name__ == "__main__":
    main()
