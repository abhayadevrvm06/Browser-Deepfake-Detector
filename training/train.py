from sklearn.metrics import confusion_matrix, classification_report

test_dir = "data/real_vs_fake/real-vs-fake/test"
test_ds = datasets.ImageFolder(test_dir, transform=val_transform)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

model.load_state_dict(torch.load("best_model.pt"))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        preds = outputs.argmax(1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

test_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
print(f"Test Accuracy: {test_acc:.4f}")
print(confusion_matrix(all_labels, all_preds))
print(classification_report(all_labels, all_preds, target_names=["fake", "real"]))