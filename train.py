import os
import math
import json
import torch
import torch.nn as nn
import trackio
from PIL import Image
from torchvision import datasets
from dataclasses import dataclass
from torch.utils.data import DataLoader, random_split
from transformers import AutoImageProcessor, AutoModel, AutoConfig, get_cosine_schedule_with_warmup
from model import DinoV3Linear

DATA_DIR = "."
STROKE_CLASSES = ["backhand_back", "backhand_front", "forehand_back", "forehand_front", "serve_back", "serve_front"]
MODEL_NAME = "./dinov3-vitb16-pretrain-lvd1689m"
CHECKPOINT_DIR = "./weights"
BATCH_SIZE = 8
NUM_WORKERS = 0
EPOCHS = 30
LR = 5e-4
WEIGHT_DECAY = 1e-4
WARMUP_RATIO = 0.05
EVAL_EVERY_STEPS = 100


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class StrokeDataset(datasets.ImageFolder):
        def find_classes(self, directory):
            classes = [c for c in STROKE_CLASSES if os.path.isdir(os.path.join(directory, c))]
            return classes, {c: i for i, c in enumerate(classes)}

    full_dataset = StrokeDataset(root=DATA_DIR)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    num_classes = len(full_dataset.classes)
    print(f"Classes: {full_dataset.classes}")

    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    backbone = AutoModel.from_pretrained(MODEL_NAME)
    image_processor_config = json.loads(image_processor.to_json_string())
    backbone_config = json.loads(AutoConfig.from_pretrained(MODEL_NAME).to_json_string())

    model = DinoV3Linear(backbone, num_classes, freeze_backbone=True).to(device)

    @dataclass
    class Collator:
        processor: AutoImageProcessor

        def __call__(self, batch):
            images, labels = zip(*batch)
            rgb_images = [img.convert("RGB") if isinstance(img, Image.Image) else img for img in images]
            inputs = self.processor(images=rgb_images, return_tensors="pt")
            return {"pixel_values": inputs["pixel_values"], "labels": torch.tensor(labels, dtype=torch.long)}

    collate_fn = Collator(image_processor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * math.ceil(len(train_loader))
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    def evaluate():
        model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                logits = model(pixel_values)
                loss = criterion(logits, labels)
                loss_sum += loss.item() * labels.size(0)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return {"val_loss": loss_sum / max(total, 1), "val_acc": correct / max(total, 1)}

    best_acc = 0.0
    global_step = 0
    trackio.init(project="dinov3", config={"epochs": EPOCHS, "learning_rate": LR, "batch_size": BATCH_SIZE})

    for epoch in range(1, EPOCHS + 1):
        model.train()
        model.backbone.eval()
        running_loss = 0.0

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(pixel_values)
            loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()
            global_step += 1

            if global_step % EVAL_EVERY_STEPS == 0:
                metrics = evaluate()
                print(f"[epoch {epoch} | step {global_step}] train_loss={running_loss / EVAL_EVERY_STEPS:.4f} val_loss={metrics['val_loss']:.4f} val_acc={metrics['val_acc']*100:.2f}%")
                running_loss = 0.0

                if metrics["val_acc"] > best_acc:
                    best_acc = metrics["val_acc"]
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "config": {
                            "model_name": MODEL_NAME,
                            "classes": full_dataset.classes,
                            "backbone": backbone_config,
                            "image_processor": image_processor_config,
                            "freeze_backbone": True,
                        },
                        "step": global_step,
                        "epoch": epoch,
                    }, os.path.join(CHECKPOINT_DIR, "model_best.pt"))

        metrics = evaluate()
        trackio.log({"epoch": epoch, "val_loss": metrics["val_loss"], "val_acc": best_acc})
        print(f"END EPOCH {epoch}: val_loss={metrics['val_loss']:.4f} val_acc={metrics['val_acc']*100:.2f}% (best_acc={best_acc*100:.2f}%)")

    trackio.finish()


if __name__ == "__main__":
    main()
