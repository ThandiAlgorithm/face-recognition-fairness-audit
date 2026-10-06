import argparse
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config
from model_arch import MultiHeadResNet

TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class FaceCSV(Dataset):
    def __init__(self, csv_path, ds):
        self.df = pd.read_csv(csv_path)
        self.ds = ds
        self.age_idx = {a: i for i, a in enumerate(config.AGE_BUCKETS)}
        self.gen_idx = {g: i for i, g in enumerate(config.GENDER)}
        self.race_idx = {r: i for i, r in enumerate(ds["race"])} if ds["has_race"] else {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["image_path"]).convert("RGB")
        x = TF(img)
        y = {
            "age": torch.tensor(self.age_idx[row["age"]]),
            "gender": torch.tensor(self.gen_idx[row["gender"]]),
        }
        if self.ds["has_race"]:
            y["race"] = torch.tensor(self.race_idx[row["race"]])
        return x, y


def train(name, csv_path, epochs, batch, lr):
    ds = config.DATASETS[name]
    data = FaceCSV(csv_path, ds)
    loader = DataLoader(data, batch_size=batch, shuffle=True, num_workers=2)
    model = MultiHeadResNet(
        n_age=len(config.AGE_BUCKETS),
        n_race=len(ds["race"]) if ds["has_race"] else 0,
        with_race=ds["has_race"],
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            x = x.to(device)
            opt.zero_grad()
            out = model(x)
            loss = ce(out["age"], y["age"].to(device)) + ce(out["gender"], y["gender"].to(device))
            if ds["has_race"]:
                loss = loss + ce(out["race"], y["race"].to(device))
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {ep+1}/{epochs} loss {total/len(loader):.4f}")

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    out_path = os.path.join(config.MODEL_DIR, name + ".pth")
    torch.save(model.state_dict(), out_path)
    print("saved", out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(config.DATASETS.keys()))
    p.add_argument("--csv", required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    a = p.parse_args()
    train(a.dataset, a.csv, a.epochs, a.batch, a.lr)
