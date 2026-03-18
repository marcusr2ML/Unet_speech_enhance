import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from generate_data import GenerateData


from unet import UNet   # <-- your UNet file

PRETRAINED_PATH = "unet_vocal_SR48k_mask_2.pth"  #loaded below

# ============================================================
# ROOT to data
# ============================================================


ROOT = "/home/marc/Downloads/train_music/test/"

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    val = 98304
    dataset = GenerateData(ROOT,WINDOW_SIZE=val,WINDOW_HOP=int(.25*val))
    loader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet(in_channels=2, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.L1Loss()
    
    if os.path.exists(PRETRAINED_PATH):
        print(f"Loading pre-trained model from {PRETRAINED_PATH}")
        model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=device))
    else:
        print("No pre-trained model found, starting from scratch.")

    from tqdm import tqdm
    for epoch in range(10):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for xb, yb in pbar:
            xb = xb.to(device)  # shape: (B, 2, F, T)
            yb = yb.to(device)
    
            if xb.size(2) == 0 or xb.size(3) == 0:
                continue  # skip empty windows
    
            pred = model(xb)
    
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
    
        print(f"Epoch {epoch+1} Complete: avg_loss={total_loss/len(loader):.5f}")
        torch.save(model.state_dict(), PRETRAINED_PATH)
        print("Model saved!")
if __name__ == "__main__":
    main()
