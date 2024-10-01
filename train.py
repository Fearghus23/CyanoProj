# train.py
import torch
from torch.utils.data import DataLoader
from dataset import CyanobacteriaDataset
from model import create_model
import torchvision.transforms as T
import torch.optim as optim

def collate_fn(batch):
    return tuple(zip(*batch))

def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = CyanobacteriaDataset('annotations/annotations.json', transforms=T.ToTensor())
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    num_classes = len(dataset.classes)
    model = create_model(num_classes)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {losses.item()}')
    torch.save(model.state_dict(), 'models/efficientdet.pth')

if __name__ == '__main__':
    train()
