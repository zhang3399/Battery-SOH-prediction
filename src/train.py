import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


def train_model(model, train_loader, val_loader, config):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    best_loss = np.inf
    train_losses = []
    val_losses = []

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 验证
        val_loss = evaluate(model, val_loader, criterion)
        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss)

        # 早停机制
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'models/best_model.pth')

        print(f'Epoch {epoch + 1}: Train Loss {train_losses[-1]:.4f}, Val Loss {val_losses[-1]:.4f}')

    # 绘制训练曲线
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.legend()
    plt.savefig('results/training_curve.png')

    return model, {
        'best_val_loss': best_loss,
        'train_losses': train_losses,
        'val_losses': val_losses
    }


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(loader)
