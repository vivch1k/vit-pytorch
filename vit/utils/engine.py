import torch
from typing import Tuple, Dict, List
from collections import defaultdict

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    model.train()
    
    train_loss, train_acc = 0, 0
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        
        loss = loss_fn(logits, y)
        train_loss += loss.item()
        
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        
        probs = torch.softmax(logits, 1)
        preds = torch.argmax(probs, 1)
        train_acc += (preds == y).sum().item() / len(y)
        
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device: torch.device) -> Tuple[float, float]:
    model.eval()
    
    test_loss, test_acc = 0, 0
    
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            logits = model(X)
            
            loss = loss_fn(logits, y)
            test_loss += loss.item()
        
            probs = torch.softmax(logits, 1)
            preds = torch.argmax(probs, 1)
            test_acc += (preds == y).sum().item() / len(y)
        
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    
    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          epochs: int) -> Dict[str, List]:
    
    results = defaultdict(list)
    
    for epoch in range(epochs):
        train_loss, train_acc = train_step(model,
                                           train_dataloader,
                                           loss_fn,
                                           optimizer,
                                           device)
        
        test_loss, test_acc = test_step(model,
                                        test_dataloader,
                                        loss_fn,
                                        device)

        print(f"Epoch: {epoch + 1}/{epochs} | "
              f"train_loss: {train_loss:.4f} | "
              f"train_acc: {train_acc:.4f} | "
              f"test_loss: {test_loss:.4f} | "
              f"test_acc: {test_acc:.4f} | ")
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
    return results