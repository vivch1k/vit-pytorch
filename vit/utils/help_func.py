import matplotlib.pyplot as plt
import seaborn as sns
import torch
import random


def plot_curve(results: dict):
    range_epochs = list(range(len(results["train_loss"])))
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.lineplot(x=range_epochs,
                 y=results["train_loss"], 
                 label="train_loss", 
                 color="red")
    sns.lineplot(x=range_epochs, 
                 y=results["val_loss"],
                 label="val_loss",
                 color="blue")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    sns.lineplot(x=range_epochs,
                 y=results["train_acc"],
                 label="train_acc",
                 color="red")
    sns.lineplot(x=range_epochs,
                 y=results["val_acc"],
                 label="val_acc",
                 color="blue")
    plt.grid(True)


def set_seeds(seed: int=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    
    