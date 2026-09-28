import torch
from tqdm import tqdm

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total = 0
    progress_bar = tqdm(train_loader, desc="Training")
    
    for anchor, positive, negative in progress_bar:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        
        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)
        
        loss = criterion(anchor_out, positive_out, negative_out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * anchor.size(0)
        total += anchor.size(0)
        
        progress_bar.set_postfix({
            'loss': running_loss / total
        })
    
    epoch_loss = running_loss / total
    return epoch_loss, 0.0 # Return 0.0 for accuracy to maintain signature compatibility

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating")
        for anchor, positive, negative in progress_bar:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)
            
            loss = criterion(anchor_out, positive_out, negative_out)
            running_loss += loss.item() * anchor.size(0)
            total += anchor.size(0)
            
            progress_bar.set_postfix({
                'loss': running_loss / total
            })
    
    epoch_loss = running_loss / total
    return epoch_loss, 0.0
