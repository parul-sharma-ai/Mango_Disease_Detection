from __future__ import absolute_import, division, print_function

# Local files
from models import BaseFeatureExtractor
from utils import create_logger

# Libraries
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datasets.mango import MangoImageFolder, build_transform

import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
import pprint
import os


# ----------------------------
# Parse command-line arguments
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Mango Disease Classification Training')
    parser.add_argument('--modelDir', type=str,
                        default='/content/drive/MyDrive/Mango Disease Classifier/weights',
                        help='Directory to save weights/checkpoints')
    parser.add_argument('--logDir', type=str,
                        default='/content/drive/MyDrive/Mango Disease Classifier/logs',
                        help='Directory to save TensorBoard logs')
    parser.add_argument('--dataDir', type=str,
                        default='/content/drive/MyDrive/Mango Disease Classifier/Dataset',
                        help='Path to dataset')
    parser.add_argument('--model_type', type=str, default='conv_trans', help='Model type')
    parser.add_argument('--split_ratio', type=float, default=0.9, help='Train-validation split')
    parser.add_argument('--n_class', type=int, default=5, help='Number of classes')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--max_lr', type=float, default=3e-5, help='Maximum learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    return parser.parse_args()


# ----------------------------
# Main training function
# ----------------------------
def main():
    opts = parse_args()

    print(f"\n=== Training {opts.model_type} ===")

    # Create directories
    os.makedirs(os.path.join(opts.modelDir, opts.model_type), exist_ok=True)
    os.makedirs(opts.logDir, exist_ok=True)

    logger, tb_log_dir = create_logger(opts)
    logger.info(pprint.pformat(opts))

    # CUDNN settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    # ----------------------------
    # Model
    # ----------------------------
    model = BaseFeatureExtractor(config=opts.model_type, n_class=opts.n_class).cuda()

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # ----------------------------
    # Dataset
    # ----------------------------
    trans = build_transform(input_size=224)
    data = MangoImageFolder(opts.dataDir, sampling=10, transform=trans)

    torch.manual_seed(1)
    indices = torch.randperm(len(data)).tolist()
    split_idx = int((1 - opts.split_ratio) * len(data))

    trainset = torch.utils.data.Subset(data, indices[:-split_idx])
    valset = torch.utils.data.Subset(data, indices[-split_idx:])

    train_loader = DataLoader(trainset, batch_size=opts.batch_size, shuffle=True, pin_memory=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(valset, batch_size=opts.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    # ----------------------------
    # Loss, optimizer, scheduler
    # ----------------------------
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.max_lr, weight_decay=opts.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, opts.max_lr, epochs=opts.epochs, steps_per_epoch=len(train_loader))

    # ----------------------------
    # Resume checkpoint if exists
    # ----------------------------
    last_ckpt_path = os.path.join(opts.modelDir, opts.model_type, 'last_model.pt')
    start_epoch = 0
    best_val_acc = 0.0
    if os.path.exists(last_ckpt_path):
        print("Resuming from last checkpoint...")
        ckpt = torch.load(last_ckpt_path)
        model.load_state_dict(ckpt['model_dict'])
        optimizer.load_state_dict(ckpt['optim_dict'])
        sched.load_state_dict(ckpt['schd_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('accuracy', 0.0)
        print(f"Resuming from epoch {start_epoch}, best validation accuracy: {best_val_acc:.4f}")

    # ----------------------------
    # Training loop
    # ----------------------------
    for epoch in range(start_epoch, opts.epochs):
        model.train()
        total_loss = 0.0
        for data_batch, target_batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            data_batch, target_batch = data_batch.cuda(), target_batch.cuda()
            optimizer.zero_grad()
            pred = model(data_batch)
            loss = loss_fn(pred, target_batch)
            loss.backward()
            optimizer.step()
            sched.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        logger.info(f"Train Epoch {epoch}: Loss = {train_loss:.4f}")
        writer_dict['writer'].add_scalar('train_loss', train_loss, writer_dict['train_global_steps'])
        writer_dict['train_global_steps'] += 1

        # ----------------------------
        # Validation
        # ----------------------------
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for data_batch, target_batch in val_loader:
                data_batch, target_batch = data_batch.cuda(), target_batch.cuda()
                pred = model(data_batch)
                loss = loss_fn(pred, target_batch)
                val_loss += loss.item()

                _, predicted = torch.max(pred.data, 1)
                all_preds.append(predicted.cpu().numpy())
                all_targets.append(target_batch.cpu().numpy())

        val_loss /= len(val_loader)
        all_preds_np = np.concatenate(all_preds)
        all_targets_np = np.concatenate(all_targets)
        val_acc = accuracy_score(all_targets_np, all_preds_np)

        print(f"Epoch {epoch} done. Validation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}")

        # ----------------------------
        # Save last checkpoint
        # ----------------------------
        last_ckpt = {
            'model_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'schd_dict': sched.state_dict(),
            'epoch': epoch,
            'accuracy': best_val_acc
        }
        torch.save(last_ckpt, last_ckpt_path)

        # ----------------------------
        # Save best model
        # ----------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt = {
                'model_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                'schd_dict': sched.state_dict(),
                'epoch': epoch,
                'accuracy': best_val_acc,
                'val_loss': val_loss
            }
            torch.save(best_ckpt, os.path.join(opts.modelDir, opts.model_type, 'best_model.pt'))
            logger.info(f"Best model saved at epoch {epoch} with accuracy {best_val_acc:.4f}")

    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
