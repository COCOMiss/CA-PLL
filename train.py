import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import time
import os
import sys
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGIR2026_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SIGIR2026_DIR)
from utils import init_logging, get_logger
from dataset import POIDataProcessor, POIProcessingConfig, CheckinSequenceDataset, seq_collate_fn
from model import TrajPOITransformer


DEFAULT_DATASET = 'tokyo'
DEFAULT_EXP_NAME = 'ca_pll'


logger = get_logger("CA-PLL")
TRAIN = True
TEST = True


def get_dataset_paths(dataset_name):
    data_dir = os.path.join(PROJECT_ROOT, 'dataset', dataset_name)
    return {
        'checkin_file': os.path.join(data_dir, 'filtered_checkin_data.csv'),
        'poi_file': os.path.join(data_dir, 'poi.csv'),
        'dist_file': os.path.join(data_dir, 'category_time_distribution_P_Category_given_Time.csv'),
    }

class TrainingConfig:
    checkin_file = None
    poi_file = None
    dist_file = None
    radius = 200
    max_candidates = 20
    max_seq_len = 20
    batch_size = 16
    epochs = 100
    learning_rate = 5e-4
    weight_decay = 1e-4
    test_size = 0.2
    seed = 42
    patience = 5
    embed_dim = 64
    curriculum_epoch = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    warmup_epochs = 0
    rampup_epochs = 5
    final_hardnegative_ratio = 0.6
    result_file = None 
    
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, data_loader, device):
    """
    Evaluation Function
    Args:
        model: TrajPOITransformer
        data_loader: DataLoader
        device: torch.device
        
    Returns:
        loss, acc_1, acc_5, acc_cat, acc_main_cat
    """
    model.eval()
    correct_1 = 0
    correct_5 = 0
    correct_cat = 0
    correct_main_cat = 0
    total_valid_steps = 0 
    total_missing_main_cats = 0
    
    logger.info(f"Start Evaluation on {len(data_loader.dataset)} trajectories...")
    
    with torch.no_grad():
        for batch in data_loader:
            for k, v in batch.items():
                batch[k] = v.to(device)
            
            # Predict scores [B, S, K]
            scores = model.predict(batch)
            
            # Flatten Mask
            mask = batch['seq_mask'].bool().view(-1)
            
            # Flatten Scores: [N_valid, K]
            scores_flat = scores.view(-1, scores.size(-1))[mask]
            
            # Flatten Labels (indices in candidate set): [N_valid]
            labels_flat = batch['label_pos'].view(-1)[mask]
            
            if len(labels_flat) == 0:
                continue
            
            # --- 1. POI Accuracy ---
            _, top5_indices = torch.topk(scores_flat, k=5, dim=1) # [N, 5]
            top1_preds = top5_indices[:, 0] # [N]
            
            correct_1 += (top1_preds == labels_flat).sum().item()
            correct_5 += (top5_indices == labels_flat.unsqueeze(1)).any(dim=1).sum().item()
            
            # --- 2. Category Accuracy ---
            # cand_cat_ids: [B, S, K] -> [N_valid, K]
            cand_cats_flat = batch['cand_cat_ids'].view(-1, batch['cand_cat_ids'].size(-1))[mask]
            
            # gather index: [N, 1]
            pred_cats = torch.gather(cand_cats_flat, 1, top1_preds.unsqueeze(1)).squeeze(1)
            gt_cats = torch.gather(cand_cats_flat, 1, labels_flat.unsqueeze(1)).squeeze(1)
            
            valid_cat_mask = (gt_cats != 0)
            if valid_cat_mask.any():
                correct_cat += (pred_cats[valid_cat_mask] == gt_cats[valid_cat_mask]).sum().item()

            cand_main_cats_flat = batch['cand_main_cat_ids'].view(-1, batch['cand_main_cat_ids'].size(-1))[mask]
            
            pred_main_cats = torch.gather(cand_main_cats_flat, 1, top1_preds.unsqueeze(1)).squeeze(1)
            gt_main_cats = torch.gather(cand_main_cats_flat, 1, labels_flat.unsqueeze(1)).squeeze(1)
            
            valid_main_mask = (gt_main_cats != -1) & (pred_main_cats != -1)
            missing_mask = (gt_main_cats == -1)
            total_missing_main_cats += missing_mask.sum().item()
            if total_missing_main_cats > 0:
                logger.error(f"Total missing main cats: {total_missing_main_cats}")
            if valid_main_mask.any():
                correct_main_cat += (pred_main_cats[valid_main_mask] == gt_main_cats[valid_main_mask]).sum().item()
            
            total_valid_steps += len(labels_flat)
            
    if total_valid_steps == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
        
    acc_1 = correct_1 / total_valid_steps
    acc_5 = correct_5 / total_valid_steps
    acc_cat = correct_cat / total_valid_steps
    acc_main_cat = correct_main_cat / total_valid_steps
    
    return 0.0, acc_1, acc_5, acc_cat, acc_main_cat
   


class EarlyStopping:
    def __init__(self, patience=5, path=None):
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc, model):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        logger.info(f'Validation Acc improved ({self.best_score:.4f} --> {val_acc:.4f}). Saving model...')
        torch.save(model.state_dict(), self.path)


def get_epoch_settings(epoch, conf):
    if epoch < conf.warmup_epochs:
        return [0.0, 1.0]
    elif epoch < (conf.warmup_epochs + conf.rampup_epochs):
        progress = (epoch - conf.warmup_epochs) / conf.rampup_epochs
        r_random = conf.final_hardnegative_ratio * progress
        return [r_random, 1-r_random]
    else:
        random_ratio = getattr(conf, 'final_hardnegative_ratio', 0.6)
        return  [random_ratio, 1-random_ratio]   


def main():
    parser = argparse.ArgumentParser(description="TrajPOI Grid Search")
    parser.add_argument('--dataset', type=str, default="tokyo",
                        help='Dataset name (e.g. gowalla_filtered, ny_filterPOI, ny, tokyo)')
    parser.add_argument('--exp_name', type=str, default=DEFAULT_EXP_NAME,
                        help='Experiment subfolder name under result/<dataset>/')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--log_file', type=str, default="result/tokyo/negative_latest2/training.log", help='Path to save log (default: result/<dataset>/<exp_name>/training.log)')
    parser.add_argument('--save_model_name', type=str, default="result/tokyo/negative_latest2/best_model.pth", help='Model save path (default: result/<dataset>/<exp_name>/best_model.pth)')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Warmup epochs')
    parser.add_argument('--rampup_epochs', type=int, default=5, help='Rampup epochs')
    parser.add_argument('--final_hardnegative_ratio', type=float, default=1.0, help='Final hard negative ratio in refinement stage')
    parser.add_argument('--semantic_thresh', type=float, default=0.0005, help='Threshold for semantic hard negative sampling')
    parser.add_argument('--radius_size', type=int, default=200, help='Radius size')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    args = parser.parse_args()
    

    result_dir = os.path.join(PROJECT_ROOT, 'result', args.dataset, args.exp_name)
    log_file = args.log_file or os.path.join(result_dir, 'training.log')
    save_path = args.save_model_name or os.path.join(result_dir, 'best_model.pth')
    if args.log_file and not os.path.isabs(args.log_file):
        log_file = os.path.join(PROJECT_ROOT, args.log_file)
    if args.save_model_name and not os.path.isabs(args.save_model_name):
        save_path = os.path.join(PROJECT_ROOT, args.save_model_name)
    result_dir = os.path.dirname(log_file)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    global logger
    init_logging(log_file=log_file, force=True)
    logger = get_logger("Training")
    logger.info(f"Arguments: {args}")
    logger.info(f"Result dir: {result_dir}")
    logger.info(f"Log file: {log_file}")


    conf = TrainingConfig()
    conf.learning_rate = args.learning_rate
    paths = get_dataset_paths(args.dataset)
    conf.checkin_file = paths['checkin_file']
    conf.poi_file = paths['poi_file']
    conf.dist_file = paths['dist_file']
    conf.result_file = os.path.join(result_dir, 'final_result.txt')
    conf.epochs = args.epochs
    conf.batch_size = args.batch_size
    conf.warmup_epochs = args.warmup_epochs
    conf.rampup_epochs = args.rampup_epochs
    conf.final_hardnegative_ratio = args.final_hardnegative_ratio
    conf.radius = args.radius_size

    set_seed(conf.seed)
    logger.info(f"Running on device: {conf.device}")
    
    # --- Phase 1: Data ---
    logger.info("[Phase 1] Initializing Data Processor...")
    if not os.path.exists(conf.checkin_file):
        logger.error(f"File not found: {conf.checkin_file}")
        return
    logger.info(f"Radius size: {conf.radius}")
    poi_config = POIProcessingConfig(
        checkin_file=conf.checkin_file,
        poi_file=conf.poi_file,
        dist_file=conf.dist_file,
        radius=conf.radius,
        max_candidates=conf.max_candidates,
        semantic_sample_thresh=args.semantic_thresh,
        device=conf.device
    )
    
    processor = POIDataProcessor(poi_config)
    
    full_dataset = CheckinSequenceDataset(processor, max_seq_len=conf.max_seq_len, mode='train')
    logger.info(f"Total trajectories: {len(full_dataset)}")
    indices = list(range(len(full_dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=conf.test_size, random_state=conf.seed)
    

    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)
    
    logger.info(f"Train trajectories: {len(train_dataset)} | Test trajectories: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, collate_fn=seq_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, collate_fn=seq_collate_fn)
    
    if isinstance(train_loader.dataset, Subset):
        raw_train_dataset = train_loader.dataset.dataset
    else:
        raw_train_dataset = train_loader.dataset
    
    # --- Phase 2: Model ---
    logger.info("[Phase 2] Initializing Model...")
    class ModelConfig:
        dataset_name = args.dataset
        user2idx = processor.user2idx
        venue_id2idx = processor.venue_id2idx
        cat2idx = processor.cat2idx
        embed_dim = conf.embed_dim
        num_main_cats = processor.num_main_cats
        num_cats = processor.num_cats
    
    model = TrajPOITransformer(ModelConfig)
    model.to(conf.device)

    optimizer = optim.AdamW(model.parameters(), lr=conf.learning_rate, weight_decay=conf.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=conf.patience, path=save_path)
    # --- Phase 3: Training ---
    logger.info("[Phase 3] Start Training (CLPL Mode)...")
    
    save_path = early_stopping.path
    if TRAIN:
        for epoch in range(conf.epochs):
            curr_neg_strategy = get_epoch_settings(epoch, conf)   
            # curr_neg_strategy=[1.0,0.0]
            raw_train_dataset.set_negative_strategy(*curr_neg_strategy)
            logger.info(f"Epoch {epoch+1} Curriculum:")
            logger.info(f"  - Negative Strategy: Random={curr_neg_strategy[0]:.2f}, Sematic={curr_neg_strategy[1]:.2f}")
            
            if epoch > 0 and os.path.exists(save_path):
                logger.info(f">>> Reloading best model from {save_path}")
                model.load_state_dict(torch.load(save_path))
                
            model.train()
            start_time = time.time()
            total_loss = 0
            
            for batch_idx, batch in enumerate(train_loader):
                for k, v in batch.items(): batch[k] = v.to(conf.device)
                
                optimizer.zero_grad()
                
                cand_scores, neg_scores, user_vector = model(batch)
   
                
                loss = model.compute_weighted_clpl_loss(
                    cand_scores, batch['cand_mask'],
                    neg_scores, batch['seq_mask'],
                    user_vector=user_vector,
                    cand_cat_ids=batch['cand_cat_ids']
                )
                
 
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if (batch_idx + 1) % 50 == 0:
                    logger.info(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                                f"Loss: {loss.item():.4f})")
                
            
            avg_loss = total_loss / len(train_loader)
            epoch_time = time.time() - start_time
            
            # Validation
            logger.info(f"--> Validating Epoch {epoch+1}...")
            _, val_acc1, val_acc5, val_acc_cat, val_acc_main_cat = evaluate(model, test_loader,conf.device)
            
            scheduler.step(val_acc1)
            current_lr = optimizer.param_groups[0]['lr']
            
            logger.info(f"="*60)
            logger.info(f"Epoch {epoch+1} Summary ({epoch_time:.1f}s):")
            logger.info(f"Train Loss: {avg_loss:.4f}")
            logger.info(f"Val Acc@1: {val_acc1:.4f} ({val_acc1*100:.2f}%)")
            logger.info(f"Val Acc@5: {val_acc5:.4f} ({val_acc5*100:.2f}%)")
            logger.info(f"Val Acc@Cat: {val_acc_cat:.4f} ({val_acc_cat*100:.2f}%)")
            logger.info(f"Val Acc@MainCat: {val_acc_main_cat:.4f} ({val_acc_main_cat*100:.2f}%)")
            logger.info(f"Current LR: {current_lr}")
            logger.info(f"="*60)
            
            early_stopping(val_acc1, model)
            if early_stopping.early_stop:
                logger.info("Early stopping triggered.")
                break
        
        
    if TEST:  
    # --- Phase 4: Final Test ---
        logger.info("\n[Phase 4] Final Testing...")
        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path))
            
        _, final_acc1, final_acc5, final_acc_cat, final_acc_main_cat = evaluate(model, test_loader,conf.device)
        logger.info(f"Final Result -> Acc@1: {final_acc1:.4f}, Acc@5: {final_acc5:.4f}, Acc@Cat: {final_acc_cat:.4f}, Acc@MainCat: {final_acc_main_cat:.4f}")
        exp_id = args.exp_name
        result_summary_path = log_file.replace('.log', '_result.txt')
        logger.info(f"Saving summary to {result_summary_path}")    
        with open(result_summary_path, 'w') as f:

            f.write(f"Experiment ID: {exp_id}\n") 
            f.write("-" * 30 + "\n")
            f.write(f"Semantic Thresh:    {args.semantic_thresh}\n")
            f.write(f"Final HN Ratio:     {args.final_hardnegative_ratio}\n")
            f.write(f"Rampup Epochs:      {args.rampup_epochs}\n")
            f.write(f"Warmup Epochs:      {args.warmup_epochs}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Acc@1: {final_acc1:.4f}\n")
            f.write(f"Acc@5: {final_acc5:.4f}\n")
            f.write(f"Acc@Cat: {final_acc_cat:.4f}\n")
            f.write(f"Acc@MainCat: {final_acc_main_cat:.4f}\n")
            
if __name__ == "__main__":
    main()
