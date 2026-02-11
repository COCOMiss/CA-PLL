
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import BallTree
import os
from utils import get_logger
import pandas as pd
import torch
from collections import Counter
import re
import unicodedata


logger = get_logger("Dataset")




class POIProcessingConfig:
    def __init__(self, checkin_file, poi_file, dist_file, radius=200, max_candidates=50, semantic_sample_thresh=0.05, device=None):
        self.checkin_file = checkin_file
        self.poi_file = poi_file
        self.dist_file = dist_file
        self.radius = radius 
        self.max_candidates = max_candidates
        self.semantic_sample_thresh = semantic_sample_thresh
        self.device = device
        logger.info(f"Config: Radius={radius}m, Max Cand={max_candidates}, Sem Thresh={semantic_sample_thresh}")

class POIDataProcessor:
    def __init__(self, config):
        self.config = config
        
        self.venue_id2idx = {} 
        self.idx2venue_id = {}
        self.cat2idx = {}
        self.idx2cat = {}
        self.user2idx = {} 

        self.poi_coords = None      
        self.poi_cat_indices = None 
        self.cat_time_probs = None
        self.main_cat_probs = None
        self.df_checkin = None
        self.ball_tree = None
        self.num_pois = 0 
        
        data_dir = os.path.dirname(config.poi_file)
        self.cat_list_path = os.path.join(data_dir, 'category_list.txt') 
        self.category_path = os.path.join(data_dir, 'main_category.csv')
        self._load_and_build()
        self.main_cat_mapping = self.get_main_cat_mapping_tensor(self.config.device)

    def get_main_cat_mapping_tensor(self, device):
        if not self.idx2cat:
            raise ValueError("idx2cat not built")

        if not os.path.exists(self.category_path):
            raise FileNotFoundError(f"can not find CSV: {self.category_path}")

        def normalize(s):
            if not isinstance(s, str):
                s = str(s)
            s = unicodedata.normalize('NFKD', s)
            s = s.encode('ascii', 'ignore').decode('ascii')
            s = s.lower()
            s = re.sub(r'[^a-z0-9]', ' ', s)

            s = ' '.join(s.split())
            return s

        try:
            df = pd.read_csv(self.category_path, encoding='utf-8')
        except:
            df = pd.read_csv(self.category_path, encoding='latin-1')
        
        unique_main_names = sorted(df['main'].unique().tolist())
        main_name_to_id = {name: i for i, name in enumerate(unique_main_names)}
        
        norm_sub_to_main_id = {}
        for _, row in df.iterrows():
            norm_name = normalize(row['original'])
            if norm_name:
                norm_sub_to_main_id[norm_name] = main_name_to_id[row['main']]

        num_small_cats = len(self.idx2cat)
        mapping_array = torch.full((num_small_cats,), -1, dtype=torch.long)

        for cat_id, cat_name in self.idx2cat.items():
            norm_cat_name = normalize(cat_name)
            
            if norm_cat_name in norm_sub_to_main_id:
                mapping_array[cat_id] = norm_sub_to_main_id[norm_cat_name]
            else:
                words = norm_cat_name.split()
                found = False
                if len(words) > 1:
                    short_name = ' '.join(words[:2])
                    if short_name in norm_sub_to_main_id:
                        mapping_array[cat_id] = norm_sub_to_main_id[short_name]
                        found = True
                
                if not found:
                    logger.error(f"Category '{cat_name}' (Normalized: '{norm_cat_name}') still not found.")
        
        self.num_main_cats = len(unique_main_names)
        self.main_cat_probs = np.zeros((self.num_main_cats, 24), dtype=np.float32)
        mapping_np = mapping_array.cpu().numpy()
        

        valid_mask = mapping_np != -1
        valid_small_indices = np.where(valid_mask)[0]
        valid_main_indices = mapping_np[valid_mask]
        
        np.add.at(self.main_cat_probs, valid_main_indices, self.cat_time_probs[valid_small_indices])
        
        col_sums = self.main_cat_probs.sum(axis=0, keepdims=True) + 1e-9
        self.main_cat_probs = self.main_cat_probs / col_sums
                
        return mapping_array.to(device)
       

    
    def _load_and_build(self):

        logger.info(f"Loading raw data...")
        df_poi = pd.read_csv(self.config.poi_file)
        df_dist = pd.read_csv(self.config.dist_file)
        df_checkin = pd.read_csv(self.config.checkin_file)
        df_checkin = df_checkin.rename(columns={ 'userid':'user_id'})

        if os.path.exists(self.cat_list_path):
            logger.info(f"Loading category list from {self.cat_list_path}")
            with open(self.cat_list_path, 'r') as f:
                cat_names = [line.strip() for line in f if line.strip()]
        else:
            logger.info("Building new category list from POI data...")
            cat_names = sorted(df_poi['category'].dropna().unique())
            with open(self.cat_list_path, 'w') as f:
                for c in cat_names: f.write(f"{c}\n")
 
        self.num_cats = len(cat_names)
        self.cat2idx = {cat: i for i, cat in enumerate(cat_names)}
        self.idx2cat = {i: cat for i, cat in enumerate(cat_names)}
        num_cats = len(self.cat2idx)

        self.venue_id2idx = {vid: i for i, vid in enumerate(df_poi['venue_id'])}
        self.idx2venue_id = {i: vid for i, vid in enumerate(df_poi['venue_id'])}
        self.num_pois = len(df_poi)
        
        unique_users = sorted(df_checkin['user_id'].unique())
        self.user2idx = {uid: i for i, uid in enumerate(unique_users)}
        
        logger.info(f"Mappings built. Users: {len(self.user2idx)}, POIs: {self.num_pois}, Cats: {num_cats}")

        self.poi_cat_indices = np.zeros(self.num_pois, dtype=int)
        unknown_cat_idx = 0 
        for i, row in df_poi.iterrows():
            cat = row['category']
            self.poi_cat_indices[i] = self.cat2idx.get(cat, unknown_cat_idx)
            
        self.poi_coords = np.radians(df_poi[['latitude', 'longitude']].values).astype(np.float32)
        self.cat_time_probs = np.zeros((num_cats, 24), dtype=np.float32)
        dist_map = df_dist.set_index('category')
        prob_cols = df_dist.columns[1:]
        for i, cat_name in enumerate(cat_names):
            if cat_name in dist_map.index:
                self.cat_time_probs[i] = dist_map.loc[cat_name, prob_cols].values.astype(np.float32)
            else:
                logger.error(f"Category '{cat_name}' not found in distribution file. using zeros.")

        logger.info("Processing Check-in dataframe...")
        df_checkin = df_checkin[df_checkin['venue_id'].isin(self.venue_id2idx)].copy()
        
        df_checkin['user_idx_mapped'] = df_checkin['user_id'].map(self.user2idx)
        df_checkin['venue_idx_mapped'] = df_checkin['venue_id'].map(self.venue_id2idx)
        
        df_checkin['local_datetime'] = pd.to_datetime(df_checkin['local_datetime'])
        df_checkin['hour'] = df_checkin['local_datetime'].dt.hour
        df_checkin['dayofweek'] = df_checkin['local_datetime'].dt.dayofweek
        base_slot = df_checkin['hour'] // 2
        is_weekend = (df_checkin['dayofweek'] >= 5).astype(int)
        df_checkin['time_slot'] = base_slot + (is_weekend * 12)
        
        self.df_checkin = df_checkin.reset_index(drop=True)

        logger.info("Building BallTree...")
        self.ball_tree = BallTree(self.poi_coords, metric='haversine')
        
        logger.info("POIDataProcessor Initialization Complete.")

    
    




class CheckinSequenceDataset(Dataset):
    def __init__(self, processor, max_seq_len=20, mode='train'):
        self.processor = processor
        self.config = processor.config
        self.mode = mode
        self.max_seq_len = max_seq_len
        
        self.radius_rad = self.config.radius / 6371000.0
        self.neg_strategy = [1.0, 0.0] 
        logger.info(f"Grouping trajectories by User and Date ({mode})...")
        df = self.processor.df_checkin.sort_values(
            by=['user_idx_mapped', 'date', 'local_datetime'],
            ascending=[True, True, True]
        )
   
        self.sorted_df = df.reset_index(drop=True)
        self.traj_groups = list(self.sorted_df.groupby(['user_idx_mapped', 'date']).indices.values())

        logger.info("Building User-Level GPS-Candidate Frequency Matrix...")
        all_true_pois = self.sorted_df['venue_idx_mapped'].values
        all_coords = self.processor.poi_coords[all_true_pois] 
        
        cands_list = self.processor.ball_tree.query_radius(all_coords, r=self.radius_rad)
        
        
        self.user_cand_freq = {}
        self.user_total_cands = {}
        
        user_grouped_indices = self.sorted_df.groupby('user_idx_mapped').indices
        for uid, indices in user_grouped_indices.items():
            all_cands_for_user = np.concatenate(cands_list[indices])
            self.user_cand_freq[uid] = Counter(all_cands_for_user)
            self.user_total_cands[uid] = len(indices) + 1e-9 

    def __len__(self):
        return len(self.traj_groups)
    
    def set_negative_strategy(self, random_ratio, semantic_ratio):
        
        total = random_ratio + semantic_ratio
        if total > 0:
            self.neg_strategy = [random_ratio/total, semantic_ratio/total]
        else:
            self.neg_strategy = [0.0, 1.0] # Fallback
 
    
    def __getitem__(self, idx):
        row_indices = self.traj_groups[idx]
        if len(row_indices) > self.max_seq_len:
            row_indices = row_indices[:self.max_seq_len]
        traj_df = self.sorted_df.iloc[row_indices]
        
        seq_len = len(traj_df)
        max_cand = self.config.max_candidates
        
        user_ids = traj_df['user_idx_mapped'].values.astype(np.int64)
        time_slots = traj_df['time_slot'].values.astype(np.int64)
        true_poi_indices = traj_df['venue_idx_mapped'].values
        true_coords = self.processor.poi_coords[true_poi_indices]
        
        # Noise
        noise_dist = np.sqrt(np.random.random(seq_len)) * 50.0 / 6371000.0
        noise_theta = np.random.random(seq_len) * 2 * np.pi
        delta_lat = noise_dist * np.cos(noise_theta)
        delta_lon = noise_dist * np.sin(noise_theta) / np.cos(true_coords[:, 0])
        center_coords = np.stack([true_coords[:, 0] + delta_lat, true_coords[:, 1] + delta_lon], axis=1).astype(np.float32)
        
        # Init Containers
        seq_cand_ids = np.zeros((seq_len, max_cand), dtype=int)
        seq_cand_cats = np.zeros((seq_len, max_cand), dtype=int)
        seq_cand_main_cats = np.zeros((seq_len, max_cand), dtype=int)
        
        seq_cand_probs = np.zeros((seq_len, max_cand), dtype=np.float32)
        seq_cand_main_probs = np.zeros((seq_len, max_cand), dtype=np.float32)
        
        seq_cand_mask = np.zeros((seq_len, max_cand), dtype=int)
        seq_cand_dists = np.zeros((seq_len, max_cand), dtype=np.float32)
        seq_cand_feats = np.zeros((seq_len, max_cand, 1), dtype=np.float32) 
        
        seq_neg_ids = np.zeros((seq_len, max_cand), dtype=int)
        seq_neg_cats = np.zeros((seq_len, max_cand), dtype=int)
        seq_neg_main_cats = np.zeros((seq_len, max_cand), dtype=int)
        
        seq_neg_probs = np.zeros((seq_len, max_cand), dtype=np.float32)
        seq_neg_main_probs = np.zeros((seq_len, max_cand), dtype=np.float32)
        
        seq_neg_dists = np.zeros((seq_len, max_cand), dtype=np.float32)
        seq_neg_feats = np.zeros((seq_len, max_cand, 1), dtype=np.float32)
        
        seq_true_labels = np.zeros(seq_len, dtype=int)
        seq_true_global_ids = true_poi_indices 
        
        n_random = int(max_cand * self.neg_strategy[0])
        n_semantic = max_cand - n_random
        
        uid = user_ids[0]
        user_freq_counter = self.user_cand_freq.get(uid, Counter())
        user_total_visits = self.user_total_cands.get(uid, 1.0)
        
        if torch.is_tensor(self.processor.main_cat_mapping):
            main_cat_mapping_np = self.processor.main_cat_mapping.cpu().numpy()
        else:
            main_cat_mapping_np = self.processor.main_cat_mapping

        for t in range(seq_len):
            t_true_id = true_poi_indices[t]
            t_coord = true_coords[t].reshape(1, -1) 
            curr_center_rad = center_coords[t].reshape(1, -1) 
            t_slot = time_slots[t]
            
            # --- Candidates ---
            cands = self.processor.ball_tree.query_radius(t_coord, r=self.radius_rad)[0]
            if t_true_id not in cands: cands = np.append(cands, t_true_id)
            if len(cands) > max_cand:
                sel = np.random.choice(cands[cands != t_true_id], max_cand - 1, replace=False)
                cands = np.append(sel, t_true_id)
            np.random.shuffle(cands)
            
            seq_true_labels[t] = np.where(cands == t_true_id)[0][0]
            curr_k = len(cands)
            
            # IDs
            seq_cand_ids[t, :curr_k] = cands
            seq_cand_cats[t, :curr_k] = self.processor.poi_cat_indices[cands] + 1
            
            # Main Cat ID Mapping
            small_cat_ids = self.processor.poi_cat_indices[cands]
            main_cat_ids = main_cat_mapping_np[small_cat_ids]
            seq_cand_main_cats[t, :curr_k] = main_cat_ids + 1
            
            # Probabilities
            # 1. Small Cat Prob
            seq_cand_probs[t, :curr_k] = self.processor.cat_time_probs[small_cat_ids, t_slot]
            valid_main_mask = (main_cat_ids != -1)
            main_probs = np.zeros(curr_k, dtype=np.float32)
            if valid_main_mask.any():
                valid_main_ids = main_cat_ids[valid_main_mask]
                main_probs[valid_main_mask] = self.processor.main_cat_probs[valid_main_ids, t_slot]
            
            seq_cand_main_probs[t, :curr_k] = main_probs
            
            seq_cand_mask[t, :curr_k] = 1
            
            dists_rad = np.linalg.norm(self.processor.poi_coords[cands] - curr_center_rad, axis=1)
            seq_cand_dists[t, :curr_k] = np.log1p(dists_rad * 6371000.0)
            
            # Feats
            raw_recur_counts = np.array([user_freq_counter.get(c, 0) for c in cands], dtype=np.float32)
            recurrence_counts = np.maximum(0, raw_recur_counts - 1) 
            pop_recur = np.log1p(recurrence_counts) / np.log1p(user_total_visits) 
            
            seq_cand_feats[t, :curr_k, 0] = pop_recur
            
            # --- Negatives ---
            negs = []
            if n_semantic > 0:
                semantic_candidates = []
                attempts = 0
                while len(semantic_candidates) < n_semantic and attempts < 20: 
                    batch_rand = np.random.randint(0, self.processor.num_pois, size=50)
                    mask_semantic = (self.processor.cat_time_probs[self.processor.poi_cat_indices[batch_rand], t_slot] <= self.config.semantic_sample_thresh) & (~np.isin(batch_rand, cands))
                    semantic_candidates.extend(batch_rand[mask_semantic])
                    attempts += 1
                negs.extend(semantic_candidates[:n_semantic])

            while len(negs) < max_cand:
                ridx = np.random.randint(0, self.processor.num_pois)
                if ridx not in cands and ridx not in negs: negs.append(ridx)
            negs = np.array(negs[:max_cand])
            
            # Negative IDs
            seq_neg_ids[t, :] = negs
            seq_neg_cats[t, :] = self.processor.poi_cat_indices[negs] + 1
            
            # Negative Main Cat IDs
            neg_small_cat_ids = self.processor.poi_cat_indices[negs]
            neg_main_cat_ids = main_cat_mapping_np[neg_small_cat_ids]
            seq_neg_main_cats[t, :] = neg_main_cat_ids + 1
            
            # Negative Probs
            seq_neg_probs[t, :] = self.processor.cat_time_probs[neg_small_cat_ids, t_slot]
            
            valid_neg_main_mask = (neg_main_cat_ids != -1)
            neg_main_probs = np.zeros(max_cand, dtype=np.float32)
            if valid_neg_main_mask.any():
                valid_neg_main_ids = neg_main_cat_ids[valid_neg_main_mask]
                neg_main_probs[valid_neg_main_mask] = self.processor.main_cat_probs[valid_neg_main_ids, t_slot]
            seq_neg_main_probs[t, :] = neg_main_probs
            neg_coords = self.processor.poi_coords[negs]
            neg_dists = np.linalg.norm(neg_coords - curr_center_rad, axis=1) * 6371000.0

            if n_semantic > 0 and len(negs) > 0:
                n_fake = min(n_semantic, len(negs))
                fake_dists = np.random.uniform(low=0, high=100, size=n_fake)
                neg_dists[:n_fake] = fake_dists
                
            seq_neg_dists[t, :] = np.log1p(neg_dists)
            n_recur_counts = np.array([user_freq_counter.get(n, 0) for n in negs], dtype=np.float32)
            n_pop_recur = np.log1p(n_recur_counts) / np.log1p(user_total_visits)
            seq_neg_feats[t, :, 0] = n_pop_recur

        pad_len = self.max_seq_len - seq_len
        seq_mask = np.concatenate([np.ones(seq_len), np.zeros(pad_len)])
        def pad_tensor(arr, pad_val=0):
            if pad_len == 0: return arr
            shape = list(arr.shape)
            shape[0] = pad_len
            return np.concatenate([arr, np.full(shape, pad_val, dtype=arr.dtype)], axis=0)

        return {
            'user_id': torch.tensor(pad_tensor(user_ids), dtype=torch.long),
            'time_slot': torch.tensor(pad_tensor(time_slots), dtype=torch.long),
            'center_coord': torch.tensor(pad_tensor(center_coords), dtype=torch.float32),
            'seq_mask': torch.tensor(seq_mask, dtype=torch.bool),
            
            'cand_poi_ids': torch.tensor(pad_tensor(seq_cand_ids), dtype=torch.long),
            'cand_cat_ids': torch.tensor(pad_tensor(seq_cand_cats), dtype=torch.long),
            'cand_main_cat_ids': torch.tensor(pad_tensor(seq_cand_main_cats), dtype=torch.long),
            
            'cand_probs': torch.tensor(pad_tensor(seq_cand_probs), dtype=torch.float32),
            'cand_main_cat_probs': torch.tensor(pad_tensor(seq_cand_main_probs), dtype=torch.float32),
            
            'cand_mask': torch.tensor(pad_tensor(seq_cand_mask), dtype=torch.float32),
            'cand_dists': torch.tensor(pad_tensor(seq_cand_dists), dtype=torch.float32),
            'cand_other_feats': torch.tensor(pad_tensor(seq_cand_feats), dtype=torch.float32), 
            
            'neg_poi_ids': torch.tensor(pad_tensor(seq_neg_ids), dtype=torch.long),
            'neg_cat_ids': torch.tensor(pad_tensor(seq_neg_cats), dtype=torch.long),
            'neg_main_cat_ids': torch.tensor(pad_tensor(seq_neg_main_cats), dtype=torch.long),
            
            'neg_probs': torch.tensor(pad_tensor(seq_neg_probs), dtype=torch.float32),
            'neg_main_cat_probs': torch.tensor(pad_tensor(seq_neg_main_probs), dtype=torch.float32),
            
            'neg_dists': torch.tensor(pad_tensor(seq_neg_dists), dtype=torch.float32),
            'neg_other_feats': torch.tensor(pad_tensor(seq_neg_feats), dtype=torch.float32),
            'label_pos': torch.tensor(pad_tensor(seq_true_labels, -1), dtype=torch.long),
            'true_poi_id': torch.tensor(pad_tensor(seq_true_global_ids, -1), dtype=torch.long),
            'sample_idx': torch.tensor(idx, dtype=torch.long) 
        }

def seq_collate_fn(batch):
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        collated[key] = torch.stack([item[key] for item in batch])
    return collated


