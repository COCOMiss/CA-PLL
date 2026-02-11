import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_logger

logger = get_logger("Model")

class LocationEncoder(nn.Module):
    def __init__(self, input_dim=2, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, coords):
        return self.net(coords)
    
class UserTrajectoryModel(nn.Module):

    def __init__(self, num_users, num_time_slots, embed_dim=64, num_heads=4, num_layers=2, max_len=50):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim, padding_idx=0)
        self.time_emb = nn.Embedding(num_time_slots, embed_dim, padding_idx=0)
        self.loc_encoder = LocationEncoder(input_dim=2, embed_dim=embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        
        concat_dim = embed_dim * 3 
        self.feature_fusion = nn.Sequential(
            nn.Linear(concat_dim, embed_dim),
            nn.LayerNorm(embed_dim), 
            nn.ReLU(),               
            nn.Dropout(0.1)          
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim*4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, user_idx, time_slot, current_coord, src_key_padding_mask=None):
        e_user = self.user_emb(user_idx)      
        e_time = self.time_emb(time_slot)     
        e_loc = self.loc_encoder(current_coord) 
      
        concat_tensor = torch.cat([e_user, e_time, e_loc], dim=-1)
        token_emb = self.feature_fusion(concat_tensor)
        seq_len = token_emb.size(1)
        pos_ids = torch.arange(seq_len, device=token_emb.device).unsqueeze(0)
        pos_embedding = self.pos_emb(pos_ids)
        
        # Add & Dropout
        token_emb = token_emb + pos_embedding
        
        output = self.transformer(token_emb, src_key_padding_mask=src_key_padding_mask)
        return output
  
class CandidatePOIEncoder(nn.Module):
    """
    Input: 
        - Embeddings: POI, SmallCat, MainCat
        - Scalars: Prob, MainProb, Dist, GlobalPop, Recurrence
    """
    def __init__(self, num_pois, num_cats, num_main_cats, embed_dim=64, dataset_name="ny_filterPOI"):
        super().__init__()
        
        # 1. POI Embedding
        self.poi_emb = nn.Embedding(num_pois, embed_dim, padding_idx=0)
        
        # 2. Small Category Embedding (Qwen Init Logic)
        qwen_emb_path = f'dataset/{dataset_name}/category_qwen_emb.pt'
        self.use_qwen = False
        
        if os.path.exists(qwen_emb_path):
            logger.info(f"Loading Qwen Embeddings from {qwen_emb_path}...")
            qwen_tensor = torch.load(qwen_emb_path)
            qwen_dim = qwen_tensor.size(1)
            
            full_weight = torch.zeros(num_cats, qwen_dim)
            valid_cats_count = min(num_cats - 1, qwen_tensor.size(0))
            full_weight[1:1+valid_cats_count] = qwen_tensor[:valid_cats_count]
            
            self.cat_emb = nn.Embedding.from_pretrained(full_weight, padding_idx=0, freeze=False)
            self.cat_proj = nn.Sequential(
                nn.Linear(qwen_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU()
            )
            self.use_qwen = True
        else:
            self.cat_emb = nn.Embedding(num_cats, embed_dim, padding_idx=0)
            
        self.main_cat_emb = nn.Embedding(num_main_cats, embed_dim, padding_idx=0)
        
        # 4. Feature Fusion Layer
        # Calculation:
        #   Embeddings: POI(64) + SmallCat(64) + MainCat(64) = 192 (3*dim)
        #   Scalars:
        #     - SmallCat Prob (1)
        #     - MainCat Prob (1)  <-- New
        #     - Dist (1)
        #     - Recurrence (1)    <-- from other_feats
        #   Total Scalars = 4
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 3 + 4, embed_dim), 
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, poi_ids, cat_ids, main_cat_ids, probs, main_probs, dists, other_feats):
        """
        Input:
            main_cat_ids: [B, S, K]
            main_probs:   [B, S, K] P(MainCat|Time)
            other_feats:  [B, S, K, 2] [GlobalPop, Recurrence]
        """
        # Embeddings
        e_poi = self.poi_emb(poi_ids) 
        
        if self.use_qwen:
            e_cat = self.cat_proj(self.cat_emb(cat_ids))
        else:
            e_cat = self.cat_emb(cat_ids)
            
        e_main = self.main_cat_emb(main_cat_ids)
        
        # Scalars
        probs_exp = probs.unsqueeze(-1)
        main_probs_exp = main_probs.unsqueeze(-1)
        dists_exp = dists.unsqueeze(-1)
        
        # Concat All
        # [B, S, K, 3*dim + 5]
        concat_features = torch.cat([
            e_poi, e_cat, e_main, 
            probs_exp, main_probs_exp, dists_exp, 
            other_feats
        ], dim=-1)
        
        return self.fusion_layer(concat_features)
 
class TrajPOITransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.embed_dim
        
        # Config needs to have num_main_cats
        # If not passed explicitly, infer from mapping max or pass as arg
        # Here assuming config has it or we calculate it
        if hasattr(config, 'num_main_cats'):
            self.num_main_cats = config.num_main_cats
        else:
            logger.error("not find num_main_cats in config")
            # Fallback or Error
            self.num_main_cats = 20 # Placeholder, ensure this matches dataset!
        
        if hasattr(config, 'num_cats'):
            self.num_cats=config.num_cats
        else:
            logger.error("not find num_cats in config")
        
        self.user_model = UserTrajectoryModel(
            num_users=len(config.user2idx) + 1,
            num_time_slots=24 + 1,
            embed_dim=embed_dim
        )
        
        self.candidate_model = CandidatePOIEncoder(
            dataset_name=config.dataset_name,
            num_pois=len(config.venue_id2idx) + 1, 
            num_cats=len(config.cat2idx) + 1,
            num_main_cats=self.num_main_cats + 1, # +1 for padding
            embed_dim=embed_dim
        )
        
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 32)
        )
        self.prototypes = nn.Linear(32, len(config.cat2idx) + 1, bias=False)
        self.cat_intent_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, self.num_cats)
        )

        self.main_intent_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, self.num_main_cats)
        )

    def forward(self, batch):
        """ Training Forward """
        padding_mask = ~batch['seq_mask'].bool()
        
        user_vector = self.user_model(
            batch['user_id'], batch['time_slot'], batch['center_coord'], padding_mask
        )
        
        # Candidates
        cand_vectors = self.candidate_model(
            poi_ids=batch['cand_poi_ids'], 
            cat_ids=batch['cand_cat_ids'],
            main_cat_ids=batch['cand_main_cat_ids'],
            probs=batch['cand_probs'], 
            main_probs=batch['cand_main_cat_probs'],
            dists=batch['cand_dists'],
            other_feats=batch['cand_other_feats']
        )
        cand_scores = (user_vector.unsqueeze(2) * cand_vectors).sum(dim=-1)
        # Negatives
        neg_vectors = self.candidate_model(
            poi_ids=batch['neg_poi_ids'], 
            cat_ids=batch['neg_cat_ids'],
            main_cat_ids=batch['neg_main_cat_ids'],
            probs=batch['neg_probs'], 
            main_probs=batch['neg_main_cat_probs'],
            dists=batch['neg_dists'],
            other_feats=batch['neg_other_feats']
        )
        
        neg_scores = (user_vector.unsqueeze(2) * neg_vectors).sum(dim=-1)
        
        return cand_scores, neg_scores, user_vector

    def forward_contrastive(self, user_vector):
        feat = F.normalize(self.proj_head(user_vector), dim=-1)
        proto_logits = self.prototypes(feat)
        return proto_logits

    
    def compute_weighted_clpl_loss(self, cand_scores, cand_mask, neg_scores, seq_mask,user_vector=None, cand_cat_ids=None):
        # --- 1. Positive Part (Candidate Weighted) ---
        masked_cand_scores = cand_scores.masked_fill(cand_mask == 0, -1e9)
        # Detach weights
        cand_weights = F.softmax(masked_cand_scores, dim=-1).detach() 
        
        # Weighted Sum (Maximize positive scores)
        cand_weighted_score = (cand_scores * cand_weights * cand_mask).sum(dim=-1)
        loss_pos = F.softplus(-cand_weighted_score) # log(1 + exp(-score))
        # --- 2. Negative Part (Negative Weighted) ---
        
        neg_weights = F.softmax(neg_scores, dim=-1).detach()
        # Weighted Sum (Minimize negative scores)
        neg_weighted_score = (neg_scores * neg_weights).sum(dim=-1)
        loss_neg = F.softplus(neg_weighted_score)

        # --- 3. Combine ---
        step_loss = loss_pos + loss_neg
        
        valid_loss = (step_loss * seq_mask.float()).sum()
        num_valid_steps = seq_mask.float().sum().clamp(min=1.0)
        
        return valid_loss / num_valid_steps
    
    
    def predict(self, batch):
        """ Test/Query Forward with Viterbi """
        padding_mask = ~batch['seq_mask'].bool()
        
        user_vector = self.user_model(
            batch['user_id'], batch['time_slot'], batch['center_coord'], padding_mask
        )
        cand_vectors = self.candidate_model(
            batch['cand_poi_ids'], batch['cand_cat_ids'], batch['cand_main_cat_ids'],
            batch['cand_probs'], batch['cand_main_cat_probs'],
            batch['cand_dists'], batch['cand_other_feats']
        )
        emission_scores = (user_vector.unsqueeze(2) * cand_vectors).sum(dim=-1)
        emission_scores = emission_scores.masked_fill(batch['cand_mask'] == 0, -1e9)
        
        return emission_scores
    
    
