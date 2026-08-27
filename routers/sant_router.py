import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances
import fasttext
import fasttext.util

fasttext.util.download_model("en", if_exists="ignore")
ft_embed = fasttext.load_model("cc.en.300.bin")

def embed_text_fasttext(text):
    text_clean = text.replace("\n", " ").replace("\r", " ")
    return ft_embed.get_sentence_vector(text_clean)

def embed_batch(texts):
    return np.array([embed_text_fasttext(t) for t in texts])

def compute_topk_neighbors(embs, k=3):
    sims = 1 - pairwise_distances(embs, metric="cosine")
    N = sims.shape[0]
    neighbors = []
    for i in range(N):
        idxs = np.argsort(-sims[i])
        topk = [j for j in idxs if j != i][:k]
        neighbors.append(topk)
    return neighbors, sims

class FastTextAnnotator(nn.Module):
    def __init__(self, texts, labels, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.model = self._train_supervised_fasttext(texts, labels)
    def _train_supervised_fasttext(self, texts, labels):
        path = "ft_train_tmp.txt"
        with open(path, "w") as f:
            for t, y in zip(texts, labels):
                t_clean = t.replace("\n", " ").replace("\r", " ")
                f.write(f"__label__{y} {t_clean}\n")
        return fasttext.train_supervised(path, lr=0.1, epoch=25, wordNgrams=2, verbose=0)
    def update(self, texts, labels):
        self.model = self._train_supervised_fasttext(texts, labels)
    def forward(self, emb_batch, texts=None):
        if texts is None:
            raise ValueError("FastTextAnnotator requires raw texts")
        probs_list = []
        for t in texts:
            labels, probs = self.model.predict(t.replace("\n", " ").replace("\r", " "), k=self.num_classes)
            probs_list.append(probs)
        probs_np = np.array(probs_list, dtype=np.float32)
        probs_tensor = torch.from_numpy(probs_np)
        logits = torch.log(probs_tensor + 1e-12)
        return logits

class EATNet(nn.Module):
    def __init__(self, emb_dim, model_out_dim, neigh_dim, hidden=256):
        super().__init__()
        input_dim = emb_dim + model_out_dim + neigh_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 2)
        )
    def forward(self, emb, model_probs, neigh_features):
        x = torch.cat([emb, model_probs, neigh_features], dim=-1)
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)
        p_error = probs[:, 1]
        return p_error, probs

def al_entropy_from_logits(logits):
    p = F.softmax(logits, dim=-1)
    ent = -torch.sum(p * torch.log(p + 1e-12), dim=-1)
    max_ent = math.log(p.size(-1))
    return ent / (max_ent + 1e-12)

def train_sant(texts, true_labels, initial_human_idx,
               budget_percent=0.2, epochs=10, k_neighbors=3,
               T0=0.2, device="cpu"):
    N = len(texts)
    budget_total = int(budget_percent * N)
    embs = embed_batch(texts)
    neighbors, sims = compute_topk_neighbors(embs, k=k_neighbors)
    emb_dim = embs.shape[1]
    warm_texts = [texts[i] for i in initial_human_idx]
    warm_labels = [true_labels[i] for i in initial_human_idx]
    model = FastTextAnnotator(warm_texts, warm_labels,
                              num_classes=len(np.unique(true_labels))).to(device)
    eat = EATNet(emb_dim=emb_dim, model_out_dim=len(np.unique(true_labels)),
                 neigh_dim=k_neighbors).to(device)
    opt_eat = torch.optim.Adam(eat.parameters(), lr=5e-4)
    labeled = set(initial_human_idx)
    unlabeled = set(range(N)) - labeled
    budget_used = len(labeled)
    sims_matrix = sims
    for epoch in range(epochs):
        s_eat_arr = np.zeros(N)
        s_al_arr = np.zeros(N)
        with torch.no_grad():
            for i in list(unlabeled):
                emb_i = torch.tensor(embs[i], dtype=torch.float32, device=device).unsqueeze(0)
                logits = model(emb_i, texts=[texts[i]])
                probs = F.softmax(logits, dim=-1)
                nbr_idxs = neighbors[i]
                nbr_probs = []
                for j in nbr_idxs:
                    emb_j = torch.tensor(embs[j], dtype=torch.float32, device=device).unsqueeze(0)
                    l_j = model(emb_j, texts=[texts[j]])
                    nbr_probs.append(F.softmax(l_j, dim=-1).cpu().numpy().squeeze())
                nbr_probs = np.array(nbr_probs)
                nbr_ent = -np.sum(nbr_probs * np.log(nbr_probs + 1e-12), axis=1)
                C_t = sims_matrix[i, nbr_idxs]
                weighted = C_t * nbr_ent
                neigh_feat = torch.tensor(weighted.astype(np.float32), device=device).unsqueeze(0)
                p_error, _ = eat(emb_i, probs, neigh_feat)
                s_eat_arr[i] = float(p_error.cpu().numpy())
                s_al_arr[i] = float(al_entropy_from_logits(logits).cpu().numpy())
        eta = math.exp((budget_used / max(1, N)) - T0)
        s_bit = (s_al_arr ** eta) * s_eat_arr
        remaining_budget = budget_total - budget_used
        if remaining_budget <= 0:
            break
        candidate_indices = np.array(list(unlabeled))
        candidate_scores = s_bit[candidate_indices]
        pick_k = min(len(candidate_indices), remaining_budget)
        selected_local = candidate_indices[np.argsort(-candidate_scores)[:pick_k]]
        for idx in selected_local:
            labeled.add(int(idx))
            unlabeled.remove(int(idx))
            budget_used += 1
        labeled_list = list(labeled)
        model.update([texts[i] for i in labeled_list], [true_labels[i] for i in labeled_list])
        eat.train()
        emb_tensor = torch.tensor(embs[labeled_list], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits_all = model(emb_tensor, texts=[texts[i] for i in labeled_list])
            probs_all = F.softmax(logits_all, dim=-1)
            preds = torch.argmax(probs_all, dim=-1).cpu().numpy()
        ys = np.array(true_labels)[labeled_list]
        wrong_indicator = (preds != ys).astype(np.float32)
        neigh_feats = []
        for ii in labeled_list:
            nbr_idxs = neighbors[ii]
            nbr_probs = []
            for j in nbr_idxs:
                emb_j = torch.tensor(embs[j], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    l_j = model(emb_j, texts=[texts[j]])
                    p_j = F.softmax(l_j, dim=-1).cpu().numpy().squeeze()
                nbr_probs.append(p_j)
            nbr_probs = np.array(nbr_probs)
            nbr_ent = -np.sum(nbr_probs * np.log(nbr_probs + 1e-12), axis=1)
            C_t = sims_matrix[ii, nbr_idxs]
            neigh_feats.append((C_t * nbr_ent).astype(np.float32))
        neigh_feats = torch.tensor(neigh_feats, dtype=torch.float32, device=device)
        opt_eat.zero_grad()
        p_error_pred, _ = eat(emb_tensor, probs_all, neigh_feats)
        Ld = F.binary_cross_entropy(torch.tensor(p_error_pred, device=device),
                                    torch.tensor(wrong_indicator, device=device))
        with torch.no_grad():
            logits_l = model(emb_tensor, texts=[texts[i] for i in labeled_list])
            per_sample_loss = F.cross_entropy(logits_l, torch.tensor(ys, device=device), reduction="none")
        mask_model = (p_error_pred.detach() < 0.5)
        mask_human = (p_error_pred.detach() >= 0.5)
        Lm_f = per_sample_loss[mask_model].mean() if mask_model.sum() > 0 else torch.tensor(0.0, device=device)
        Lh_f = per_sample_loss[mask_human].mean() if mask_human.sum() > 0 else torch.tensor(0.0, device=device)
        alpha = 0.1
        Lm_margin = torch.clamp(alpha + Lm_f - Lh_f, min=0.0)
        LEAT = Ld + Lm_margin
        LEAT.backward()
        opt_eat.step()
    return model, eat, embs, neighbors, sims

class SANTRouter:
    def __init__(self, human_budget, fixed_budget=False, T0=0.2, device="cpu"):
        self.T0 = T0
        self.device = device
        self.fixed_budget = fixed_budget
        self.human_budget = human_budget
    def compute_eta(self, t, N):
        return math.exp((t / max(1, N)) - self.T0)
    def compute_dAL(self, logits):
        p = torch.softmax(logits, dim=-1)
        ent = -torch.sum(p * torch.log(p + 1e-12), dim=-1)
        max_ent = math.log(p.size(-1))
        return ent / (max_ent + 1e-12)
    def neigh_feats(self, idx, embs, texts, f_model, neighbors, sims):
        nbr = neighbors[idx]
        feats = []
        for j in nbr:
            emb_j = torch.tensor(embs[j], dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                p_j = torch.softmax(f_model(emb_j, texts=[texts[j]]), dim=-1).cpu().numpy().squeeze()
            ent = -np.sum(p_j * np.log(p_j + 1e-12))
            feats.append(ent * sims[idx, j])
        return torch.tensor([feats], dtype=torch.float32, device=self.device)
    def compute_dEAT(self, emb, model_probs, neigh_feats, eat):
        p_error, _ = eat(emb, model_probs, neigh_feats)
        return p_error.squeeze().item()
    
    def route_standard(self, llm_predictions, human_labels, texts_to_annotate,
        num_samples_to_train=30, threshold=0.5, **kwargs):
        self.actions = [1 for i in range(num_samples_to_train)]
        output_set = [human_labels[i] for i in range(num_samples_to_train)]
        self.human_used = num_samples_to_train
        self.llm_used = 0
        self.money_used = 0
        backup = kwargs.get("backup", None)

        self.texts = texts_to_annotate[:num_samples_to_train]
        self.samples_not_for_training = texts_to_annotate[num_samples_to_train:]
        labels = human_labels[:num_samples_to_train]
        initial_human_idx = list(range(num_samples_to_train))
        model_f, eat, embs, neighbors, sims = train_sant(
            self.texts,
            labels,
            initial_human_idx=initial_human_idx,
            epochs=40,
            k_neighbors=3,
            device=self.device
        )
        embs_all = embed_batch(texts_to_annotate)
        neighbors_all, sims_all = compute_topk_neighbors(embs_all, k=3)
        offset = num_samples_to_train
        candidate_indices = list(range(offset, len(texts_to_annotate)))
        d_bit_scores = []
        with torch.no_grad():
            for global_idx in candidate_indices:
                txt = texts_to_annotate[global_idx]
                emb_vec = embs_all[global_idx]
                emb = torch.tensor(emb_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
                logits = model_f(emb, texts=[txt])
                probs = torch.softmax(logits, dim=-1)
                dAL = float(self.compute_dAL(logits).cpu().numpy())
                neigh = self.neigh_feats(global_idx, embs_all, texts_to_annotate,
                                     model_f, neighbors_all, sims_all)
                dEAT = self.compute_dEAT(emb, probs, neigh, eat)
                eta = self.compute_eta(global_idx, len(texts_to_annotate))
                d_bit = (dAL ** eta) * dEAT
                d_bit_scores.append(d_bit)

        candidate_indices = np.array(candidate_indices)
        d_bit_scores = np.array(d_bit_scores)
        sorted_order = np.argsort(-d_bit_scores)

        money_budget = kwargs.get('money_budget', None)
        annotator_cost = kwargs.get('annotator_cost', None)
        human_cost = annotator_cost[-1] if annotator_cost else 0
        llm_cost = annotator_cost[0] if annotator_cost else 0

        if money_budget is not None:
            self.money_used = num_samples_to_train * human_cost

        num_candidates = len(candidate_indices)
        human_indices = set()
        if money_budget is not None:
            budget_after_training = money_budget - self.money_used
            # Greedily assign human to highest-scored candidates while
            # ensuring remaining candidates can still afford at least LLM
            remaining_to_assign = num_candidates
            budget_left = budget_after_training
            for rank_idx in sorted_order:
                if d_bit_scores[rank_idx] < threshold:
                    break
                others_after = remaining_to_assign - 1
                cost_if_human = human_cost + others_after * llm_cost
                if cost_if_human <= budget_left:
                    human_indices.add(int(candidate_indices[rank_idx]))
                    budget_left -= human_cost
                    remaining_to_assign -= 1
                else:
                    break
        else:
            remaining_human = self.human_budget - self.human_used
            for rank_idx in sorted_order:
                if d_bit_scores[rank_idx] < threshold:
                    break
                if remaining_human <= 0:
                    break
                human_indices.add(int(candidate_indices[rank_idx]))
                remaining_human -= 1

        for global_idx in range(offset, len(texts_to_annotate)):
            if global_idx in human_indices:
                action = 1
                self.human_used += 1
                output_set.append(human_labels[global_idx])
                if money_budget is not None:
                    self.money_used += human_cost
            else:
                if money_budget is not None and self.money_used + llm_cost > money_budget:
                    action = -1
                    if backup:
                        output_set.append(llm_predictions[global_idx])
                    else:
                        output_set.append(human_labels[global_idx] + 1)
                else:
                    action = 0
                    self.llm_used += 1
                    output_set.append(llm_predictions[global_idx])
                    if money_budget is not None:
                        self.money_used += llm_cost
            self.actions.append(action)

        return output_set
    
    def route(self, llm_predictions, human_labels, texts_to_annotate,
        num_samples_to_train=30, threshold=0.5, **kwargs):
        if not self.fixed_budget:
            return self.route_standard(llm_predictions=llm_predictions,
                                       human_labels=human_labels,
                                       texts_to_annotate=texts_to_annotate,
                                       num_samples_to_train=num_samples_to_train,
                                       threshold=threshold, **kwargs)
        self.actions = [1 for i in range(num_samples_to_train)]
        output_set = [human_labels[i] for i in range(num_samples_to_train)]
        self.human_used = num_samples_to_train
        self.money_used = 0
        
        backup =  kwargs.get("backup", None)
        
        self.texts = texts_to_annotate[:num_samples_to_train]
        self.samples_not_for_training = texts_to_annotate[num_samples_to_train:]
        labels = human_labels[:num_samples_to_train]
        initial_human_idx = list(range(num_samples_to_train))
        model_f, eat, embs, neighbors, sims = train_sant(
            self.texts,
            labels,
            initial_human_idx=initial_human_idx,
            epochs=40,
            k_neighbors=3,
            device=self.device
        )
        embs_new = embed_batch(self.samples_not_for_training)
        embs_all = np.vstack([embs, embs_new]) if embs_new.size else embs
        candidate_indices = np.arange(num_samples_to_train, len(texts_to_annotate))
        d_bit_scores = []
        with torch.no_grad():
            for idx in candidate_indices:
                txt = texts_to_annotate[idx]
                emb_vec = embs_all[idx]
                emb = torch.tensor(emb_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
                logits = model_f(emb, texts=[txt])
                probs = torch.softmax(logits, dim=-1)
                dAL = float(self.compute_dAL(logits).cpu().numpy())
                if idx < num_samples_to_train:
                    neigh = self.neigh_feats(idx, embs, self.texts, model_f, neighbors, sims)
                else:
                    neigh = torch.zeros((1, 3), device=self.device)
                dEAT = self.compute_dEAT(emb, probs, neigh, eat)
                eta = self.compute_eta(idx, len(texts_to_annotate))
                d_bit = (dAL ** eta) * dEAT
                d_bit_scores.append(d_bit)
        d_bit_scores = np.array(d_bit_scores)
        remaining_budget = self.human_budget - num_samples_to_train
        if self.fixed_budget and remaining_budget > 0:
            idx_sorted = candidate_indices[np.argsort(-d_bit_scores)[:remaining_budget]]
            human_indices = set(idx_sorted)
        else:
            human_indices = set(candidate_indices[d_bit_scores >= threshold])
        for idx in range(len(texts_to_annotate)):
            if idx < num_samples_to_train:
                action = 1
            else:
                action = 1 if idx in human_indices else 0
            self.actions.append(action)
            if action == 1:
                output_set.append(human_labels[idx])
                if idx >= num_samples_to_train:
                    self.human_used += 1
            else:
                output_set.append(llm_predictions[idx])
        return output_set

if __name__ == "__main__":
    from src.data import NLPDataset
    data = NLPDataset('banking77')
    texts = data.x[:101]
    real_labels = data.y[:101]
    llm_predictions = np.load("predictions/all_predictions_banking77.npy")[:101]
    router = SANTRouter(human_budget=50)
    samples = router.route(llm_predictions=llm_predictions,
                           human_labels=real_labels,
                           texts_to_annotate=texts)
    print(samples)
    print(router.actions)