import math
import random
import numpy as np
from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import fasttext
import fasttext.util

fasttext.util.download_model("en", if_exists="ignore")
ft_embed = fasttext.load_model("cc.en.300.bin")

try:
    import faiss

    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


def embed_text_fasttext(text):
    text = text.replace("\n", " ").replace("\r", " ")
    return ft_embed.get_sentence_vector(text)


def embed_batch(texts):
    return np.array([embed_text_fasttext(t) for t in texts])


class FastTextAnnotatorWrapper(nn.Module):
    def __init__(
        self, train_texts: List[str], train_labels: List[int], num_classes: int
    ):
        super().__init__()
        self.num_classes = num_classes
        tmp = "ft_train_tmp.txt"
        with open(tmp, "w", encoding="utf-8") as f:
            for t, y in zip(train_texts, train_labels):
                t_clean = t.replace("\n", " ").replace("\r", " ")
                f.write(f"__label__{y} {t_clean}\n")
        self.ft_model = fasttext.train_supervised(
            tmp, lr=0.1, epoch=20, wordNgrams=2, verbose=0
        )

    def forward(self, x_emb: torch.Tensor):
        raise RuntimeError("Use predict_raw(texts) for FastTextAnnotatorWrapper")

    def predict_raw(self, texts: List[str]) -> torch.Tensor:
        all_probs = []
        for t in texts:
            labels, probs = self.ft_model.predict(t.replace("\n", " ").replace("\r", " "), k=self.num_classes)
            probs = np.array(probs, dtype=np.float32)
            if len(probs) < self.num_classes:
                pad = np.zeros(self.num_classes - len(probs), dtype=np.float32)
                probs = np.concatenate([probs, pad])
            all_probs.append(probs)
        probs = np.stack(all_probs, axis=0)
        logits = np.log(probs + 1e-12)
        return torch.tensor(logits, dtype=torch.float32)

    def predict_classes(self, texts: List[str]) -> np.ndarray:
        preds = []
        for t in texts:
            lbls, probs = self.ft_model.predict(t, k=1)
            preds.append(int(lbls[0].replace("__label__", "")))
        return np.array(preds, dtype=np.int64)


def label_smoothing_onehot(
    y_onehot: np.ndarray, num_classes: int, alpha: Optional[float] = None
):
    if alpha is None:
        alpha = 1.0 - 1.0 / float(num_classes)
    return y_onehot * (1.0 - alpha) + alpha / float(num_classes)


class KNNDatastore:
    def __init__(self, emb_dim: int, max_size: int = 1000, use_faiss: bool = True):
        self.emb_dim = emb_dim
        self.max_size = max_size
        self.use_faiss = use_faiss and _HAS_FAISS
        self.embs = []
        self.labels = []
        self.ids = []
        self.class_counts = {}
        self.index = None
        if self.use_faiss:
            self.index = faiss.IndexFlatIP(emb_dim)

    def add(self, emb: np.ndarray, label: int, idx: Optional[int] = None):
        emb = emb.astype(np.float32)
        self.embs.append(emb)
        self.labels.append(int(label))
        if idx is None:
            idx = len(self.ids)
        self.ids.append(idx)
        self.class_counts.setdefault(int(label), []).append(len(self.embs) - 1)
        if self.use_faiss:
            v = emb.reshape(1, -1)
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
            self.index.add(v)
        if len(self.embs) > self.max_size:
            self._maintain()

    def _maintain(self):
        major_class, _ = max(
            ((c, len(idxs)) for c, idxs in self.class_counts.items()),
            key=lambda t: t[1],
        )
        idxs = self.class_counts[major_class]
        emb_matrix = np.stack([self.embs[i] for i in idxs], axis=0)
        prototype = emb_matrix.mean(axis=0)
        sims = (
            emb_matrix
            @ prototype
            / (np.linalg.norm(emb_matrix, axis=1) * (np.linalg.norm(prototype) + 1e-12))
        )
        remove_pos = int(np.argmax(sims))
        global_idx = idxs[remove_pos]
        self._remove_at(global_idx)

    def _remove_at(self, pos: int):
        del self.embs[pos]
        del self.labels[pos]
        del self.ids[pos]
        self.class_counts = {}
        for i, lab in enumerate(self.labels):
            self.class_counts.setdefault(lab, []).append(i)
        if self.use_faiss:
            if len(self.embs) == 0:
                self.index.reset()
            else:
                arr = np.stack(self.embs, axis=0).astype(np.float32)
                arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
                self.index.reset()
                self.index.add(arr)

    def retrieve(
        self, query_emb: np.ndarray, k: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.embs) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        q = query_emb.astype(np.float32).reshape(1, -1)
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        if self.use_faiss and self.index is not None:
            D, I = self.index.search(qn, k)
            I = I[0]
            D = D[0]
            valid_mask = I >= 0
            return I[valid_mask].astype(np.int64), D[valid_mask].astype(np.float32)
        else:
            mat = np.stack(self.embs, axis=0)
            matn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
            sims = (matn @ qn.T).squeeze(-1)
            idxs = np.argsort(-sims)[:k]
            return idxs.astype(np.int64), sims[idxs].astype(np.float32)

    def size(self):
        return len(self.embs)


class LambdaNet(nn.Module):
    def __init__(self, k_neighbors: int, hidden: int = 128):
        super().__init__()
        input_dim = k_neighbors
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x_vec: torch.Tensor) -> torch.Tensor:
        out = self.net(x_vec)
        return torch.sigmoid(out).squeeze(-1)


class ARAIDA:
    def __init__(
        self,
        annotator_model,
        embed_fn,
        num_classes: int,
        k_neighbors: int = 20,
        datastore_max: int = 1000,
        device: str = "cpu",
    ):
        self.f = annotator_model
        self.embed_fn = embed_fn
        self.num_classes = num_classes
        self.k = k_neighbors
        self.device = device
        emb_dim = len(self.embed_fn("dummy input for dim").reshape(-1))
        self.datastore = KNNDatastore(
            emb_dim=emb_dim, max_size=datastore_max, use_faiss=True
        )
        self.lambda_net = LambdaNet(k_neighbors=self.k).to(device)
        self.opt_lambda = torch.optim.Adam(self.lambda_net.parameters(), lr=5e-4)

    def knn_infer(self, query_emb: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        if k is None:
            k = self.k
        idxs, sims = self.datastore.retrieve(query_emb, k=k)
        if len(idxs) == 0:
            return np.ones(self.num_classes, dtype=np.float32) / float(self.num_classes)
        Y = np.zeros((len(idxs), self.num_classes), dtype=np.float32)
        for i, nid in enumerate(idxs):
            lab = self.datastore.labels[int(nid)]
            Y[i, lab] = 1.0
        Ys = label_smoothing_onehot(Y, self.num_classes)
        dists = 1.0 - sims
        weights = 1.0 / (dists + 1e-6)
        weighted = (weights.reshape(-1, 1) * Ys).sum(axis=0) / (weights.sum() + 1e-12)
        return weighted.astype(np.float32)

    def build_lambda_input(
        self, query_text: str, neighbor_idxs: np.ndarray
    ) -> np.ndarray:
        if len(neighbor_idxs) == 0:
            return np.zeros(self.k, dtype=np.float32)
        q_emb = self.embed_fn(query_text).astype(np.float32).reshape(1, -1)
        nbr_embs = np.stack(
            [self.datastore.embs[int(i)] for i in neighbor_idxs], axis=0
        )
        qn = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
        nbrn = nbr_embs / (np.linalg.norm(nbr_embs, axis=1, keepdims=True) + 1e-12)
        sims = (nbrn @ qn.T).squeeze(-1)
        dists = 1.0 - sims
        Ei = []
        for idx_pos, emb in enumerate(nbr_embs):
            neighbor_text = None
            Ei.append(
                1
                if self.f.predict_classes(["dummy"])[0]
                == self.datastore.labels[int(neighbor_idxs[idx_pos])]
                else 0
            )
        Ei = np.array(Ei, dtype=np.float32)
        x_vec = dists * Ei - dists * (1.0 - Ei)
        if len(x_vec) < self.k:
            pad = np.zeros(self.k - len(x_vec), dtype=np.float32)
            x_vec = np.concatenate([x_vec, pad], axis=0)
        return x_vec.astype(np.float32)

    def combine_predictions(
        self, f_vec: np.ndarray, g_vec: np.ndarray, lam: float
    ) -> np.ndarray:
        return lam * f_vec + (1.0 - lam) * g_vec

    def interact_and_train(
        self,
        batch_texts: List[str],
        human_labels: Optional[List[int]],
        budget_human: int,
        use_active_learning: bool = False,
        gumbel_tau: float = 0.5,
    ):
        B = len(batch_texts)
        f_probs = []
        embeddings = []
        logits_tensor = self.f.predict_raw(batch_texts)
        f_probs = F.softmax(logits_tensor, dim=-1).cpu().numpy()
        embeddings = [self.embed_fn(txt).astype(np.float32) for txt in batch_texts]
        g_probs = []
        lambda_inputs = []
        neighbor_idx_lists = []
        for i, emb in enumerate(embeddings):
            idxs, sims = self.datastore.retrieve(emb, k=self.k)
            neighbor_idx_lists.append(idxs)
            g = self.knn_infer(emb, k=self.k)
            g_probs.append(g)
            x_vec = self.build_lambda_input(batch_texts[i], idxs)
            lambda_inputs.append(x_vec)
        g_probs = np.stack(g_probs, axis=0)
        lambda_inputs = np.stack(lambda_inputs, axis=0)
        self.lambda_net.train()
        lam_tensor = torch.tensor(
            lambda_inputs, dtype=torch.float32, device=self.device
        )
        lam_preds = self.lambda_net(lam_tensor).detach().cpu().numpy()
        final_suggestions = []
        for i in range(B):
            lam = float(lam_preds[i])
            final = self.combine_predictions(f_probs[i], g_probs[i], lam)
            final_suggestions.append(final)
        final_suggestions = np.stack(final_suggestions, axis=0)
        if human_labels is not None:
            for i, y in enumerate(human_labels):
                emb = embeddings[i]
                lab = int(y)
                self.datastore.add(emb, lab)
            self._train_lambda(lambda_inputs, f_probs, human_labels)
        return final_suggestions, lam_preds, g_probs

    def _train_lambda(
        self,
        lambda_inputs_np: np.ndarray,
        f_probs_np: np.ndarray,
        human_labels: List[int],
    ):
        if len(human_labels) == 0:
            return
        X = torch.tensor(lambda_inputs_np, dtype=torch.float32, device=self.device)
        f_preds = np.argmax(f_probs_np, axis=-1)
        targets = (f_preds == np.array(human_labels, dtype=np.int64)).astype(np.float32)
        y = torch.tensor(targets, dtype=torch.float32, device=self.device)
        self.lambda_net.train()
        logits = self.lambda_net(X)
        loss = F.mse_loss(logits, y)
        self.opt_lambda.zero_grad()
        loss.backward()
        self.opt_lambda.step()


class ARAIDARouter:
    def __init__(self, human_budget, fixed_budget=False):
        self.human_budget = human_budget
        self.fixed_budget = fixed_budget

    def route(
        self,
        llm_predictions,
        human_labels,
        texts_to_annotate,
        num_samples_to_train=30,
        threshold=0.5,
        **kwargs
    ):

        self.actions = [1 for i in range(num_samples_to_train)]
        output_set = [human_labels[i] for i in range(num_samples_to_train)]
        self.human_used = num_samples_to_train
        self.llm_used = 0
        self.money_used = 0

        texts_to_train = texts_to_annotate[:num_samples_to_train]
        labels_to_annotate = human_labels[:num_samples_to_train]
        num_classes = max(human_labels) + 1
        backup =  kwargs.get("backup", None)

        annotator_model = FastTextAnnotatorWrapper(
            texts_to_train, labels_to_annotate, num_classes=num_classes
        )

        araida = ARAIDA(
            annotator_model=annotator_model,
            embed_fn=embed_text_fasttext,
            num_classes=num_classes,
            k_neighbors=20,
            datastore_max=1000,
            device="cpu",
        )

        final_suggestions, lam_preds, g_probs = araida.interact_and_train(
            batch_texts=texts_to_train,
            human_labels=labels_to_annotate,
            budget_human=num_samples_to_train,
        )

        candidate_texts = texts_to_annotate[num_samples_to_train:]
        lam_scores = []
        with torch.no_grad():
            for new_text in candidate_texts:
                emb = embed_text_fasttext(new_text)
                neighbor_idxs, _ = araida.datastore.retrieve(emb)
                lambda_input = araida.build_lambda_input(new_text, neighbor_idxs)
                lambda_tensor = torch.tensor(lambda_input, dtype=torch.float32).unsqueeze(0)
                lam = araida.lambda_net(lambda_tensor).item()
                lam_scores.append(lam)

        lam_scores = np.array(lam_scores)
        # Lower lambda = more uncertain = higher priority for human
        priority_order = np.argsort(lam_scores)

        money_budget = kwargs.get('money_budget', None)
        annotator_cost = kwargs.get('annotator_cost', None)
        human_cost = annotator_cost[-1] if annotator_cost else 0
        llm_cost = annotator_cost[0] if annotator_cost else 0

        if money_budget is not None:
            self.money_used = num_samples_to_train * human_cost

        num_candidates = len(candidate_texts)
        human_indices = set()
        if money_budget is not None:
            budget_after_training = money_budget - self.money_used
            # Greedily assign human to highest-priority candidates while
            # ensuring remaining candidates can still afford at least LLM
            remaining_to_assign = num_candidates
            budget_left = budget_after_training
            for rank_idx in priority_order:
                if lam_scores[rank_idx] >= threshold:
                    break
                others_after = remaining_to_assign - 1
                cost_if_human = human_cost + others_after * llm_cost
                if cost_if_human <= budget_left:
                    human_indices.add(rank_idx)
                    budget_left -= human_cost
                    remaining_to_assign -= 1
                else:
                    break
        else:
            remaining_human = self.human_budget - self.human_used
            for rank_idx in priority_order:
                if lam_scores[rank_idx] >= threshold:
                    break
                if remaining_human <= 0:
                    break
                human_indices.add(rank_idx)
                remaining_human -= 1

        for index in range(len(candidate_texts)):
            global_idx = index + num_samples_to_train
            if index in human_indices:
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

if __name__ == "__main__":
    
    from src.data import NLPDataset
    
    data = NLPDataset('banking77')
    texts = data.x[:101]
    real_labels = data.y[:101]
    llm_predictions = np.load("predictions/all_predictions_banking77.npy")[:101]
    
    router = ARAIDARouter(human_budget=50)
    
    samples = router.route(llm_predictions=llm_predictions,
                           human_labels=real_labels,
                           texts_to_annotate=texts)
    
    print(samples)
    print(router.actions)
    