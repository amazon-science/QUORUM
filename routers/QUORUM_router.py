import numpy as np
from collections import Counter
from typing import List, Literal, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity


class QUORUMRouter:
    """Annotation router using Bayesian Thompson Sampling to allocate samples across LLMs and a human annotator under budget constraints."""

    def __init__(
        self,
        base_features: np.ndarray,
        llm_names: List[str] = ['claude', 'nova_pro', 'qwen'],
        method: Literal["bandit", "QUORUM"] = "QUORUM",
        **kwargs
    ) -> None:

        self.n_base = base_features.shape[1]
        self.base_features = self._normalize(base_features)
        self.n_samples = base_features.shape[0]

        self.llm_names = llm_names
        self.n_llms = len(llm_names)
        self.n_arms = self.n_llms + 1
        self.human_arm = self.n_llms

        self.n_dynamic = 2 + self.n_llms
        self.n_features = self.n_base + self.n_dynamic

        self.eval_type: str = kwargs.get("eval_type", "")
        self.human_budget: int = kwargs["human_budget"]

        costs = kwargs["annotator_cost"]
        self.arm_cost = {i: costs[i] for i in range(self.n_llms)}
        self.arm_cost[self.human_arm] = costs[-1]

        if self.eval_type != "auditor_style":
            self.max_annotations = kwargs.get("max_annotations", 3)
            self.min_annotations = kwargs.get("min_annotations", 1)
        else:
            self.max_annotations = 1
            self.min_annotations = 1

        self.agreement_threshold = kwargs.get("agreement_threshold", 0.6)

        reservation_fraction = kwargs.get("budget_reservation", 0.2)
        self.reserved_budget = int(reservation_fraction * self.human_budget)
        self.working_budget = self.human_budget - self.reserved_budget
        self.reserved_used = 0

        self.posterior_mean = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.posterior_cov = [np.eye(self.n_features) for _ in range(self.n_arms)]

        self.posterior_mean[self.human_arm] = np.full(self.n_features, 0.3)
        self.posterior_cov[self.human_arm] *= 0.5

        self.noise_var = kwargs.get("noise_var", 0.5)

        self.actions = []
        self.annotations_debug = []
        self.human_used = 0
        self.llm_used = 0
        self.money_used = 0.0
        self._budget = None

        self.difficulties = None

        self.epsilon_start = kwargs.get("epsilon_start", 0.1)
        self.epsilon_decay = kwargs.get("epsilon_decay", 0.005)
        self.human_threshold = kwargs.get("human_threshold", 0.5)

        self.task_type = None

    @staticmethod
    def _normalize(X: np.ndarray) -> np.ndarray:
        return (X - X.mean(0)) / (X.std(0) + 1e-8)

    def _detect_task_type(self, predictions) -> str:
        if isinstance(predictions[list(predictions.keys())[0]], str):
            return "summarization"
        return "classification"

    def _context(self, t: int, used_mask: np.ndarray) -> np.ndarray:
        base = self.base_features[t]

        total_budget = self.working_budget + self.reserved_budget
        total_used = self.human_used + self.reserved_used

        if self._budget is not None:
            budget_frac = max(0.0, (self._budget - self.money_used) / self._budget)
        else:
            budget_frac = max(0.0, (total_budget - total_used) / max(total_budget, 1))

        progress = t / max(self.n_samples - 1, 1)
        return np.concatenate([base, [budget_frac, progress], used_mask])

    def _estimate_difficulty(self, ctx: np.ndarray) -> float:
        return np.mean([ctx @ self.posterior_cov[a] @ ctx for a in range(self.n_llms)])

    def _select(self, ctx: np.ndarray, available: List[int], t: int = 0) -> int:
        """Select an arm via epsilon-greedy with human escalation for low-quality predictions."""
        available_llms = [a for a in available if a != self.human_arm]
        human_available = self.human_arm in available

        if not available_llms:
            return self.human_arm

        llm_qualities = {}
        for arm in available_llms:
            llm_qualities[arm] = np.clip(self.posterior_mean[arm] @ ctx, 0.0, 1.0)

        best_llm = max(llm_qualities, key=llm_qualities.get)
        best_quality = llm_qualities[best_llm]

        if human_available and best_quality < self.human_threshold:
            return self.human_arm

        epsilon = self.epsilon_start / (1.0 + t * self.epsilon_decay)
        if np.random.random() < epsilon:
            return np.random.choice(available_llms)

        return best_llm

    def _update_posterior(self, ctx: np.ndarray, arm: int, reward: float) -> None:
        """Bayesian linear regression update for the given arm's posterior."""
        x = ctx.reshape(-1, 1)
        S_inv = np.linalg.inv(self.posterior_cov[arm])
        S_new = np.linalg.inv(S_inv + (x @ x.T) / self.noise_var)
        m_new = S_new @ (S_inv @ self.posterior_mean[arm] + (reward / self.noise_var) * ctx)
        self.posterior_cov[arm] = S_new
        self.posterior_mean[arm] = m_new

    def _check_consensus(self, annotations, actions):
        """Return (reached_consensus, majority_label) based on weighted agreement."""
        if len(annotations) < self.min_annotations:
            return False, None

        if self.task_type == "summarization":
            embeddings = self.base_features[:, -1].reshape(-1, 1)
            sim_matrix = cosine_similarity(embeddings)

            n = len(annotations)
            avg_sim = (sim_matrix.sum() - n) / (n * (n - 1))
            return avg_sim >= self.agreement_threshold, None

        weighted = Counter()
        for a, src in zip(annotations, actions):
            weighted[a] += 10 if src == self.human_arm else 1.0

        total = sum(weighted.values())
        best, weight = weighted.most_common(1)[0]
        return (weight / total >= self.agreement_threshold), best

    def _majority(self, annotations, actions):
        """Compute weighted majority vote (classification) or centroid-closest pick (summarization)."""
        if self.task_type == "summarization":
            embeddings = self.base_features[:, -1].reshape(-1, 1)
            centroid = embeddings.mean(axis=0, keepdims=True)
            sims = cosine_similarity(embeddings, centroid).flatten()
            return annotations[int(np.argmax(sims))]

        weighted = Counter()
        for a, src in zip(annotations, actions):
            weighted[a] += 10 if src == self.human_arm else 1.0
        return weighted.most_common(1)[0][0]

    def route(self, llm_predictions, human_labels, **kwargs):
        """Main entry point: dispatches to auditor-style or feedback-based routing."""
        self.task_type = self._detect_task_type(llm_predictions)

        if self.eval_type == "auditor_style":
            return self.route_auditor_style(llm_predictions, human_labels)
        else:
            return self.route_with_feedback(llm_predictions, human_labels, **kwargs)

    def route_with_feedback(
        self,
        llm_predictions: Dict[str, List],
        human_labels: List,
        **kwargs
    ) -> List:
        """Calibration + routing: learn per-LLM quality, then Thompson sampling routes."""
        self.actions, self.annotations_debug = [], []
        self.human_used = self.reserved_used = self.llm_used = 0
        self.money_used = 0.0
        backup = kwargs.get("backup", None)

        self._budget = kwargs.get("money_budget", None)
        arm_to_llm = {i: self.llm_names[i] for i in range(self.n_llms)}

        T = self.n_samples

        self.difficulties = np.array([
            self._estimate_difficulty(self._context(t, np.zeros(self.n_llms)))
            for t in range(T)
        ])

        outputs = [None] * T
        sample_annotations = [[] for _ in range(T)]
        sample_actions = [[] for _ in range(T)]
        used_masks = [np.zeros(self.n_llms) for _ in range(T)]

        cheapest_arm = min(range(self.n_llms), key=lambda i: self.arm_cost[i])

        # --- Calibration phase: human-label most uncertain samples to learn per-LLM accuracy ---
        cal_size = self._compute_cal_size(T)
        uncertainty_scores = self._compute_uncertainty()
        cal_indices = np.argsort(-uncertainty_scores)[:cal_size]

        for t in cal_indices:
            self.human_used += 1
            self.money_used += self.arm_cost[self.human_arm]
            sample_annotations[t].append(human_labels[t])
            sample_actions[t].append(self.human_arm)

            ctx = self._context(t, used_masks[t])
            for arm in range(self.n_llms):
                pred = llm_predictions[arm_to_llm[arm]][t]
                reward = 1.0 if pred == human_labels[t] else 0.0
                self._update_posterior(ctx, arm, reward)

        # --- Routing phase: epsilon-greedy with informed posteriors ---
        uncovered = [t for t in range(T) if len(sample_annotations[t]) == 0]

        n_uncovered = len(uncovered)
        for step_idx, t in enumerate(uncovered):
            if self._budget is not None:
                remaining_budget = self._budget - self.money_used
                remaining_steps = n_uncovered - step_idx
                per_step_budget = remaining_budget / remaining_steps
                available = [a for a in range(self.n_arms)
                             if self.arm_cost[a] <= per_step_budget]
                if not available:
                    if remaining_budget < self.arm_cost[cheapest_arm]:
                        break
                    available = [cheapest_arm]
            else:
                available = list(range(self.n_arms))
                if self.human_used >= self.human_budget:
                    available = [a for a in available if a != self.human_arm]

            ctx = self._context(t, used_masks[t])
            arm = self._select(ctx, available, t)

            if arm == self.human_arm:
                value = human_labels[t]
                self.human_used += 1
            else:
                value = llm_predictions[arm_to_llm[arm]][t]
                used_masks[t][arm] = 1
                self.llm_used += 1

            self.money_used += self.arm_cost[arm]

            sample_annotations[t].append(value)
            sample_actions[t].append(arm)

        # --- Extra annotations: spend remaining budget on additional annotations ---
        while True:
            if self._budget is not None:
                min_cost = min(self.arm_cost[a] for a in range(self.n_arms))
                if self.money_used + min_cost > self._budget:
                    break
            else:
                if self.human_used >= self.human_budget:
                    break

            priorities = self._compute_priorities(sample_annotations, sample_actions, used_masks)
            if not priorities:
                break

            t = priorities[0]
            available = list(range(self.n_arms))

            already_used_llms = [a for a in sample_actions[t] if a != self.human_arm]
            available = [a for a in available if a not in already_used_llms]

            if self._budget is not None:
                available = [a for a in available
                             if self.money_used + self.arm_cost[a] <= self._budget]

            if not available:
                break

            ctx = self._context(t, used_masks[t])
            arm = self._select(ctx, available, t)

            if arm == self.human_arm:
                value = human_labels[t]
                self.human_used += 1
            else:
                value = llm_predictions[arm_to_llm[arm]][t]
                used_masks[t][arm] = 1
                self.llm_used += 1

            self.money_used += self.arm_cost[arm]

            sample_annotations[t].append(value)
            sample_actions[t].append(arm)

        # --- Produce final outputs ---
        for t in range(T):
            annotations = sample_annotations[t]
            actions = sample_actions[t]

            valid_annotations = [a for a in annotations if a is not None]
            valid_actions = [actions[i] for i, a in enumerate(annotations) if a is not None]

            if valid_annotations:
                outputs[t] = self._majority(valid_annotations, valid_actions)
                self.actions.append(valid_actions)
                self.annotations_debug.append(valid_annotations)
            else:
                if backup:
                    outputs[t] = llm_predictions['qwen'][t]
                    self.annotations_debug.append([llm_predictions['qwen'][t]])
                else:
                    try:
                        outputs[t] = human_labels[t] + 1
                        self.annotations_debug.append([human_labels[t] + 1])
                    except TypeError:
                        outputs[t] = "-1"
                        self.annotations_debug.append(["-1"])
                self.actions.append([-1])

        return outputs

    def _compute_cal_size(self, N: int) -> int:
        if self._budget is not None:
            affordable = int(self._budget / self.arm_cost[self.human_arm])
            return max(1, int(0.80 * affordable))
        return max(1, int(0.80 * self.human_budget))

    def _compute_priorities(
        self,
        sample_annotations: List[List],
        sample_actions: List[List],
        used_masks: List[np.ndarray]
    ) -> List[int]:
        """Rank samples by need for additional annotation (difficulty + uncertainty + disagreement)."""
        scores = []
        for t in range(self.n_samples):
            annotations = [a for a in sample_annotations[t] if a is not None]
            actions = sample_actions[t]

            if len(annotations) >= self.max_annotations:
                continue

            n_annotations = len(annotations)
            if n_annotations == 0:
                scores.append((t, float('inf')))
                continue

            consensus, _ = self._check_consensus(annotations, actions)
            if consensus and n_annotations >= max(self.min_annotations, 2):
                continue

            difficulty = self.difficulties[t]
            ctx = self._context(t, used_masks[t])
            uncertainty = np.mean([
                ctx @ self.posterior_cov[a] @ ctx
                for a in range(self.n_llms)
            ])

            has_human = self.human_arm in actions
            disagreement = 0.0
            if self.task_type == "classification" and n_annotations >= 2:
                counts = Counter(annotations)
                disagreement = 1.0 - counts.most_common(1)[0][1] / n_annotations

            score = difficulty + uncertainty + disagreement * 2.0
            if not has_human:
                score += 0.5

            scores.append((t, score))

        scores.sort(key=lambda x: -x[1])
        return [t for t, _ in scores]

    def route_auditor_style(
        self,
        llm_predictions: Dict[str, List],
        human_labels: List
    ) -> List:
        """Two-stage routing: calibrate on random subset, then target error region."""
        self.actions, self.annotations_debug = [], []
        self.human_used = self.llm_used = 0
        self.money_used = 0.0

        T = self.n_samples
        cheapest_arm = min(range(self.n_llms), key=lambda i: self.arm_cost[i])
        cheapest_name = self.llm_names[cheapest_arm]
        preds = np.array(llm_predictions[cheapest_name])

        calibration_size = max(5, int(0.15 * self.human_budget))
        remaining_budget = self.human_budget - calibration_size

        cal_indices = set(np.random.choice(T, calibration_size, replace=False).tolist())
        cal_errors = [t for t in cal_indices if preds[t] != human_labels[t]]

        remaining = [t for t in range(T) if t not in cal_indices]

        if len(cal_errors) >= 2:
            error_centroid = self.base_features[cal_errors].mean(axis=0)
            distances = np.linalg.norm(self.base_features[remaining] - error_centroid, axis=1)
            order = np.argsort(distances)
            remaining_sorted = [remaining[i] for i in order]
        else:
            remaining_sorted = remaining

        human_indices = cal_indices | set(remaining_sorted[:remaining_budget])

        outputs = []
        for t in range(T):
            if t in human_indices:
                value = human_labels[t]
                arm = self.human_arm
                self.human_used += 1
                self.actions.append([1])
            else:
                value = preds[t]
                arm = cheapest_arm
                self.llm_used += 1
                self.actions.append([0])

            self.money_used += self.arm_cost[arm]
            outputs.append(value)
            self.annotations_debug.append([value])

        return outputs

    def _compute_uncertainty(self) -> np.ndarray:
        """Mahalanobis distance from the feature centroid as an uncertainty proxy."""
        X = self.base_features
        mu = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
        diff = X - mu
        return np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
