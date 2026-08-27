import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import norm

def loss_mean(theta, X, H):
    return H - theta

def grad_mean(theta, X, H):
    return -1.0

def hess_mean(theta, X, H):
    return 0.0

class ConfidenceDrivenInference:

    def __init__(self, loss_fn, grad_fn, hess_fn, target_dim=1):
        self.loss_fn = loss_fn
        self.grad_fn = grad_fn
        self.hess_fn = hess_fn
        self.target_dim = target_dim
        self.error_model = GradientBoostingRegressor(n_estimators=1000,
                                                     learning_rate=0.01,
                                                     max_depth=3)

    def fit_error_model(self, confidences, squared_errors):
        confidences = np.array(confidences).reshape(-1, 1)
        squared_errors = np.array(squared_errors)
        if len(confidences) > 0:
            self.error_model.fit(confidences, squared_errors)

    def predict_error(self, Ci):
        Ci = np.array(Ci).reshape(-1, 1)
        if Ci.size == 0:
            return np.array([])
        preds = self.error_model.predict(Ci)
        preds = np.maximum(preds, 1e-12)
        return preds

    def compute_sampling_probs(self, confidences, nhuman):
        pred_errors = self.predict_error(confidences)
        if pred_errors.size == 0:
            return np.array([])
        stds = np.sqrt(pred_errors)
        n = len(stds)
        if nhuman <= 0:
            return np.zeros(n)
        if nhuman >= n:
            return np.ones(n)
        low, high = 0.0, 1e6
        for _ in range(50):
            mid = 0.5*(low+high)
            pi = np.minimum(1.0, mid*stds)
            if pi.sum() > nhuman:
                high = mid
            else:
                low = mid
        alpha = 0.5*(low+high)
        pi = np.minimum(1.0, alpha*stds)
        pi = np.clip(pi, 1e-8, 1.0)
        return pi

    def compute_theta_conf(self, H_all, Hhat_all, pi_full, xi_full, lambda_value):
        lam = float(lambda_value)
        H_all = np.array(H_all, dtype=float)
        Hhat_all = np.array(Hhat_all, dtype=float)
        pi_full = np.array(pi_full, dtype=float)
        xi_full = np.array(xi_full, dtype=float)
        term = lam * Hhat_all
        debias = (xi_full / pi_full) * (H_all - lam * Hhat_all)
        theta = (term + debias).mean()
        return float(theta)

    def compute_optimal_lambda_value(self, H_warm, Hhat_warm):
        H = np.array(H_warm, dtype=float)
        Hhat = np.array(Hhat_warm, dtype=float)
        if len(H) <= 1:
            return 0.0
        cov = np.cov(Hhat, H, bias=True)[0,1]
        var = np.var(Hhat)
        if var <= 0:
            return 0.0
        lam = cov/var
        return float(np.clip(lam, 0.0, 1.0))

class PPIRouter:

    def __init__(self, human_budget, fixed_budget=False, metric_fn=None):
        """
        Args:
            human_budget: Maximum number of human annotations
            fixed_budget: Whether to use fixed budget allocation
            metric_fn: Optional. For generation tasks, a function that computes numeric score
                      between prediction and reference (e.g., ROUGE, BLEU, embedding similarity)
                      If None and task is detected as generation, will use exact match (0 or 1)
                      Signature: metric_fn(prediction, reference) -> float
        """
        self.human_budget = human_budget
        self.fixed_budget = fixed_budget
        self.metric_fn = metric_fn
        self.actions = []
        self.human_used = 0
        self.llm_used = 0
        self.theta_conf = None
        self.lambda_value = None

        # Will be set automatically
        self.task_type = None
        self.label_to_idx = None
        self.idx_to_label = None

    def _detect_task_type(self, labels):
        """
        Automatically detect if task is classification or generation based on label types.
        Returns: 'classification' if labels are numeric/categorical, 'generation' if text.
        """
        # Sample a few labels to check type
        sample_size = min(10, len(labels))
        sample_labels = labels[:sample_size]

        # Check if labels are numeric (int or float)
        try:
            for label in sample_labels:
                float(label)
            return 'classification'
        except (ValueError, TypeError):
            pass

        # Check if labels are strings with length > 1 (likely text generation)
        if all(isinstance(label, str) and len(label) > 1 for label in sample_labels):
            return 'generation'

        # Default to classification for short strings or mixed types
        return 'classification'

    def _default_generation_metric(self, prediction, reference):
        """Simple exact match metric for generation tasks when no metric_fn provided."""
        return 1.0 if prediction.strip() == reference.strip() else 0.0

    def route(self, llm_predictions, human_labels, confidence_scores, num_samples_to_train=30, **kwargs):
        # Automatically detect task type
        self.task_type = self._detect_task_type(human_labels)

        cdi = ConfidenceDrivenInference(loss_fn=loss_mean,
                                        grad_fn=grad_mean,
                                        hess_fn=hess_mean,
                                        target_dim=1)

        n_total = len(llm_predictions)
        warm_n = min(num_samples_to_train, n_total)
        warm_idx = np.arange(warm_n)
        backup =  kwargs.get("backup", None)

        # Convert labels to numeric format based on detected task type
        if self.task_type == 'classification':
            # Create mapping using ALL labels (both human and LLM predictions)
            all_labels = list(human_labels) + list(llm_predictions)
            unique_labels = sorted(set(all_labels))
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

            numeric_human_labels = np.array([self.label_to_idx[label] for label in human_labels], dtype=float)
            numeric_llm_preds = np.array([self.label_to_idx[pred] for pred in llm_predictions], dtype=float)
        else:  # generation
            numeric_human_labels = None
            numeric_llm_preds = None

        # Warm-up phase
        if self.task_type == 'classification':
            H_warm = [float(numeric_human_labels[i]) for i in warm_idx]
            Hhat_warm = [float(numeric_llm_preds[i]) for i in warm_idx]
        else:  # generation
            metric = self.metric_fn if self.metric_fn is not None else self._default_generation_metric
            H_warm = [1.0 for _ in warm_idx]  # Perfect match with itself
            Hhat_warm = [float(metric(llm_predictions[i], human_labels[i])) for i in warm_idx]

        C_warm = [float(confidence_scores[i]) for i in warm_idx]
        err_warm = [(Hhat_warm[i] - H_warm[i])**2 for i in range(warm_n)]

        if warm_n > 0:
            cdi.fit_error_model(C_warm, err_warm)

        remaining_conf = np.array(confidence_scores[warm_n:], dtype=float)
        remaining_llm = llm_predictions[warm_n:]
        remaining_human = human_labels[warm_n:]

        m = len(remaining_conf)
        nhuman_budget = max(0, self.human_budget - warm_n)

        pi = cdi.compute_sampling_probs(remaining_conf, nhuman=nhuman_budget)

        if m == 0:
            xi = np.array([], dtype=int)
        else:
            if self.fixed_budget:
                take = min(nhuman_budget, m)
                idx_sorted = np.argsort(-pi)
                xi = np.zeros(m, dtype=int)
                xi[idx_sorted[:take]] = 1
            else:
                if kwargs.get('money_budget') is None:
                    rng = np.random.default_rng()
                    xi = rng.binomial(1, pi).astype(int)
                    total = xi.sum()
                    if total > nhuman_budget:
                        ones = np.where(xi == 1)[0]
                        drop = rng.choice(ones, size=total - nhuman_budget, replace=False)
                        xi[drop] = 0
                else:
                    llm_cost = kwargs['annotator_cost'][0]
                    human_cost = kwargs['annotator_cost'][-1]

                    warm_cost = warm_n * human_cost
                    remaining_budget = kwargs['money_budget'] - warm_cost

                    base_cost = m * llm_cost
                    extra_budget = remaining_budget - base_cost
                    extra_cost_per_human = human_cost - llm_cost

                    if extra_cost_per_human > 0:
                        max_human_remaining = int(extra_budget // extra_cost_per_human)
                        max_human_remaining = max(min(max_human_remaining, m), 0)
                    else:
                        max_human_remaining = m

                    rng = np.random.default_rng()
                    xi = rng.binomial(1, pi).astype(int)
                    total = xi.sum()
                    if total > max_human_remaining:
                        ones = np.where(xi == 1)[0]
                        drop = rng.choice(ones, size=total - max_human_remaining, replace=False)
                        xi[drop] = 0

        # Build output set with ORIGINAL label format
        output_set = [human_labels[i] for i in range(warm_n)]
        self.actions = [1]*warm_n
        self.human_used = warm_n
        self.money_used = kwargs['annotator_cost'][2] * len(human_labels)

        for i in range(m):
            if kwargs.get('money_budget') is None:
                if xi[i] == 1 and self.human_used < self.human_budget:
                    output_set.append(remaining_human[i])
                    self.actions.append(1)
                    self.human_used += 1
                else:
                    output_set.append(remaining_llm[i])
                    self.llm_used += 1
                    self.actions.append(0)
            else:
                if self.money_used >= kwargs['money_budget']:
                    print(f"Stopped annotating at sample {i + warm_n}")
                    if backup:
                        output_set.append(remaining_llm[i])
                    else:
                        try:
                            output_set.append(remaining_human[i] + 1)
                        except TypeError:
                            output_set.append("-1")
                    self.actions.append(-1)
                elif xi[i] == 1:
                    output_set.append(remaining_human[i])
                    self.actions.append(1)
                    self.human_used += 1
                    self.money_used += human_cost
                else:
                    output_set.append(remaining_llm[i])
                    self.actions.append(0)
                    self.llm_used += 1
                    self.money_used += llm_cost

        # Compute PPI statistics using numeric representations
        lambda_hat = cdi.compute_optimal_lambda_value(H_warm, Hhat_warm)
        self.lambda_value = lambda_hat

        # For theta computation, we need numeric values
        if self.task_type == 'classification':
            H_all_numeric = np.array([float(numeric_human_labels[i]) for i in range(warm_n)] +
                             [float(numeric_human_labels[warm_n + i]) if xi[i]==1
                              else float(numeric_llm_preds[warm_n + i]) for i in range(m)])
            Hhat_all_numeric = numeric_llm_preds
        else:  # generation
            metric = self.metric_fn if self.metric_fn is not None else self._default_generation_metric
            H_all_numeric = np.array([1.0] * warm_n +
                             [1.0 if xi[i]==1 else float(metric(remaining_llm[i], remaining_human[i]))
                              for i in range(m)])
            Hhat_all_numeric = np.array([float(metric(llm_predictions[i], human_labels[i]))
                                        for i in range(n_total)])

        pi_full = np.concatenate([np.ones(warm_n), pi])
        xi_full = np.concatenate([np.ones(warm_n, dtype=int), xi])

        self.theta_conf = cdi.compute_theta_conf(H_all_numeric, Hhat_all_numeric, pi_full, xi_full, lambda_hat)

        return output_set

