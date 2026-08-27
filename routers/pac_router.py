import numpy as np
from scipy.stats import norm


def asymptotic_ucb(U, u_array, loss_array, weights, alpha):
    mask0 = (U < u_array[0])
    mask1 = (U >= u_array[0]) & (U < u_array[1])
    mask2 = (U >= u_array[1]) & (U < u_array[2])
    mask3 = (U >= u_array[2])
    
    

    X = (
        loss_array[0] * mask0 +
        loss_array[1] * mask1 +
        loss_array[2] * mask2 +
        loss_array[3] * mask3
    ) * weights

    mean = X.mean()
    std = X.std(ddof=1) if X.size > 1 else 0.0
    z = norm.ppf(1 - alpha)
    return mean + z * std / np.sqrt(X.size)


def compute_cost(cost_array, U, u_array):
    return (
        np.sum(cost_array[0][U <= u_array[0]]) +
        np.sum(cost_array[1][(U > u_array[0]) & (U <= u_array[1])]) +
        np.sum(cost_array[2][(U > u_array[1]) & (U <= u_array[2])]) +
        np.sum(cost_array[3][U > u_array[2]])
    )
    


def get_thresholds(y_array, U, cost_array, epsilon, alpha, pi):
    rng = np.random.default_rng()
    m = y_array.shape[1]

    loss_array = (y_array != y_array[-1]).astype(float)
    phi = (rng.random(m) < pi).astype(float)
    weights = phi / pi

    u_grid = np.sort(np.unique(U))

    best_u = None
    best_cost = np.inf

    for u1 in u_grid:
        for u2 in u_grid[u_grid >= u1]:
            for u3 in u_grid[u_grid >= u2]:
                u_array = np.array([u1, u2, u3])

                ucb = asymptotic_ucb(U, u_array, loss_array, weights, alpha)
                if ucb > epsilon:
                    continue

                cost = compute_cost(cost_array, U, u_array)
                if cost < best_cost:
                    best_cost = cost
                    best_u = u_array
    return best_u if best_u is not None else np.array([0.0, 0.0, 0.0])


class PACRouter:
    def __init__(self, human_budget, **kwargs):
        self.human_budget = human_budget
        self.epsilon = 0.1 
        self.alpha = 0.05
        self.pi = kwargs.get("pi", 0.5)
        self.costs = kwargs['annotator_cost']
        self.eval_type = kwargs["eval_type"]
        self.name = "PAC"

    def uncertainty(self, confidence_scores):
        return 1.0 - confidence_scores

    def build_y_array(self, llm_predictions, human_labels):
        keys = list(llm_predictions.keys())
        return np.vstack([
            llm_predictions[keys[0]],
            llm_predictions[keys[1]],
            llm_predictions[keys[2]],
            human_labels
        ])

    def build_cost_array(self, N):
        return np.vstack([
            np.full(N, self.costs[0]),
            np.full(N, self.costs[1]),
            np.full(N, self.costs[2]),
            np.full(N, self.costs[3])
        ])

    def _ideal_tier(self, u, u_array):
        if u <= u_array[0]:
            return 0
        elif u <= u_array[1]:
            return 1
        elif u <= u_array[2]:
            return 2
        else:
            return 3

    def _compute_cal_size(self, N, money_budget):
        if self.eval_type in ("auditor_style", "human_budget"):
            return max(1, int(0.15 * self.human_budget))
        
        if self.eval_type == "dollars":
            cost_per_cal = self.costs[3]
            affordable = int(money_budget / cost_per_cal)
            return max(1, int(0.15 * affordable))
        
        # fallback
        return max(1, int(0.15 * N))

    def route(self, llm_predictions, human_labels,
              confidence_scores, **kwargs):

        U = self.uncertainty(confidence_scores)
        N = len(U)
        keys = list(llm_predictions.keys())

        money_budget = kwargs.get("money_budget", None)
        backup =  kwargs.get("backup", None)

        self.human_used = 0
        self.llm_used = 0
        self.money_used = sum(kwargs['annotator_cost'][:3]) * len(human_labels)#0.0

        cal_size = self._compute_cal_size(N, money_budget)
        indices = np.random.permutation(N)
        cal_idx = indices[:cal_size]
        test_idx = indices[cal_size:]

        self.human_used += cal_size

        if self.eval_type == "dollars":
            self.money_used += cal_size * self.costs[3]

        y_array = self.build_y_array(llm_predictions, human_labels)
        cost_array = self.build_cost_array(N)

        u_array = get_thresholds(
            y_array[:, cal_idx],
            U[cal_idx],
            cost_array[:, cal_idx],
            epsilon=self.epsilon,
            alpha=self.alpha,
            pi=self.pi
        )

        outputs = [None] * N
        actions = [None] * N

        for i in cal_idx:
            outputs[i] = human_labels[i]
            actions[i] = 1

        test_order = test_idx[np.argsort(-U[test_idx])]

        for i in test_order:
            u = U[i]
            tier = self._ideal_tier(u, u_array)

            if self.eval_type in ("auditor_style", "human_budget"):
                human_left = self.human_budget - self.human_used
                if tier == 3 and human_left <= 0:
                    for fallback in range(2, -1, -1):
                        tier = fallback
                        break

            elif self.eval_type == "dollars":
                money_left = money_budget - self.money_used
                while tier >= 0 and money_left < self.costs[tier]:
                    tier -= 1
                if tier < 0:
                    if backup:
                        outputs[i] = llm_predictions['qwen'][i]
                    else:
                        try:
                            outputs[i] = human_labels[i] + 1
                        except TypeError:
                            outputs[i] = "-1"
                    actions[i] = -1
                    continue

            if tier == 3:
                outputs[i] = human_labels[i]
                actions[i] = 1
                self.human_used += 1
            else:
                outputs[i] = llm_predictions[keys[tier]][i]
                actions[i] = 0
                self.llm_used += 1

            if self.eval_type == "dollars" and tier == 3:
                self.money_used += self.costs[tier]

        self.actions = actions
        return outputs

