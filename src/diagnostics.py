import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from collections import defaultdict
import os
import pickle
import json


class BanditPlotter:
    """
    Standalone visualization for a QUORUM Bandit Router.

    Usage:
        router.route(...)  # run the router first
        plotter = BanditPlotter(router, llm_predictions, true_labels)
        plotter.plot_all()
    """

    def __init__(self, router, llm_predictions, true_labels,
                 human_labels=None, eval_type=None):
        """
        Args:
            router:           the wrapper object (router.router gives the inner bandit)
            llm_predictions:  dict {llm_name: np.array of predictions}
            true_labels:      np.array of ground truth
            human_labels:     np.array of human annotations (optional)
        """
        self.router = router
        self.inner = router.router  # the actual bandit object
        self.llm_predictions = llm_predictions
        self.true_labels = np.array(true_labels)
        self.human_labels = np.array(human_labels) if human_labels is not None else None
        self.n_samples = len(true_labels)
        self.eval_type = eval_type

        self._precompute()

    def _precompute(self):
        """Extract and precompute everything we need from the router."""
        inner = self.inner
        self.arm_names = inner.llm_names + ["human"]
        self.n_arms = inner.n_arms

        # Per-sample outputs and correctness
        self.outputs = []
        self.correct = []
        self.per_sample_actions = []  
        self.flat_actions = []       

        assert len(inner.annotations_debug) == len(inner.actions), "annotations and actions do not share the same length."
        
        for t, (anns, acts) in enumerate(
            zip(inner.annotations_debug, inner.actions)
        ):
            result = inner._majority(anns, acts)
            self.outputs.append(result)
            self.correct.append(int(result == self.true_labels[t]))
            self.per_sample_actions.append(acts)
            self.flat_actions.extend(acts)

        self.correct = np.array(self.correct)
        self.flat_actions = np.array(self.flat_actions)

        self.arm_correct = {i: [] for i in range(self.n_arms)}
        self.arm_correct[-1] = []
        self.arm_timestamps = {i: [] for i in range(self.n_arms)}
        self.arm_timestamps[-1] = []
        flat_t = 0
        for t, (anns, acts) in enumerate(
            zip(inner.annotations_debug, inner.actions)
        ):
            for ann, act in zip(anns, acts):
                is_correct = int(ann == self.true_labels[t])
                self.arm_correct[act].append(is_correct)
                self.arm_timestamps[act].append(flat_t)
                flat_t += 1

        # Cumulative reward (approximate from correctness)
        self.cumulative_correct = np.cumsum(self.correct)

        # Colors
        self.arm_colors = plt.cm.Set2(np.linspace(0, 1, self.n_arms))
        self.human_arm_idx = self.n_arms - 1

    def save_plot_data(self, save_path=None) -> str:
        """
        Save all data necessary to regenerate plots later.
        Uses pickle to preserve numpy arrays and complex structures.
        """
        # Extract weights before saving
        thetas = self._extract_weights()
        n_features = len(thetas[0]) if thetas and len(thetas) > 0 else 0
        
        plot_data = {
            # Metadata
            'n_samples': self.n_samples,
            'n_arms': self.n_arms,
            'arm_names': self.arm_names,

            # Predictions and labels
            'llm_predictions': self.llm_predictions,
            'true_labels': self.true_labels.tolist(),
            'human_labels': self.human_labels.tolist() if self.human_labels is not None else None,
            'outputs': self.outputs,

            # Correctness tracking
            'correct': self.correct.tolist(),
            'cumulative_correct': self.cumulative_correct.tolist(),

            # Actions
            'per_sample_actions': self.per_sample_actions,
            'flat_actions': self.flat_actions.tolist(),

            # Per-arm data
            'arm_correct': {k: v for k, v in self.arm_correct.items()},
            'arm_timestamps': {k: v for k, v in self.arm_timestamps.items()},

            # Router internals (if available)
            'thetas': [t.tolist() if isinstance(t, np.ndarray) else t for t in thetas] if thetas else None,
            'feature_names': self._get_feature_names(n_features) if thetas else None,

            # Base features (for human trigger analysis)
            'base_features': self.inner.base_features.tolist() if hasattr(self.inner, 'base_features') else None,
        }

        # Save with pickle
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, f"plot_data_{self.eval_type}.pkl")
        with open(full_path, 'wb') as f:
            pickle.dump(plot_data, f)

        return full_path

    @classmethod
    def load_from_data(cls, data_path, save_dir="plots"):
        """
        Load saved data and create a plotter without needing the original router.

        Usage:
            plotter = BanditPlotter.load_from_data("plots/plot_data.pkl")
            plotter.plot_dashboard()
        """
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        # Create a "mock" plotter object with saved data
        plotter = object.__new__(cls)
        plotter.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Restore all attributes
        plotter.n_samples = data['n_samples']
        plotter.n_arms = data['n_arms']
        plotter.arm_names = data['arm_names']
        plotter.llm_predictions = data['llm_predictions']
        plotter.true_labels = np.array(data['true_labels'])
        plotter.human_labels = np.array(data['human_labels']) if data['human_labels'] else None
        plotter.outputs = data['outputs']
        plotter.correct = np.array(data['correct'])
        plotter.cumulative_correct = np.array(data['cumulative_correct'])
        plotter.per_sample_actions = data['per_sample_actions']
        plotter.flat_actions = np.array(data['flat_actions'])
        plotter.arm_correct = data['arm_correct']
        plotter.arm_timestamps = data['arm_timestamps']

        # Colors
        plotter.arm_colors = plt.cm.Set2(np.linspace(0, 1, plotter.n_arms))
        plotter.human_arm_idx = plotter.n_arms - 1

        # Mock router object (for compatibility with methods that use it)
        class MockRouter:
            pass

        plotter.router = MockRouter()
        plotter.inner = MockRouter()
        plotter.inner.llm_names = plotter.arm_names[:-1]
        plotter.inner.n_arms = plotter.n_arms

        # Restore features and thetas if available
        if data.get('base_features'):
            plotter.inner.base_features = np.array(data['base_features'])

        if data.get('thetas'):
            plotter._cached_thetas = [np.array(t) for t in data['thetas']]
            plotter._cached_feature_names = data.get('feature_names')

        print(f"✅ Loaded plot data from {data_path}")
        return plotter


    def _extract_weights(self):
        """
        Extract weight vectors from the router, handling different model types:
        - Thompson Sampling: posterior_mean
        - LinUCB: A^{-1} @ b
        - Sklearn QUORUM: model.coef_
        - Per-arm sklearn models: models[i].coef_
        """
        # Use cached thetas if available (for loaded data)
        if hasattr(self, '_cached_thetas') and self._cached_thetas:
            return self._cached_thetas

        inner = self.inner
        thetas = []
        source = "unknown"

        # Case 1: Thompson Sampling (posterior_mean)
        if hasattr(inner, 'posterior_mean') and inner.posterior_mean is not None:
            for arm_idx in range(self.n_arms):
                theta = np.array(inner.posterior_mean[arm_idx]).flatten()
                thetas.append(theta)
            source = "posterior_mean (Thompson Sampling)"

        # Case 2: LinUCB (A, b matrices)
        elif hasattr(inner, 'A') and hasattr(inner, 'b'):
            for arm_idx in range(self.n_arms):
                try:
                    A_inv = np.linalg.inv(inner.A[arm_idx])
                    theta = (A_inv @ inner.b[arm_idx]).flatten()
                    thetas.append(theta)
                except Exception as e:
                    print(f"  ⚠️ Error extracting theta for arm {arm_idx}: {e}")
            source = "A^{-1} @ b (LinUCB)"

        # Case 3: Single sklearn QUORUM model
        elif hasattr(inner, 'model') and hasattr(inner.model, 'coef_'):
            thetas.append(inner.model.coef_.flatten())
            source = "model.coef_ (sklearn)"

        # Case 4: Per-arm sklearn models
        elif hasattr(inner, 'models'):
            for arm_idx in range(self.n_arms):
                model = inner.models[arm_idx]
                if hasattr(model, 'coef_'):
                    thetas.append(model.coef_.flatten())
                elif hasattr(model, 'weights'):
                    thetas.append(np.array(model.weights).flatten())
            source = "per-arm models"

        # Case 5: Raw weights/theta attribute
        elif hasattr(inner, 'weights'):
            w = np.array(inner.weights)
            if w.ndim == 2:
                for row in w:
                    thetas.append(row.flatten())
            else:
                thetas.append(w.flatten())
            source = "weights attribute"

        elif hasattr(inner, 'theta'):
            t = np.array(inner.theta)
            if t.ndim == 2:
                for row in t:
                    thetas.append(row.flatten())
            else:
                thetas.append(t.flatten())
            source = "theta attribute"

        if not thetas:
            print("  ⚠️ Could not extract weights from router!")
            print(f"     Available attributes: "
                  f"{[a for a in dir(inner) if not a.startswith('_')]}")
            return None

        # Sanity check
        all_zero = all(np.allclose(t, 0) for t in thetas)
        if all_zero:
            print(f"  ⚠️ All thetas are zero! (source: {source})")
            print(f"     The model may not have learned anything.")
        else:
            max_norm = max(np.linalg.norm(t) for t in thetas)

        return thetas

    def _get_feature_names(self, n_features):
        """Generate feature names matching the router's feature structure."""
        # Use cached feature names if available (for loaded data)
        if hasattr(self, '_cached_feature_names') and self._cached_feature_names:
            return self._cached_feature_names[:n_features]

        inner = self.inner

        # If the router provides names, use them
        if hasattr(inner, 'feature_names') and inner.feature_names is not None:
            return inner.feature_names[:n_features]

        names = []

        # Base features: linguistic + embedding
        n_ling = getattr(inner, 'n_ling', 6)
        n_emb = getattr(inner, 'n_emb', 1)

        ling_names = [
            "text_length", "avg_word_len", "vocab_richness",
            "punct_density", "digit_density", "uppercase_ratio"
        ]
        for i in range(min(n_ling, n_features)):
            name = ling_names[i] if i < len(ling_names) else f"ling_{i}"
            names.append(name)

        for i in range(min(n_emb, max(0, n_features - len(names)))):
            names.append(f"emb_{i}")

        # Dynamic features
        dynamic_names = [
            "n_ann_ratio",       # n_ann / max_annotations
            "human_ratio",       # n_human / n_ann
            "llm_ratio",         # n_llm / n_ann
            "variance",          # annotation variance
            "last_action",       # last action taken
            "budget_fraction",   # remaining budget
            "progress",          # t / T
        ]

        for name in dynamic_names:
            if len(names) < n_features:
                names.append(name)

        # Per-LLM mask features
        llm_names = getattr(inner, 'llm_names', [])
        for llm_name in llm_names:
            if len(names) < n_features:
                names.append(f"used_{llm_name}")

        # Pad if needed
        while len(names) < n_features:
            names.append(f"feat_{len(names)}")

        return names[:n_features]


    def plot_cumulative_accuracy(self, ax=None, show=False):
        """Cumulative accuracy over time — is the router improving?"""
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(10, 5))

        steps = np.arange(1, len(self.correct) + 1)
        cum_acc = self.cumulative_correct / steps

        ax.plot(steps, cum_acc, color='#2196F3', linewidth=1.8, label='Router')
        ax.fill_between(steps, 0, cum_acc, alpha=0.1, color='#2196F3')

        # Baseline: best single LLM
        best_acc = 0
        best_name = ""
        for llm_key, preds in self.llm_predictions.items():
            acc = (np.array(preds) == self.true_labels).mean()
            if acc > best_acc:
                best_acc = acc
                best_name = llm_key
        ax.axhline(y=best_acc, color='red', linestyle='--', alpha=0.6,
                    label=f'Best LLM ({best_name}: {best_acc:.3f})')

        ax.set_xlabel("Sample")
        ax.set_ylabel("Cumulative Accuracy")
        ax.set_title("Cumulative Accuracy Over Time")
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        if standalone and show:
            plt.tight_layout()
            plt.savefig("cumulative_accuracy.png",
                        dpi=150)
            plt.show()
        return ax

    def plot_rolling_accuracy(self, window=200, ax=None, show=False):
        """Rolling window accuracy — shows learning dynamics."""
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(10, 5))

        if len(self.correct) < window:
            window = max(10, len(self.correct) // 5)

        rolling = np.convolve(
            self.correct.astype(float),
            np.ones(window) / window,
            mode='valid'
        )
        x = np.arange(window, len(self.correct) + 1)
        ax.plot(x, rolling, color='#2196F3', linewidth=2, label='Router')

        # Per-arm rolling accuracy
        for arm_idx in range(self.n_arms):
            vals = self.arm_correct[arm_idx]
            if len(vals) >= window:
                arm_rolling = np.convolve(
                    np.array(vals, dtype=float),
                    np.ones(window) / window,
                    mode='valid'
                )
                timestamps = np.array(self.arm_timestamps[arm_idx])
                ax.plot(
                    timestamps[window - 1:], arm_rolling,
                    color=self.arm_colors[arm_idx],
                    linewidth=1.2, alpha=0.7,
                    label=self.arm_names[arm_idx]
                )

        ax.set_xlabel("Step")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Rolling Accuracy (window={window})")
        ax.legend(loc='best', fontsize=18)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        if standalone and show:
            plt.tight_layout()
            plt.savefig("rolling_accuracy.png",
                        dpi=150)
            plt.show()
        return ax

    def plot_arm_selection_over_time(self, window=200, ax=None, show=False):
        """Stacked area: proportion of each arm selected over time."""
        cost = {"qwen" : "0.01$ LLM",
                "nova_pro" : "0.03$ LLM",
                "claude" : "0.05$ LLM",
                "human" : "human"}
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(12, 5))

        # Build per-sample arm counts
        arm_series = {i: [] for i in range(self.n_arms)}
        for acts in self.per_sample_actions:
            counts = defaultdict(int)
            for a in acts:
                counts[a] += 1
            total = len(acts)
            for i in range(self.n_arms):
                arm_series[i].append(counts[i] / total)

        # Smooth with rolling window
        if len(arm_series[0]) < window:
            window = max(5, len(arm_series[0]) // 5)

        smoothed = {}
        for i in range(self.n_arms):
            smoothed[i] = np.convolve(
                np.array(arm_series[i]),
                np.ones(window) / window,
                mode='valid'
            )

        x = np.arange(window, len(arm_series[0]) + 1)
        bottom = np.zeros(len(x))

        for i in range(self.n_arms):
            ax.fill_between(
                x, bottom, bottom + smoothed[i],
                alpha=0.6, color=self.arm_colors[i],
                label=cost[self.arm_names[i]]
            )
            bottom += smoothed[i]

        ax.set_xlabel("Sample")
        ax.set_ylabel("Proportion")
        ax.set_title(f"Arm Selection Proportions")
        ax.legend(loc='upper center', fontsize=16, ncol=2, bbox_to_anchor=(0.5, -0.17))
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        if standalone and show:
            plt.tight_layout()
            #plt.subplots_adjust(bottom=0.17)
            plt.savefig("arm_selection_over_time.png", dpi=150)
            plt.show()
        return ax

    def plot_arm_usage_bar(self, ax=None, show=False):
        """Bar chart: total arm usage + accuracy per arm."""
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(10, 5))

        counts = []
        accs = []
        for i in range(self.n_arms):
            c = len(self.arm_correct[i])
            counts.append(c)
            accs.append(np.mean(self.arm_correct[i]) if c > 0 else 0)

        x_pos = np.arange(self.n_arms)
        width = 0.35

        ax.bar(x_pos - width / 2, counts, width,
               color=[self.arm_colors[i] for i in range(self.n_arms)],
               alpha=0.7, label='# Selections')

        ax2 = ax.twinx()
        ax2.bar(x_pos + width / 2, accs, width,
                color=[self.arm_colors[i] for i in range(self.n_arms)],
                alpha=0.4, edgecolor='black', linewidth=1.5,
                label='Accuracy')

        ax.set_xlabel("Arm")
        ax.set_ylabel("Number of Selections")
        ax2.set_ylabel("Accuracy")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(self.arm_names, rotation=45, ha='right')
        ax.set_title("Arm Usage & Accuracy")

        handles = [
            Patch(facecolor='gray', alpha=0.7, label='# Selections'),
            Patch(facecolor='gray', alpha=0.4, edgecolor='black',
                  linewidth=1.5, label='Accuracy')
        ]
        ax.legend(handles=handles, loc='upper left')
        ax2.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')

        if standalone and show:
            plt.tight_layout()
            plt.savefig("arm_usage_bar.png",
                        dpi=150)
            plt.show()
        return ax

    def plot_arm_usage_shift(self, ax=None, show=False):
        """Compare arm usage: first half vs second half."""
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(10, 5))

        mid = len(self.per_sample_actions) // 2
        first_half = defaultdict(int)
        second_half = defaultdict(int)

        for acts in self.per_sample_actions[:mid]:
            for a in acts:
                first_half[a] += 1
        for acts in self.per_sample_actions[mid:]:
            for a in acts:
                second_half[a] += 1

        total_first = sum(first_half.values())
        total_second = sum(second_half.values())

        x_pos = np.arange(self.n_arms)
        width = 0.35

        pcts_first = [first_half[i] / max(total_first, 1) * 100
                      for i in range(self.n_arms)]
        pcts_second = [second_half[i] / max(total_second, 1) * 100
                       for i in range(self.n_arms)]

        ax.bar(x_pos - width / 2, pcts_first, width,
               color='#90CAF9', edgecolor='#1565C0', label='First half')
        ax.bar(x_pos + width / 2, pcts_second, width,
               color='#FFB74D', edgecolor='#E65100', label='Second half')

        for i in range(self.n_arms):
            shift = pcts_second[i] - pcts_first[i]
            y = max(pcts_first[i], pcts_second[i]) + 1
            symbol = "↑" if shift > 1 else "↓" if shift < -1 else "→"
            ax.text(i, y, f"{symbol}{shift:+.1f}%", ha='center', fontsize=18,
                    fontweight='bold',
                    color='green' if shift > 1 else 'red' if shift < -1
                    else 'gray')

        ax.set_xlabel("Arm")
        ax.set_ylabel("Usage %")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(self.arm_names, rotation=45, ha='right')
        ax.set_title("Arm Usage Shift: First Half → Second Half")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        if standalone and show:
            plt.tight_layout()
            plt.savefig("arm_usage_shift.png",
                        dpi=150)
            plt.show()
        return ax

    def plot_routing_timeline(self, max_samples=500, ax=None, show=False):
        """Scatter: each sample colored by arm chosen, y = correct/wrong."""
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(14, 4))

        n_show = min(max_samples, self.n_samples)

        for t in range(n_show):
            acts = self.per_sample_actions[t]
            arm = (max(set(acts), key=acts.count)
                   if isinstance(acts, list) else acts)
            color = self.arm_colors[arm]
            marker = 'o' if self.correct[t] else 'x'
            alpha = 0.7 if self.correct[t] else 0.4

            ax.scatter(t, self.correct[t] + np.random.uniform(-0.1, 0.1),
                       c=[color], marker=marker, s=15, alpha=alpha)

        handles = [
            Patch(facecolor=self.arm_colors[i], label=self.arm_names[i])
            for i in range(self.n_arms)
        ]
        handles.append(plt.Line2D([0], [0], marker='o', color='gray',
                                  label='Correct', markersize=6,
                                  linestyle='None'))
        handles.append(plt.Line2D([0], [0], marker='x', color='gray',
                                  label='Wrong', markersize=6,
                                  linestyle='None'))

        ax.set_xlabel("Sample")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Wrong", "Correct"])
        ax.set_title(f"Routing Timeline (first {n_show} samples)")
        ax.legend(handles=handles, loc='upper right', fontsize=18, ncol=2)
        ax.grid(True, alpha=0.3, axis='x')

        if standalone and show:
            plt.tight_layout()
            plt.savefig("routing_timeline.png",
                        dpi=150)
            plt.show()
        return ax


    def plot_theta_norms(self, ax=None, show=False):
        """Bar chart of theta (weight) norms per arm."""
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(10, 5))

        thetas = self._extract_weights()

        if thetas is None or len(thetas) == 0:
            ax.text(0.5, 0.5, "No weights available",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=18, color='red')
            ax.set_title("Weight Norm per Arm (NO DATA)")
            if standalone and show:
                plt.tight_layout()
                plt.show()
            return ax

        norms = [np.linalg.norm(t) for t in thetas]
        n_bars = len(norms)

        if n_bars == self.n_arms:
            labels = self.arm_names
            colors = [self.arm_colors[i] for i in range(n_bars)]
        else:
            labels = [f"Component_{i}" for i in range(n_bars)]
            colors = plt.cm.Set2(np.linspace(0, 1, n_bars))

        x_pos = np.arange(n_bars)
        bars = ax.bar(x_pos, norms, color=colors, edgecolor='black', alpha=0.7)

        for bar, norm_val in zip(bars, norms):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(norms) * 0.02,
                    f'{norm_val:.4f}', ha='center', fontsize=18)

        ax.set_xlabel("Arm")
        ax.set_ylabel("||θ||")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_title("Weight Norm per Arm (higher = more learned)")
        ax.grid(True, alpha=0.3, axis='y')

        if standalone and show:
            plt.tight_layout()
            plt.savefig("theta_norms.png",
                        dpi=150)
            plt.show()
        return ax

    def plot_theta_heatmap(self, ax=None, show=False):
        """Heatmap of theta vectors across all arms."""
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(12, 5))

        thetas = self._extract_weights()

        if thetas is None or len(thetas) == 0:
            ax.text(0.5, 0.5, "No weights available",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=18, color='red')
            ax.set_title("θ Heatmap (NO DATA)")
            if standalone and show:
                plt.tight_layout()
                plt.show()
            return ax

        # Ensure same length
        min_len = min(len(t) for t in thetas)
        theta_matrix = np.array([t[:min_len] for t in thetas])
        n_features = theta_matrix.shape[1]

        feature_names = self._get_feature_names(n_features)

        if len(thetas) == self.n_arms:
            y_labels = self.arm_names
        else:
            y_labels = [f"Component_{i}" for i in range(len(thetas))]

        vmax = np.abs(theta_matrix).max()
        if vmax < 1e-10:
            vmax = 1.0

        im = ax.imshow(theta_matrix, aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax)

        ax.set_xticks(range(n_features))
        ax.set_xticklabels(feature_names, rotation=90, fontsize=18)
        ax.set_yticks(range(len(thetas)))
        ax.set_yticklabels(y_labels)
        ax.set_title("θ Heatmap (blue=negative, red=positive)")
        plt.colorbar(im, ax=ax, shrink=0.8)

        if standalone and show:
            plt.tight_layout()
            plt.savefig("theta_heatmap.png",
                        dpi=150)
            plt.show()
        return ax
        
    def plot_dashboard(self, window=200, cost_per_human=0.02,
                       save_name="dashboard.png"):
        """All plots in one comprehensive dashboard."""

        fig = plt.figure(figsize=(24, 20))
        fig.suptitle("Bandit Router — Diagnostic Dashboard",
                     fontsize=18, y=0.98, fontweight='bold')

        gs = gridspec.GridSpec(4, 3, hspace=0.4, wspace=0.35)

        # Row 1: Performance over time
        self.plot_cumulative_accuracy(ax=fig.add_subplot(gs[0, 0]))
        self.plot_rolling_accuracy(window=window,
                                   ax=fig.add_subplot(gs[0, 1]))
        self.plot_oracle_gap(window=window,
                             ax=fig.add_subplot(gs[0, 2]))

        # Row 2: Arm behavior
        self.plot_arm_selection_over_time(window=window,
                                          ax=fig.add_subplot(gs[1, 0]))
        self.plot_arm_usage_bar(ax=fig.add_subplot(gs[1, 1]))
        self.plot_arm_usage_shift(ax=fig.add_subplot(gs[1, 2]))

        # Row 3: Model internals
        self.plot_theta_heatmap(ax=fig.add_subplot(gs[2, 0]))
        self.plot_feature_importance(ax=fig.add_subplot(gs[2, 1]))
        self.plot_theta_norms(ax=fig.add_subplot(gs[2, 2]))

        # Row 4: Cost & decisions
        self.plot_cost_vs_accuracy_curve(
            cost_per_human=cost_per_human,
            ax=fig.add_subplot(gs[3, 0]))
        self.plot_human_trigger_analysis(ax=fig.add_subplot(gs[3, 1]))
        self.plot_reward_per_arm_over_time(
            window=window, ax=fig.add_subplot(gs[3, 2]))

        os.makedirs(save_name, exist_ok=True)
        path = os.path.join(save_name, f"dashboard_{self.eval_type}.pdf")
        plt.savefig(path, format='pdf', bbox_inches='tight')
        plt.close()
        
    def plot_oracle_gap(self, window=200, ax=None, show=False):
        """Compare router accuracy vs oracle (best per-sample) over time."""
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(10, 5))

        llm_keys = list(self.llm_predictions.keys())

        oracle_correct = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            for llm_key in llm_keys:
                if self.llm_predictions[llm_key][i] == self.true_labels[i]:
                    oracle_correct[i] = 1
                    break

        if len(self.correct) < window:
            window = max(5, len(self.correct) // 5)

        router_rolling = np.convolve(
            self.correct.astype(float),
            np.ones(window) / window, mode='valid'
        )
        oracle_rolling = np.convolve(
            oracle_correct.astype(float),
            np.ones(window) / window, mode='valid'
        )

        best_llm_correct = np.zeros(self.n_samples)
        best_acc = 0
        best_name = ""
        for llm_key in llm_keys:
            preds = np.array(self.llm_predictions[llm_key])
            acc = (preds == self.true_labels).mean()
            if acc > best_acc:
                best_acc = acc
                best_name = llm_key
                best_llm_correct = (preds == self.true_labels).astype(float)

        best_rolling = np.convolve(
            best_llm_correct,
            np.ones(window) / window, mode='valid'
        )

        x = np.arange(window, self.n_samples + 1)

        ax.plot(x, oracle_rolling, color='gold', linewidth=2,
                label='Oracle (best per-sample)', linestyle='--')
        ax.plot(x, router_rolling, color='#2196F3', linewidth=2,
                label='Router')
        ax.plot(x, best_rolling, color='red', linewidth=1.5,
                label=f'Best LLM ({best_name})', alpha=0.7)

        ax.fill_between(x, router_rolling, oracle_rolling,
                        alpha=0.1, color='gold', label='Gap to oracle')

        ax.set_xlabel("Sample")
        ax.set_ylabel("Rolling Accuracy")
        ax.set_title(f"Router vs Oracle vs Best LLM (window={window})")
        ax.legend(loc='lower right', fontsize=18)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        if standalone and show:
            plt.tight_layout()
            plt.savefig( "oracle_gap.png",
                        dpi=150)
            plt.show()
        return ax
    
    def plot_cost_vs_accuracy_curve(self, cost_per_human=0.02, ax=None,
                                     show=False):
        """Cumulative cost vs cumulative accuracy tradeoff."""
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(10, 5))

        costs = []
        for acts in self.per_sample_actions:
            n_human = sum(1 for a in acts if a == self.human_arm_idx)
            costs.append(n_human * cost_per_human)

        cum_cost = np.cumsum(costs)
        steps = np.arange(1, len(self.correct) + 1)
        cum_acc = self.cumulative_correct / steps

        scatter = ax.scatter(cum_cost, cum_acc,
                             c=steps, cmap='viridis', s=8, alpha=0.5)
        plt.colorbar(scatter, ax=ax, label='Sample #')

        ax.set_xlabel("Cumulative Cost ($)")
        ax.set_ylabel("Cumulative Accuracy")
        ax.set_title("Cost vs Accuracy Tradeoff Over Time")
        ax.grid(True, alpha=0.3)

        if standalone and show:
            plt.tight_layout()
            plt.savefig("cost_vs_accuracy.png", dpi=150)
            plt.show()
        return ax

    def plot_human_trigger_analysis(self, ax=None, show=False):
        """When does the router call the human? Analyze difficulty."""
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(10, 5))

        human_idx = self.human_arm_idx
        features = self.inner.base_features

        human_samples = []
        llm_samples = []
        for t, acts in enumerate(self.per_sample_actions):
            if human_idx in acts:
                human_samples.append(t)
            else:
                llm_samples.append(t)

        if features.ndim == 2:
            difficulty = features.mean(axis=1)
        else:
            difficulty = features

        if len(human_samples) > 0 and len(llm_samples) > 0:
            ax.hist(difficulty[llm_samples], bins=40, alpha=0.6,
                    color='#4CAF50', label='LLM-only', density=True)
            ax.hist(difficulty[human_samples], bins=40, alpha=0.6,
                    color='#FF9800', label='Human involved', density=True)
        else:
            ax.hist(difficulty, bins=40, alpha=0.6, color='#2196F3',
                    label='All samples', density=True)

        ax.set_xlabel("Mean Feature Value (difficulty proxy)")
        ax.set_ylabel("Density")
        ax.set_title("When Does the Router Call a Human?")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if standalone and show:
            plt.tight_layout()
            plt.savefig("human_trigger.png", dpi=150)
            plt.show()
        return ax

    def plot_oracle_gap(self, window=200, ax=None, show=False):
        """Compare router accuracy vs oracle (best per-sample) over time."""
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(10, 5))

        llm_keys = list(self.llm_predictions.keys())

        oracle_correct = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            for llm_key in llm_keys:
                if self.llm_predictions[llm_key][i] == self.true_labels[i]:
                    oracle_correct[i] = 1
                    break

        if len(self.correct) < window:
            window = max(5, len(self.correct) // 5)

        router_rolling = np.convolve(
            self.correct.astype(float),
            np.ones(window) / window, mode='valid'
        )
        oracle_rolling = np.convolve(
            oracle_correct.astype(float),
            np.ones(window) / window, mode='valid'
        )

        best_llm_correct = np.zeros(self.n_samples)
        best_acc = 0
        best_name = ""
        for llm_key in llm_keys:
            preds = np.array(self.llm_predictions[llm_key])
            acc = (preds == self.true_labels).mean()
            if acc > best_acc:
                best_acc = acc
                best_name = llm_key
                best_llm_correct = (preds == self.true_labels).astype(float)

        best_rolling = np.convolve(
            best_llm_correct,
            np.ones(window) / window, mode='valid'
        )

        x = np.arange(window, self.n_samples + 1)

        ax.plot(x, oracle_rolling, color='gold', linewidth=2,
                label='Oracle (best per-sample)', linestyle='--')
        ax.plot(x, router_rolling, color='#2196F3', linewidth=2,
                label='Router')
        ax.plot(x, best_rolling, color='red', linewidth=1.5,
                label=f'Best LLM ({best_name})', alpha=0.7)

        ax.fill_between(x, router_rolling, oracle_rolling,
                        alpha=0.1, color='gold', label='Gap to oracle')

        ax.set_xlabel("Sample")
        ax.set_ylabel("Rolling Accuracy")
        ax.set_title(f"Router vs Oracle vs Best LLM (window={window})")
        ax.legend(loc='lower right', fontsize=18)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        if standalone and show:
            plt.tight_layout()
            plt.savefig("oracle_gap.png",
                        dpi=150)
            plt.show()
        return ax

    def plot_reward_per_arm_over_time(self, window=100, ax=None, show=False):
        """Rolling average reward per arm over time."""
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(10, 5))

        for arm_idx in range(self.n_arms):
            vals = np.array(self.arm_correct[arm_idx], dtype=float)
            timestamps = np.array(self.arm_timestamps[arm_idx])

            if len(vals) >= window:
                rolling = np.convolve(vals, np.ones(window) / window,
                                      mode='valid')
                ax.plot(timestamps[window - 1:], rolling,
                        color=self.arm_colors[arm_idx],
                        linewidth=1.5, label=self.arm_names[arm_idx])

        ax.set_xlabel("Global Step")
        ax.set_ylabel("Rolling Reward")
        ax.set_title(f"Per-Arm Rolling Reward (window={window})")
        ax.legend(loc='best', fontsize=18)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        if standalone and show:
            plt.tight_layout()
            plt.savefig("reward_per_arm.png", dpi=150)
            plt.show()
        return ax

    def plot_reward_per_arm_over_time(self, window=100, ax=None, show=False):
        """Rolling average reward per arm over time."""
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(10, 5))

        for arm_idx in range(self.n_arms):
            vals = np.array(self.arm_correct[arm_idx], dtype=float)
            timestamps = np.array(self.arm_timestamps[arm_idx])

            if len(vals) >= window:
                rolling = np.convolve(vals, np.ones(window) / window,
                                      mode='valid')
                ax.plot(timestamps[window - 1:], rolling,
                        color=self.arm_colors[arm_idx],
                        linewidth=1.5, label=self.arm_names[arm_idx])

        ax.set_xlabel("Global Step")
        ax.set_ylabel("Rolling Reward")
        ax.set_title(f"Per-Arm Rolling Reward (window={window})")
        ax.legend(loc='best', fontsize=18)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        if standalone and show:
            plt.tight_layout()
            plt.savefig("reward_per_arm.png", dpi=150)
            plt.show()
        return ax
    
    def plot_feature_importance(self, ax=None, show=False):
        """Average absolute theta across arms = feature importance."""
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(10, 5))

        thetas = self._extract_weights()

        if thetas is None or len(thetas) == 0:
            ax.text(0.5, 0.5, "No weights available",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=16, color='red')
            ax.set_title("Feature Importance (NO DATA)")
            if standalone and show:
                plt.tight_layout()
                plt.show()
            return ax

        # Ensure same length
        min_len = min(len(t) for t in thetas)
        thetas_trimmed = [np.abs(t[:min_len]) for t in thetas]

        avg_importance = np.mean(thetas_trimmed, axis=0)
        n_features = len(avg_importance)

        if n_features == 0:
            ax.text(0.5, 0.5, "Zero features found",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=16, color='red')
            if standalone and show:
                plt.tight_layout()
                plt.show()
            return ax

        feature_names = self._get_feature_names(n_features)
        sorted_idx = np.argsort(avg_importance)[::-1]

        # Color intensity by importance
        max_imp = max(avg_importance.max(), 1e-10)
        colors = plt.cm.Blues(avg_importance[sorted_idx] / max_imp)

        ax.barh(
            range(n_features),
            avg_importance[sorted_idx],
            color=colors, edgecolor='#1565C0', alpha=0.8
        )

        # Value labels
        for i, idx in enumerate(sorted_idx):
            val = avg_importance[idx]
            ax.text(val + max_imp * 0.02, i,
                    f'{val:.4f}', va='center', fontsize=16)

        ax.set_yticks(range(n_features))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx],
                           fontsize=16)
        ax.set_xlabel("Mean |θ| across arms")
        ax.set_title("Feature Importance (from bandit weights)")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        if standalone and show:
            plt.tight_layout()
            plt.savefig("feature_importance.png", dpi=150)
            plt.show()
        return ax
        
    def plot_all(self, window=200, cost_per_human=0.02, save_path=None):
        self.plot_dashboard(window=window, cost_per_human=cost_per_human, save_name=save_path)
        if save_path is not None:
            self.save_plot_data(save_path=save_path)


