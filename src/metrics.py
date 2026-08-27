import evaluate
import numpy as np
import os
import json
from scipy.stats import norm
from os.path import exists

class Metrics:
    def __init__(self, predictions, labels, task, basic_predictions, money_used, eval_type):
        self.predictions = predictions
        self.labels = labels
        self.task = task
        self.basic_predictions = basic_predictions 
        self.money_used = money_used
        self.eval_type = eval_type
        
    def metric(self, metric_name, is_base:bool=False, model_name=None):
        metric = evaluate.load(metric_name)
        
        preds = self.basic_predictions[model_name] if is_base else self.predictions
        
        if self.task in ["classification", "qa"] and metric_name != 'accuracy':
            return metric.compute(references=self.labels,
                                predictions=preds, 
                                average='micro')
        else:
            return metric.compute(references=self.labels,
                                predictions=preds)
    
    def accuracy(self, is_base:bool=False, model_name=None):
        return self.metric("accuracy", is_base=is_base, model_name=model_name)["accuracy"]
    
    def precision(self, is_base:bool=False, model_name=None):
        return self.metric("precision", is_base=is_base, model_name=model_name)
    
    def recall(self, is_base:bool=False, model_name=None):
        return self.metric("recall", is_base=is_base, model_name=model_name)
    
    def f1(self, is_base:bool=False, model_name=None):
        return self.metric("f1", is_base=is_base, model_name=model_name)
    
    def rouge(self, is_base:bool=False, model_name=None):
        return self.metric("rouge", is_base=is_base, model_name=model_name)
    
    def _compute_coverage_stat(self, scores_pred, scores_true, confidence=0.9):
        """
        Compute statistical coverage — i.e., how often the true score lies within
        a confidence interval around the predicted score.
        """
        mse = np.mean((scores_pred - scores_true) ** 2)
        se = np.sqrt(mse / len(scores_true))
        z = norm.ppf(0.5 + confidence / 2)
        ci_lower = scores_pred - z * se
        ci_upper = scores_pred + z * se
        covered = (scores_true >= ci_lower) & (scores_true <= ci_upper)
        return np.mean(covered)

    def machine_cumulative_accuracy(self, actions):
        llm_indices = []
        for index, value in enumerate(actions):
            if isinstance(value, list):
                if 1 not in value:
                    llm_indices.append(index)
            else:
                if value == 0:
                    llm_indices.append(index)
        correct = sum(self.basic_predictions[i] == self.labels[i] for i in llm_indices)
        total = len(llm_indices)
        return correct / total if total > 0 else 0.0
    
    def human_cumulative_accuracy(self, actions):
        human_indices = []
        for index, value in enumerate(actions):
            if isinstance(value, list):
                if 1 in value:
                    human_indices.append(index)
            else:
                if value == 1:
                    human_indices.append(index)
        if self.eval_type == "auditor_style":
            model = 'qwen'
        else:
            model = 'claude'
        correct = sum(self.basic_predictions[model][i] != self.labels[i] for i in human_indices)
        total = len(human_indices)
        return correct / total if total > 0 else 0.0
    
    def effective_sample_size(self, number_of_human_annotations):
        if self.task in ['classification', 'qa']:
            method_score = self.accuracy()
        else:
            method_score = self.rouge()["rougeL"]
        neffective = number_of_human_annotations * method_score
        try:
            gain = (neffective - number_of_human_annotations) / number_of_human_annotations * 100
        except ZeroDivisionError:
            gain = -1
        return neffective, gain
    
    def cost_tradeoff(self, actions, number_of_human_annotations):
        human_indices = [i for i, v in enumerate(actions)
                        if (1 in v if isinstance(v, list) else v == 1)]
        correct = sum(self.basic_predictions['qwen'][i] != self.labels[i] for i in human_indices)

        if number_of_human_annotations > 0:
            precision = correct / len(human_indices) if len(human_indices) > 0 else 0
            return precision
        return -1

        

    def compute_all_metrics(self, actions, number_of_human_annotations,
                        number_of_llm_annotations,
                        base_folder="results",
                        save_path="metrics.json",
                        seed=42):

        results = {}
        model_metrics = {}

        existing_base_metrics = {}
        if os.path.exists(base_folder):
            for fname in os.listdir(base_folder):
                if fname.endswith(".json") and fname != save_path:
                    fpath = os.path.join(base_folder, fname)
                    try:
                        with open(fpath, "r") as f:
                            existing_data = json.load(f)
                        first_seed = next(iter(existing_data.values()))
                        for key, value in first_seed.items():
                            if key.startswith("base_"):
                                existing_base_metrics[key] = value
                        if existing_base_metrics:
                            break
                    except Exception as e:
                        print(f"Warning: could not read {fname}: {e}")

        if existing_base_metrics:
            for key, value in existing_base_metrics.items():
                results[key] = value
                if key != "base_average":
                    model_name = key[len("base_"):]
                    model_metrics[model_name] = value
        else:
            for model_name in self.basic_predictions.keys():
                model_results = {}
                if self.task in ["classification", "qa"]:
                    model_results["accuracy"] = float(self.accuracy(is_base=True, model_name=model_name))
                    model_results["precision"] = float(self.precision(is_base=True, model_name=model_name)["precision"])
                    model_results["recall"] = float(self.recall(is_base=True, model_name=model_name)["recall"])
                    model_results["f1"] = float(self.f1(is_base=True, model_name=model_name)["f1"])
                else:
                    rouge_base = self.rouge(is_base=True, model_name=model_name)
                    model_results["rouge1"] = float(rouge_base["rouge1"])
                    model_results["rouge2"] = float(rouge_base["rouge2"])
                    model_results["rougeL"] = float(rouge_base["rougeL"])
                results[f"base_{model_name}"] = model_results
                model_metrics[model_name] = model_results

            avg_metrics = {}
            metric_keys = list(next(iter(model_metrics.values())).keys())
            for metric_key in metric_keys:
                values = [model_metrics[model][metric_key] for model in model_metrics]
                avg_metrics[metric_key] = float(np.mean(values))
            results["base_average"] = avg_metrics

        if self.task in ["classification", "qa"]:
            results["method_accuracy"] = float(self.accuracy())
            results["method_precision"] = float(self.precision()["precision"])
            results["method_recall"] = float(self.recall()["recall"])
            results["method_f1"] = float(self.f1()["f1"])
            results["human_cumulative_accuracy"] = self.human_cumulative_accuracy(actions)
        else:
            rouge_scores = self.rouge()
            results["method_rouge1"] = float(rouge_scores["rouge1"])
            results["method_rouge2"] = float(rouge_scores["rouge2"])
            results["method_rougeL"] = float(rouge_scores["rougeL"])

        neffective, gain = self.effective_sample_size(number_of_human_annotations)
        results["effective_sample_size"] = float(neffective)
        results["cost_tradeoff"] = self.cost_tradeoff(actions, number_of_human_annotations)
        results["ess_gain_percent"] = float(gain) if number_of_human_annotations > 0 else -1
        results["number_of_human_annotations"] = number_of_human_annotations
        results["number_of_llm_annotations"] = number_of_llm_annotations
        results["number_of_total_samples"] = len(self.predictions)

        if self.money_used != -1:
            results["money_used"] = self.money_used

        os.makedirs(base_folder, exist_ok=True)
        full_path = os.path.join(base_folder, save_path)
        if os.path.exists(full_path):
            with open(full_path, "r") as f:
                final_results = json.load(f)
            final_results[str(seed)] = results
        else:
            final_results = {str(seed): results}

        with open(full_path, "w") as f:
            json.dump(final_results, f, indent=4)

        return results