import argparse
import json

from main import *
from src.data import NLPDataset

if __name__ == "__main__":
    
    split_values = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='pubmed')
    parser.add_argument('--language', type=str, default=None)
    parser.add_argument('--eval_type', type=str, default="human_budget", choices=["dollars", "auditor_style", "human_budget"])
    parser.add_argument('--annotator_cost', type=float, nargs="+", default=[0.05, 0.03, 0.01, 0.1])
    parser.add_argument('--methods', type=str, nargs="+",  default=['QUORUM', 'Random', 'SANT', 'CoAnnotating','PPI', 'Araida', 'PAC'])
    parser.add_argument('--lw', type=float, default=0.5)
    parser.add_argument('--additional_info', type=str, default=None)
    parser.add_argument('--split_values', type=str, nargs="+", default=split_values)
    parser.add_argument('--backup', action='store_true', default=None)
    args = parser.parse_args()
    
    num_dataset_samples = len(NLPDataset(dataset_name=args.dataset, language=args.language).x)
    
    for seed in range(args.n_seeds):
        if args.eval_type == "dollars":
            for split_value in split_values[2:-1]:
                budget = int(split_value * args.annotator_cost[-1] * num_dataset_samples)
                metrics = main(dataset_name=args.dataset, annotator_cost=args.annotator_cost,
                                    alpha=0.9, money_budget=budget, 
                                    linguistic_weights=args.lw, eval_type=args.eval_type,
                                    methods=args.methods,
                                    seed=seed,
                                    additional_info=args.additional_info,
                                    language=args.language,
                                    backup=args.backup)
        elif args.eval_type == "human_budget" or args.eval_type == "auditor_style":
            for split_value in split_values:
                budget = int(num_dataset_samples*split_value)
                metrics = main(dataset_name=args.dataset, annotator_cost=args.annotator_cost,
                                        alpha=0.9, human_budget=budget, 
                                        linguistic_weights=args.lw, eval_type=args.eval_type,
                                        methods=args.methods,
                                        seed=seed,
                                        additional_info=args.additional_info,
                                        language=args.language,
                                        backup=args.backup)
        else:
            raise ValueError("Invalid eval_type. Must be one of ['dollars', 'human_budget', 'auditor_style]")