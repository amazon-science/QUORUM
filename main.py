import numpy as np
import os
from collections import defaultdict
from sklearn.utils import shuffle
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import nltk
import argparse
import evaluate
from sklearn.metrics import accuracy_score
from scipy.stats import norm
import time

from src.data import NLPDataset
from routers.router import RouterExecutor
from src.utils import *
from src.metrics import Metrics
from src.diagnostics import *

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')

def main(methods:list = ['QUORUM'],
         dataset_name: str = 'agnews',
         alpha: float = 0.7,
         linguistic_weights: float = 0.5,
         annotator_cost: list = [0.01, 0.01, 0.01, 0.1],
         human_budget: int = None,
         seed:int=42,
         money_budget:int = None,
         eval_type:str=None,
         language=None,
         **kwargs):
    if not os.path.isdir("summaries"):
        download_data()

    seed_everything(seed)

    
    dataset = NLPDataset(dataset_name, language=language)
    texts, labels, task_name = dataset.x, dataset.y, dataset.task 

    predictors = ['qwen', 'nova_pro', 'claude']
    complete_responses = defaultdict(list)
    
    
    # --- Load model responses ---
    if task_name == "classification":
        for predictor in predictors:
            file_name = f'summaries/{predictor}/all_predictions_{dataset_name}'
            if language is not None:
                file_name += f"_{language}"
            responses = np.load(f"{file_name}.npy",
                        allow_pickle=True).astype(int)
            complete_responses[predictor] = responses
    else:
        for predictor in predictors:
            file_name = f'summaries/{predictor}/all_predictions_{dataset_name}'
            if language is not None:
                file_name += f"_{language}"
            responses = np.load(f"{file_name}.npy",
                            allow_pickle=True)
            complete_responses[predictor] = responses
        
    file_name = f'summaries/qwen/all_confidences_{dataset_name}'
    if language is not None:
                file_name += f"_{language}"
    confidences = np.load(f"{file_name}.npy",
                        allow_pickle=True)
    
    file_name = f'embeddings/{dataset_name}_distances'
    if language is not None:
        file_name += f"_{language}"
    
    if os.path.exists(f"{file_name}.npy"):
        distances = np.load(f"{file_name}.npy")
        D_ling = np.load(f"{file_name}.npy".replace("distances", "linguistic_features"))
    else:
        os.makedirs("embeddings", exist_ok=True)
        model = SentenceTransformer("NovaSearch/stella_en_1.5B_v5")
        embeddings = model.encode(texts, normalize_embeddings=True)
        print(f"Computing the embeddings!")
        k = 5
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)
        np.save(file_name, distances)
        
        D_ling = []
        for t in texts:
            f = extract_linguistic_features(t)
            if f is None:
                D_ling.append(np.zeros(6))
            else:
                D_ling.append(f)
        
        D_ling = np.array(D_ling)
        D_ling = (D_ling - D_ling.min()) / (D_ling.max() - D_ling.min())
        np.save(file_name.replace("distances", "linguistic_features"), D_ling)
        
    local_density = 1 / (np.mean(distances[:, 1:], axis=1) + 1e-6)
    D_emb = 1 - (local_density - local_density.min()) / (local_density.max() - local_density.min())

    D_emb = D_emb.reshape(-1, 1)
    X = np.hstack([D_ling, D_emb])

    additional_info = kwargs.get('additional_info', None)
    if money_budget is not None:
        human_budget = money_budget
        if money_budget > annotator_cost[2]*len(texts):
            num_samples_to_train = int(0.66*(money_budget - annotator_cost[2]*len(texts)))
        else:
            num_samples_to_train = 30
    else:
        if human_budget > int(0.2*len(texts)):
            num_samples_to_train = int(0.2*len(texts))
        else:
            num_samples_to_train = 30
            
    backup = kwargs['backup']
        
    kwargs = {'linguistic_weights' : linguistic_weights,
              'difficulty_features' : X,
              'annotator_cost' : annotator_cost,
              'alpha' : alpha,
              'threshold' : 0.485,
              'num_samples_to_train' : num_samples_to_train,
              'texts_to_annotate' : texts,
              'confidence_scores' : confidences,
              'annotator_cost' : annotator_cost,
              'eval_type' : eval_type,
              'money_budget' : money_budget,
              'backup' : backup,
              }
    
    for single_method in methods:
        router = RouterExecutor(name=single_method, llm_predictions=complete_responses, human_labels=labels,
                                human_budget=human_budget, **kwargs)

        if language is None:
            base_folder = os.path.join("results", dataset_name, str(human_budget))
        else:
            base_folder = os.path.join("results", f"{dataset_name}_{language}", str(human_budget))  
        annotated_samples = router.run(**kwargs)
        if single_method == "QUORUM":
            plotter = BanditPlotter(
                router=router,
                llm_predictions=complete_responses,
                true_labels=labels,
                eval_type=eval_type
            )
            plotter.plot_all(window=200, cost_per_human=annotator_cost[-1], save_path=base_folder)
            
        if eval_type == "dollars":
            money_used = router.router.money_used
        else:
            money_used = -1
        metrics = Metrics(predictions=annotated_samples, labels=labels,
                                  task=task_name, basic_predictions=complete_responses, money_used=money_used,
                                  eval_type=eval_type)
        
        if additional_info is None:
            save_path = f'{single_method}_{eval_type}.json'
        else:
            save_path = f'{single_method}_{eval_type}_{additional_info}.json'
        
        if backup:
            save_path = f"{save_path.split('.json')[0]}_backup.json"
            

        metrics.compute_all_metrics(actions=router.router.actions,
                                number_of_human_annotations = router.router.human_used,
                                number_of_llm_annotations = router.router.llm_used,
                                base_folder = base_folder,
                                save_path=save_path,
                                seed=seed)
        
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="banking77")
    parser.add_argument('--language', type=str, default=None)
    parser.add_argument('--eval_type', type=str, default="human_budget", choices=["dollars", "auditor_style", "human_budget"])
    parser.add_argument('--budget', type=int, default=163)
    parser.add_argument('--annotator_cost', type=float, nargs="+", default=[0.05, 0.03, 0.01, 0.1])
    parser.add_argument('--methods', type=str, nargs="+",  default=['QUORUM', 'Random', 'SANT', 'CoAnnotating','PPI', 'Araida', 'PAC'])
    parser.add_argument('--lw', type=float, default=0.5)
    parser.add_argument('--additional_info', type=str, default=None)
    parser.add_argument('--backup', action='store_true', default=True)
    args = parser.parse_args()
    
    dataset_name = args.dataset
    annotator_cost = args.annotator_cost
    linguistic_weights = args.lw
    
    if args.eval_type == "dollars":
        money_budget = args.budget
        metrics = main(dataset_name=dataset_name, annotator_cost=annotator_cost,
                             alpha=0.9, money_budget=args.budget, 
                             linguistic_weights=linguistic_weights, eval_type=args.eval_type,
                             methods=args.methods,
                             additional_info=args.additional_info,
                             language=args.language,
                             backup=args.backup)
    elif args.eval_type == "human_budget" or args.eval_type == "auditor_style":
        metrics = main(dataset_name=dataset_name, annotator_cost=annotator_cost,
                                alpha=0.9, human_budget=args.budget, 
                                linguistic_weights=linguistic_weights, eval_type=args.eval_type,
                                methods=args.methods,
                                additional_info=args.additional_info,
                                language=args.language,
                                backup=args.backup)
    else:
        raise ValueError("Invalid eval_type. Must be one of ['dollars', 'human_budget', 'auditor_style]")
