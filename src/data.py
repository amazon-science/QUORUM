import numpy as np
from datasets import load_dataset
import pandas as pd

class NLPDataset:
    
    def __init__(self, dataset_name:str, language:str="spanish"):
        self.available_datasets = ['imdb', 'pubmed',
                                   'agnews', 'sst2',
                                   'cnn', 'xlsum']
        self.dataset_name = dataset_name
        self.language = language
        self.load_data()

    def load_data(self):
        label_alias = 'label'
        x_name = 'text'
        self.task = "classification"
        
        if self.dataset_name == 'imdb':
            dataset = load_dataset("imdb", split="test")
        elif self.dataset_name == 'dbpedia':
            dataset = load_dataset("pietrolesci/dbpedia_14_indexed", split="test")
            label_alias = 'labels'
            x_name = 'content'

        elif self.dataset_name == "pubmed":
            dataset = load_dataset("pietrolesci/pubmed-20k-rct", split="test")
            label_alias = 'labels'
        elif self.dataset_name == "emotions":
            dataset = load_dataset("dair-ai/emotion", split="test")
        elif self.dataset_name == "rotten":
            dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="test")
        elif self.dataset_name == "agnews":
            dataset = load_dataset("fancyzhx/ag_news", split="test")
        elif self.dataset_name == "banking77":
            dataset = load_dataset("mteb/banking77", split="test")
        elif self.dataset_name == "fnc1":
            dataset = load_dataset("nid989/FNC-1", split="test")
            x_name = 'articleBody'
            label_alias = "Stance"
        elif self.dataset_name == "mnli":
            dataset = load_dataset("nyu-mll/glue", "mnli",  split="validation_matched")
            x_name = 'sentence'
        elif self.dataset_name == "qnli":
            dataset = load_dataset("nyu-mll/glue", "qnli", split="validation")
            x_name = 'sentence'
        elif self.dataset_name == "sst2":
            dataset = load_dataset("stanfordnlp/sst2", split="validation")
            x_name = 'sentence'
        elif self.dataset_name == "trec6":
            dataset = load_dataset("OxAISH-AL-LLM/trec6", split="test")
        elif self.dataset_name == "multilingual":
            dataset = load_dataset("tyqiangz/multilingual-sentiments", self.language, split="test")
        elif self.dataset_name == 'cnn':
            dataset = load_dataset("abisee/cnn_dailymail", '3.0.0', split="test")
            x_name = 'article'
            label_alias = "highlights"
            self.task = "summarization"
        elif self.dataset_name == 'xlsum':
            dataset = load_dataset("csebuetnlp/xlsum", self.language, split="test")
            x_name = 'text'
            label_alias = "summary"
            self.task = "summarization"
        elif self.dataset_name == "politeness":
            dataset = pd.read_csv("data/politeness_dataset.csv")
            label_alias = 'Politeness'
            x_name = 'Text'
        elif self.dataset_name == 'global-mmlu':
            dataset = load_dataset("CohereLabs/Global-MMLU", self.language, split="test")
            x_name = 'question'
            self.options = []
            label_alias = "answer"
            self.options = [[item[f"option_{i}"] for i in ["a", "b", "c", "d"]] for item in dataset]
            self.task = "qa"
        elif self.dataset_name == "mmlu-redux":
            categories = ['anatomy', 'business_ethics', 'clinical_knowledge', 
                'college_chemistry', 'college_computer_science', 'college_mathematics', 
                'college_medicine', 'college_physics', 'econometrics', 'electrical_engineering', 
                'formal_logic', 'global_facts', 'high_school_chemistry', 'high_school_mathematics',  
                'high_school_physics', 'high_school_statistics', 'human_aging', 'logical_fallacies',  
                'machine_learning', 'miscellaneous', 'philosophy', 'professional_accounting', 
                'public_relations', 'virology', 'conceptual_physics', 'high_school_us_history', 
                'astronomy', 'high_school_geography', 'high_school_macroeconomics', 'professional_law']
            
            complete_data = []
            complete_labels = []
            complete_options = []

            for category in categories:
                cat_dataset = load_dataset("edinburgh-dawg/mmlu-redux", category, split="test")
                complete_data.extend(cat_dataset['question'])
                complete_labels.extend(cat_dataset['answer'])
                complete_options.extend(cat_dataset['choices'])

            dataset = {
                'question': complete_data,
                'answer': complete_labels
            }

            x_name = 'question'
            label_alias = "answer"
            self.options = complete_options
            self.task = "qa"
                
        else:
            raise  ValueError(f"Dataset {self.dataset_name} not recognized!")
        
        self.complete_dataset = dataset
        if self.task == "classification":
            self.labels = self.get_labels(label_alias=label_alias)
        else:
            self.labels = None
            
        self.x = dataset[x_name]

        self.y = np.array(dataset[label_alias])
        self.postprocess()
        

    def get_labels(self, label_alias='label') -> dict:

        if label_alias == 'Stance':
            return {
            1 : "Agrees",
            2 : "Disagrees",
            3 : "Discusses",
            4 : "Unrelated",
        }
        
        elif label_alias == "Politeness":   
            return {
                1 : "Polite",
                0 : "Impolite",
                }
        else:
            output = {}
            label_feature = self.complete_dataset.features[label_alias]
            if hasattr(label_feature, "names"):
                labels = label_feature.names
            else:
                labels = sorted(set(self.complete_dataset[label_alias]))
            for index, class_value in enumerate(labels):
                output[index] = class_value
            return output

    def __len__(self):
        return len(self.x)
    
    def postprocess(self):
        mapping = {'A': 0,
                   'B' : 1,
                   'C' : 2,
                   'D' : 3}
        if self.task == "qa":
            x_list = list(self.x)

            for i, (sentence, choices) in enumerate(zip(x_list, self.options)):
                x_list[i] = sentence + f" Possible choices are: {choices}"

            self.x = x_list
        
            y_list = list(self.y)
            if self.dataset_name != "mmlu-redux":
                for i, label in enumerate(self.y):
                    y_list[i] = mapping[label]
            self.y = y_list
            
    def get_class_distribution(self, normalize=True):
        unique, counts = np.unique(self.y, return_counts=True)
        if normalize:
            counts = counts / counts.sum()
        return dict(zip(unique, counts))


if __name__ == "__main__":
    dataset = NLPDataset("mmlu-redux")
    print(dataset.x[:5], dataset.y[:5], len(dataset.x))
    print(dataset.get_class_distribution())
   
