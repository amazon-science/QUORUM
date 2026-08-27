import numpy as np
import random

class RandomRouter:
    
    def __init__(self, human_budget):
        self.human_budget = human_budget
        self.human_used = human_budget
    
    def route(self, llm_predictions, human_labels, routing_strategy="minimize", **kwargs):

        n = len(llm_predictions)
        output_set = np.array(llm_predictions)
        self.actions = np.zeros(n)    
        self.money_used = 0
        self.llm_used = 0
        backup =  kwargs.get("backup", None)
        
        if kwargs['money_budget'] is None:
            self.llm_used = len(llm_predictions) - self.human_budget
            self.actions = np.zeros(len(llm_predictions))
            random_human_indices = random.sample(range(len(human_labels)), int(self.human_budget))

            random_predicted = np.array(llm_predictions)
            for idx in random_human_indices:
                random_predicted[idx] = human_labels[idx]  
                self.actions[idx] = 1
            return random_predicted
        else:
            if routing_strategy == "maximize":
                llm_cost = kwargs['annotator_cost'][0]
                human_cost = kwargs['annotator_cost'][-1]

                base_cost = n * llm_cost

                extra_budget = kwargs['money_budget'] - base_cost
                extra_cost_per_human = human_cost - llm_cost

                if extra_cost_per_human > 0:
                    max_human = int(extra_budget // extra_cost_per_human)
                    max_human = min(max_human, n)
                else:
                    max_human = n
                
                random_human_indices = random.sample(range(n), max_human)

                for idx in random_human_indices:
                    output_set[idx] = human_labels[idx]
                    self.actions[idx] = 1

                self.human_used = max_human
                self.llm_used = n - max_human
                self.money_used = self.human_used * human_cost + self.llm_used * llm_cost
            else:
                llm_cost = kwargs['annotator_cost'][0]
                human_cost = kwargs['annotator_cost'][-1]
                human_prob = self.human_budget / n
                
                for i in range(n):
                    action = 1 if random.random() < human_prob else 0
                    
                    cost_of_action = human_cost if action == 1 else llm_cost
                    
                    if self.money_used + cost_of_action > kwargs['money_budget']:
                        if action == 1 and self.money_used + llm_cost <= kwargs['money_budget']:
                            action = 0
                        elif self.money_used + llm_cost <= kwargs['money_budget']:
                            action = 0
                        else:
                            action = -1
                            print(f"Random - Stopped annotating at sample {i}")
                    
                    self.actions[i] = action
                    
                    if action == 1:
                        output_set[i] = human_labels[i]
                        self.human_used += 1
                        self.money_used += human_cost
                    
                    elif action == 0:
                        output_set[i] = llm_predictions[i]
                        self.llm_used += 1
                        self.money_used += llm_cost
                    
                    else:  
                        if backup:
                            output_set[i] = llm_predictions[i]
                        else:
                            try:
                                output_set[i] = human_labels[i]+1
                            except TypeError:
                                output_set[i] = "-1"
            return output_set