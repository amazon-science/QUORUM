import numpy as np    

class CoAnnotatingRouter:
    def __init__(self, human_budget):
        self.human_budget = human_budget 
        self.actions = []
        self.human_used = 0
        self.llm_used = 0
    
    def route(self, llm_predictions, human_labels, confidence_scores, routing_strategy="minimize", **kwargs):
                
        indexed_samples = list(enumerate(confidence_scores))
        output_set = []
        self.money_used = kwargs['annotator_cost'][2] * len(human_labels)
        backup =  kwargs.get("backup", None)
        
        indexed_samples.sort(key=lambda x: x[1])
        if kwargs['money_budget'] is None:
            human_indices = [idx for idx, _ in indexed_samples[:self.human_budget]]
            self.actions = []
            
            for i in range(len(confidence_scores)):
                if i in human_indices:
                    self.actions.append(1)
                    self.human_used += 1
                    output_set.append(human_labels[i])
                else:
                    self.actions.append(0)
                    self.llm_used += 1
                    output_set.append(llm_predictions[i])
        else:
            if routing_strategy == "maximize":
                n = len(confidence_scores)
                llm_cost = kwargs['annotator_cost'][0]
                human_cost = kwargs['annotator_cost'][-1]

                base_cost = n * llm_cost

                extra_budget = kwargs['money_budget'] - base_cost
                extra_cost_per_human = human_cost - llm_cost

                if extra_cost_per_human > 0:
                    max_human = int(extra_budget // extra_cost_per_human)
                    max_human = min(max_human, n)
                elif extra_cost_per_human <= 0:
                    max_human = n

                human_indices = set(idx for idx, _ in indexed_samples[:max_human])

                for i in range(n):
                    if i in human_indices:
                        self.actions.append(1)
                        self.human_used += 1
                        output_set.append(human_labels[i])
                        self.money_used += human_cost
                    else:
                        self.actions.append(0)
                        self.llm_used += 1
                        output_set.append(llm_predictions[i])
            else:
                confidence_threshold = .5
                n = len(confidence_scores)
                for i in range(n):
                    if confidence_scores[i] < confidence_threshold:
                        action = 1
                    else:
                        action = 0 
                    llm_cost = kwargs['annotator_cost'][0]
                    human_cost = kwargs['annotator_cost'][-1]

                    cost_of_action = human_cost if action == 1 else llm_cost

                    if self.money_used + cost_of_action > kwargs['money_budget']:
                        if action == 1 and self.money_used + llm_cost <= kwargs['money_budget']:
                            action = 0
                        elif self.money_used + llm_cost <= kwargs['money_budget']:
                            action = 0
                        else:
                            action = -1
                            print(f"Co-Annotating - Stopped annotating at sample {i}")

                    self.actions.append(action)

                    if action == 1:
                        output_set.append(human_labels[i])
                        self.human_used += 1
                        if kwargs['money_budget'] is not None:
                            self.money_used += human_cost

                    elif action == 0:
                        output_set.append(llm_predictions[i])
                        self.llm_used += 1
                    else:  
                        if backup:
                            output_set.append(llm_predictions[i])
                        else:
                            try:
                                output_set.append(human_labels[i]+1)
                            except TypeError:
                                output_set.append("-1")
        
        return output_set

if __name__ == "__main__":
    from src.data import NLPDataset
    data = NLPDataset('banking77')

    real_labels = data.y[:101]
    llm_predictions = np.load("predictions/all_predictions_banking77.npy")[:101]
    confidence_scores = np.load("summaries/claude/all_confidences_cnn.npy")

    router = CoAnnotatingRouter(human_budget=60)

    samples = router.route(llm_predictions=llm_predictions,
                        human_labels=real_labels,
                        confidence_scores=confidence_scores)

    print(samples)
    print(router.actions)