
from routers.random_router import RandomRouter
from routers.sant_router import SANTRouter
from routers.ppi_router import PPIRouter
from routers.co_annotating_router import CoAnnotatingRouter
from routers.araida_router import ARAIDARouter
from routers.QUORUM_router import QUORUMRouter
from routers.pac_router import PACRouter
from routers.ablator_router import AblatorRouter

class RouterExecutor:
    def __init__(self, name, llm_predictions, human_labels,
                human_budget, **kwargs):
        
        self.name = name
        self.human_budget = human_budget
        self.llm_predictions = llm_predictions
        self.human_labels = human_labels
        self.kwargs = kwargs
        self.create_router()
    
    def create_router(self):
        
        if self.name == "QUORUM":
            self.router = QUORUMRouter(base_features = self.kwargs['difficulty_features'],
                                        human_budget = self.human_budget if self.human_budget is not None else len(texts),
                                        **self.kwargs)
        elif self.name == "Random":
            self.router = RandomRouter(self.human_budget)
        elif self.name == "Araida":
            self.router = ARAIDARouter(self.human_budget)
        elif self.name == "PPI":
            self.router = PPIRouter(self.human_budget)
        elif self.name == "CoAnnotating":
            self.router = CoAnnotatingRouter(self.human_budget)
        elif self.name == "SANT":
            self.router = SANTRouter(self.human_budget)
        elif self.name == "PAC":
            self.router = PACRouter(self.human_budget, **self.kwargs)
        elif self.name == "Ablator":
            self.router = AblatorRouter(self.human_budget)
        else:
            raise ValueError(f"Routing strategy {self.name} not recognized.")
            
    
    def run(self, **kwargs):

        if self.name == 'QUORUM':
            output =  self.router.route(llm_predictions=self.llm_predictions,
                                     human_labels=self.human_labels,
                                     **kwargs)
        elif self.name == "Random":
            output = self.router.route(llm_predictions=self.llm_predictions['qwen'],
                                     human_labels=self.human_labels,
                                     **kwargs)
        elif self.name == "Araida":
            output = self.router.route(llm_predictions=self.llm_predictions['qwen'],
                                       human_labels=self.human_labels,
                                       **kwargs
                                       )
        elif self.name == "PPI":
            output = self.router.route(llm_predictions=self.llm_predictions['qwen'],
                                       human_labels=self.human_labels,
                                       **kwargs
                                       )
        elif self.name == "CoAnnotating" or self.name == "Ablator":
            output = self.router.route(llm_predictions=self.llm_predictions['qwen'],
                                       human_labels=self.human_labels,
                                       **kwargs
                                       )
        elif self.name == "PAC":
            output = self.router.route(llm_predictions=self.llm_predictions,
                                       human_labels=self.human_labels,
                                       **kwargs
                                       )
        elif self.name == "SANT":
            output = self.router.route(llm_predictions=self.llm_predictions['qwen'],
                                       human_labels=self.human_labels,
                                       **kwargs
                                       )
        else:
            raise ValueError(f"Routing strategy {self.name} not recognized.")   
        return output
            
