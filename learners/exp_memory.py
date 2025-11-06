from dataclasses import dataclass
from . base import BaseLearner
from ..teacher.items import TeachingItem
import numpy as np


@dataclass
class MemoryState:
    item: TeachingItem
    n_occurrences: int
    last_occurrence: int
    alpha: float
    beta: float
    
    def get_probability(self, time: int):
        return np.exp(- self.alpha * (1 - self.beta)**self.n_occurrences * (time - self.last_occurrence))
    


class ExpMemoryLearner(BaseLearner):
    def __init__(self, alpha, beta):
        self.memory = dict()
        self.alpha = alpha
        self.beta = beta
        
    def reply(self, question: str, time: int):
        assert isinstance(question, str), "question must be a character string"
        if question in self.memory:
            memorized = np.random.rand() < self.memory[question].get_probability(time)
            return self.memory[question].item.get_answer() if memorized else None
        return None
    

    def learn(self, item: TeachingItem, time: int):
        question = item.get_question()
        if question in self.memory:
            self.memory[question].n_occurrences += 1
            self.memory[question].last_occurrence = time
        else:
            state = MemoryState(item, 1, time, self.alpha, self.beta)
            self.memory[question] = state