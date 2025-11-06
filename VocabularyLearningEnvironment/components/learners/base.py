from abc import ABC, abstractmethod
from components.teacher.items import TeachingItem
from copy import deepcopy


class BaseLearner(ABC):
    @abstractmethod
    def reply(self, question, time: int):
        pass

    @abstractmethod
    def learn(self, item: TeachingItem, time: int):
        pass

    def deepcopy(self):
        return deepcopy(self)
