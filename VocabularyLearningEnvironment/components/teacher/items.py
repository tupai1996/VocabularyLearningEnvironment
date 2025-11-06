from dataclasses import dataclass
from abc import ABC, abstractmethod


class TeachingItem(ABC):
    @abstractmethod
    def get_question(self):
        pass

    @abstractmethod
    def get_answer(self):
        pass

    @abstractmethod
    def is_answer_correct(self, answer) -> bool:
        pass


@dataclass
class WordItem(TeachingItem):
    source: str
    target: str

    def get_question(self):
        return self.source

    def get_answer(self):
        return self.target

    def is_answer_correct(self, answer: str) -> bool:
        return self.target.strip().lower() == answer.strip().lower()

    def __hash__(self):
        return hash((self.source, self.target))