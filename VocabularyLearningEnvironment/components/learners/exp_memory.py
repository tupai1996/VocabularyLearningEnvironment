from dataclasses import dataclass
from vocab.models import UserMemory, Vocabulary
from .base import BaseLearner
from components.teacher.items import TeachingItem, WordItem
import numpy as np
from datetime import datetime, timezone
from django.db import transaction
import time
from django.db import OperationalError, transaction


@dataclass
class MemoryState:
    item: TeachingItem
    vocab_id: int
    n_occurrences: int
    last_occurrence: int
    alpha: float
    beta: float

    def get_probability(self, time: int):
        return np.exp(
            -self.alpha
            * (1 - self.beta) ** self.n_occurrences
            * (time - self.last_occurrence)
        )


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

    def learn(self, item: TeachingItem, vocab_id: int, time: int):
        question = item.get_question()
        if question in self.memory:
            self.memory[question].n_occurrences += 1
            self.memory[question].last_occurrence = time
        else:
            state = MemoryState(item, vocab_id, 1, time, self.alpha, self.beta)
            self.memory[question] = state

    
    def load_memory(self, memory_dict):
        self.memory = {}

        for question, state in memory_dict.items():
            item = WordItem(state['item']['source'], state['item']['target'])
            
            mem_state = MemoryState(
                item=item,
                n_occurrences=state['n_occurrences'],
                last_occurrence=state['last_occurrence'],
                alpha=state['alpha'],
                beta=state['beta']
            )

            self.memory[question] = mem_state


    def save_memory_to_db(self, user):
        for q, state in self.memory.items():
            try:
                vocab = Vocabulary.objects.get(id=state.vocab_id)
            except Vocabulary.DoesNotExist:
                raise ValueError(f"Vocabulary with id={state.vocab_id} not found in DB!")
            
            user_mem, created = UserMemory.objects.get_or_create(
                user=user, 
                vocabulary=vocab,
                vocabulary_list=vocab.vocabulary_list,
                
                defaults={
                    'n_occurrences': state.n_occurrences,
                    'last_occurrence': state.last_occurrence,
                    'alpha': state.alpha,
                    'beta': state.beta
                }
            )
            if not created:
                user_mem.vocabulary_list = vocab.vocabulary_list
                user_mem.n_occurrences = state.n_occurrences
                user_mem.last_occurrence = state.last_occurrence
                user_mem.alpha = state.alpha
                user_mem.beta = state.beta
                with transaction.atomic():
                    user_mem.save()


    def save_memory_to_db_with_retry(self, user, retries=5, delay=0.1):
        for _ in range(retries):
            try:
                with transaction.atomic():
                    self.save_memory_to_db(user)
                break
            except OperationalError as e:
                if 'database is locked' in str(e):
                    time.sleep(delay)
                else:
                    raise
    @classmethod
    def load_memory_from_db(cls, user, alpha: float = 0.1, beta: float = 0.5, retries: int = 5, delay: float = 0.1):
        for attempt in range(retries):
            try:
                learner = cls(alpha=alpha, beta=beta)
                user_memory_qs = UserMemory.objects.select_related("vocabulary").filter(user=user)
                
                for user_memory in user_memory_qs:
                    item = WordItem(source=user_memory.vocabulary.source_word, target=user_memory.vocabulary.target_word)
                    question=item.get_question()
        
                    memory_state = MemoryState(
                        item=item,
                        vocab_id=user_memory.vocabulary.id,
                        n_occurrences=user_memory.n_occurrences,
                        last_occurrence=user_memory.last_occurrence,
                        alpha=user_memory.alpha,
                        beta=user_memory.beta,
                    )
                
                    learner.memory[question] = memory_state
                return learner
            
            except OperationalError as e:
                if 'database is locked' in str(e):
                    time.sleep(delay)  
                else:
                    raise

        raise OperationalError(f"Could not load memory for user {user.id} after {retries} retries")


