from abc import ABC, abstractmethod
from typing import List
from .items import TeachingItem
from . planning_contexts import PlanningContext



class Planner(ABC):
    @abstractmethod
    def choose_item(self, material: List[TeachingItem], context: PlanningContext, time: int):
        pass


class Teacher:
    
    def __init__(self, material: List[TeachingItem], planner: Planner, context: PlanningContext):
        # Assertions
        is_list = isinstance(material, list)
        contains_teaching_items = all(isinstance(item, TeachingItem) for item in material)
        if not is_list or not(material) or not contains_teaching_items:
            raise TypeError("material must be a non-empty list of TeachingItem instances")
        assert isinstance(planner, Planner), "planner must be a Planner"
        
        # Initialization
        self.material = material
        self.planner = planner
        self.context = context
    
    
    def choose_item(self, time: int):
        return self.planner.choose_item(self.material, self.context, time)
    
    def gets_answer(self, queried_item: TeachingItem, answer, time: int):
        self.context.update(queried_item, answer, time)
    
