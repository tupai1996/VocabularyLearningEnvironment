from . base import Planner
from .planning_contexts import PlanningContext
from typing import List
from . items import TeachingItem
import random

class RandomPlanner(Planner):
    def choose_item(self, material: List[TeachingItem], context: PlanningContext, time: int):
        return random.choice(material)