from components.teacher.items import WordItem
from components.teacher.base import Teacher
from components.teacher.planning_contexts import (
    EmptyPlanningContext,
    FixedHorizonContext,
    FixedLearnerContext,
)
from components.teacher.planners import RandomPlanner
from components.learners.exp_memory import ExpMemoryLearner


material = [WordItem("dog", "hund"), WordItem("cat", "katze")]


# context = EmptyPlanningContext()
context = FixedLearnerContext(ExpMemoryLearner(0.4, 0.1))
planner = RandomPlanner()
teacher = Teacher(material, planner, context)

learner = ExpMemoryLearner(0.4, 0.1)

for t in range(10):
    item = teacher.choose_item(t)
    reply = learner.reply(item.get_question(), t)
    learner.learn(item, t)
    teacher.gets_answer(item, reply, t)
    print("Question:", item.get_question(), " |  Answer:", reply)
