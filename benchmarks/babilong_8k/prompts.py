"""Official BABILong prompt template and qa1-qa5 prompt text.

Adapted from ``babilong/prompts.py`` at revision
38da79d79519ef87aa46ae804f838e1eab7f86d7 (Apache-2.0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type TaskName = Literal["qa1", "qa2", "qa3", "qa4", "qa5"]


@dataclass(frozen=True)
class TaskPrompt:
    instruction: str
    examples: str
    post_prompt: str


TASK_PROMPTS: dict[TaskName, TaskPrompt] = {
    "qa1": TaskPrompt(
        instruction=(
            "I will give you context with the facts about positions of different persons hidden in some random text "
            "and a question. You need to answer the question based only on the information from the facts. "
            "If a person was in different locations, use the latest location to answer the question."
        ),
        examples=(
            "<example>\n"
            "Charlie went to the hallway. Judith come back to the kitchen. Charlie travelled to balcony. "
            "Where is Charlie?\n"
            "Answer: The most recent location of Charlie is balcony.\n"
            "</example>\n\n"
            "<example>\n"
            "Alan moved to the garage. Charlie went to the beach. Alan went to the shop. Rouse "
            "travelled to balcony. Where is Alan?\n"
            "Answer: The most recent location of Alan is shop.\n"
            "</example>"
        ),
        post_prompt=(
            "Always return your answer in the following format: "
            "The most recent location of ’person’ is ’location’. Do not write anything else after that."
        ),
    ),
    "qa2": TaskPrompt(
        instruction=(
            "I give you context with the facts about locations and actions of different persons "
            "hidden in some random text and a question."
            "You need to answer the question based only on the information from the facts.\n"
            "If a person got an item in the first location and travelled to the second location "
            "the item is also in the second location. "
            "If a person dropped an item in the first location and moved to the second location "
            "the item remains in the first location."
        ),
        examples=(
            "<example>\n"
            "Charlie went to the kitchen. Charlie got a bottle. Charlie moved to the balcony. "
            "Where is the bottle?\n"
            "Answer: The bottle is in the balcony.\n"
            "</example>\n"
            "<example>\n"
            "Alan moved to the garage. Alan got a screw driver. Alan moved to the kitchen. Where "
            "is the screw driver?\n"
            "Answer: The screw driver is in the kitchen.\n"
            "</example>"
        ),
        post_prompt=(
            "Always return your answer in the following format: The ’item’ is in ’location’. "
            "Do not write anything else after that."
        ),
    ),
    "qa3": TaskPrompt(
        instruction=(
            "I give you context with the facts about locations and actions of different persons "
            "hidden in some random text and a question. "
            "You need to answer the question based only on the information from the facts.\n"
            "If a person got an item in the first location and travelled to the second location "
            "the item is also in the second location. "
            "If a person dropped an item in the first location and moved to the second location "
            "the item remains in the first location."
        ),
        examples=(
            "<example>\n"
            "John journeyed to the bedroom. Mary grabbed the apple. Mary went back to the bathroom. "
            "Daniel journeyed to the bedroom. Daniel moved to the garden. Mary travelled to the kitchen. "
            "Where was the apple before the kitchen?\n"
            "Answer: Before the kitchen the apple was in the bathroom.\n"
            "</example>\n"
            "<example>\n"
            "John went back to the bedroom. John went back to the garden. John went back to the kitchen. "
            "Sandra took the football. Sandra travelled to the garden. Sandra journeyed to the bedroom. "
            "Where was the football before the bedroom?\n"
            "Answer: Before the bedroom the football was in the garden.\n"
            "</example>"
        ),
        post_prompt=(
            "Always return your answer in the following format: "
            "Before the $location_1$ the $item$ was in the $location_2$. Do not write anything else after that."
        ),
    ),
    "qa4": TaskPrompt(
        instruction=(
            "I will give you context with the facts about different people, their location and actions, hidden in "
            "some random text and a question. "
            "You need to answer the question based only on the information from the facts."
        ),
        examples=(
            "<example>\n"
            "The hallway is south of the kitchen. The bedroom is north of the kitchen. "
            "What is the kitchen south of?\n"
            "Answer: bedroom\n"
            "</example>\n"
            "<example>\n"
            "The garden is west of the bedroom. The bedroom is west of the kitchen. What is west of the bedroom?\n"
            "Answer: garden\n"
            "</example>"
        ),
        post_prompt="Your answer should contain only one word - location. Do not write anything else after that.",
    ),
    "qa5": TaskPrompt(
        instruction=(
            "I will give you context with the facts about locations and their relations hidden in some random text "
            "and a question. You need to answer the question based only on the information from the facts."
        ),
        examples=(
            "<example>\n"
            "Mary picked up the apple there. Mary gave the apple to Fred. Mary moved to the bedroom. "
            "Bill took the milk there. Who did Mary give the apple to?\n"
            "Answer: Fred\n"
            "</example>\n"
            "<example>\n"
            "Jeff took the football there. Jeff passed the football to Fred. Jeff got the milk there. "
            "Bill travelled to the bedroom. Who gave the football?\n"
            "Answer: Jeff\n"
            "</example>\n"
            "<example>\n"
            "Fred picked up the apple there. Fred handed the apple to Bill. Bill journeyed to the bedroom. "
            "Jeff went back to the garden. What did Fred give to Bill?\n"
            "Answer: apple\n"
            "</example>"
        ),
        post_prompt=(
            "Your answer should contain only one word. Do not write anything else after that. "
            "Do not explain your answer."
        ),
    ),
}


def format_prompt(task: TaskName, context: str, question: str) -> str:
    """Build the full input sent to the answer model with the official default template."""
    task_prompt = TASK_PROMPTS[task]
    return (
        f"{task_prompt.instruction}\n\n"
        f"{task_prompt.examples}\n\n"
        f"{task_prompt.post_prompt}\n\n"
        f"<context>\n{context.strip()}\n</context>\n\n"
        f"Question: {question}"
    ).strip()
