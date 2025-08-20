from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

import os
load_dotenv()
api_key= os.getenv("gemini")

model = ChatGoogleGenerativeAI(temperature=0, model="gemma-3-27b-it", api_key= api_key)

class TweetState(TypedDict):
    topic: str
    tweet: str
    evaluation: Literal["approved", "needs improvement "]
    feedback: str
    iteration: int
    max_iterations: int

def generate_tweet(state: TweetState) -> TweetState:
    messages = [
        SystemMessage( content=" You are funny and clever Twitter/ Influencer"),
        HumanMessage(content=f"""Generate a tweet about 
                     {state['topic']}. Keep it short and engaging. 
                     Use hashtags where appropriate.
                     Rules:
                     - Do not use question - answer format
                     - max 280 characters
                     - Use humor, irony,sarcasm or cultutal references
                     - Use meme logic
                     - Use Simple English""")
    ]
    response = model.invoke(messages).content

    return {"tweet": response}

from pydantic import BaseModel, Field

class Evaluation(BaseModel):
    evaluation: Literal["approved", "needs improvement"] =  Field(..., description="Evaluation of the tweet")
    feedback: str = Field(..., description="Feedback for the tweet")
    
   def evaluate_tweet(state: TweetState) -> TweetState:
    messages = [
        SystemMessage(content="You are a social media expert"),
        HumanMessage(content=f"""Evaluate the following tweet:
                     {state['tweet']}
                     use the following criteria:
                     - Is it engaging?, Is this fresh?
                     - Does it use humor, irony, sarcasm or cultural references?
                     virality potential
                     - Is it concise and clear?
                     - Does it use meme logic?
                     - Does it use simple English?
                     - Is it within 280 characters?
                     Auto reject if:
                     - it is in questin answer format
                     - exceeds 280 characters
                     - it is not engaging
                     ##respond only in Structured format:
                     # {"evaluation": "approved" or "needs improvement",
                     # "feedback": "Your feedback strengths and weakness",}
                     """)
    ]
    response = model.invoke(messages).content

    return {"evaluation": "needs improvement", "feedback": response, "iteration": state['iteration'] + 1}


def optimize_tweet(state: TweetState) -> TweetState:
    messages = [
        SystemMessage(content="You are a social media expert You punch up tweets to make them more engaging and funny"),
        HumanMessage(content=f"""Optimize the following tweet based on the feedback:
                     {state['tweet']}
                     Feedback: {state['feedback']}
                     Rules:
                     - Do not use question - answer format
                     - max 280 characters
                     - Use humor, irony,sarcasm or cultutal references
                     - Use meme logic
                     - Use Simple English""")
    ]
    response = model.invoke(messages).content

    return {"tweet": response, "iteration": state['iteration'] + 1}


def route_evalution(state: TweetState) -> TweetState:
    if state['evaluation'] == "approved":
        return state
    elif state['iteration'] < state['max_iterations']:
        return {"topic": state['topic'], "tweet": state['tweet'], "evaluation": "needs improvement", 
                "feedback": state['feedback'], "iteration": state['iteration'], 
                "max_iterations": state['max_iterations']}
    else:
        return {"topic": state['topic'], "tweet": state['tweet'], "evaluation": "needs improvement", 
                "feedback": "Max iterations reached. Please review manually.", 
                "iteration": state['iteration'], "max_iterations": state['max_iterations']}


graph = StateGraph(TweetState)

graph.add_node("generate", generate_tweet)
graph.add_node("evaluate", evaluate_tweet)
graph.add_node("optimize", optimize_tweet)

graph.add_edge(START, "generate")
graph.add_edge("generate", "evaluate")
graph.add_conditional_edges("evaluate", route_evalution, {'approved': END, 'needs improvement': "optimize"})
graph.add_edge("optimize", "evaluate")

work_flow = graph.compile()

# Define the initial state correctly
initial_state = {
    "topic": "AI and its impact on society",
    "iteration": 0,
    "max_iterations": 5
}

# Invoke the workflow
work_flow.invoke(initial_state)

