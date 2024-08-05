import os
# set OPENAI_API_KEY environment variable
os.environ["OPENAI_API_KEY"] = "sk-"
import sys
import pandas as pd
# Import necessary libraries
import autogen
from opto.trace.nodes import node, GRAPH, ParameterNode
from opto.optimizers import OptoPrime
from datasets import load_dataset
from textwrap import dedent
from opto.trace.bundle import bundle
from opto.trace.modules import model
from opto.trace.errors import ExecutionError
from opto.trace.nodes import ExceptionNode
from typing import List
import re

def eval_metric(true, prediction):
    matches = re.findall(r"\([A-Z]\)", true)
    if matches:
        pred = str(prediction)
        matches = re.findall(r"\([A-Z]\)", pred)
        parsed_answer = matches[-1] if matches else ""
        return parsed_answer == true
    else:
        return prediction == true

class LLMCallable:
    def __init__(self, config_list=None, max_tokens=1024, verbose=False):
        if config_list is None:
            config_list = autogen.config_list_from_json("OAI_CONFIG_LIST_gpt-4o-mini")
        self.llm = autogen.OpenAIWrapper(config_list=config_list)
        self.max_tokens = max_tokens
        self.verbose = verbose

    @bundle(catch_execution_error=True)
    def call_llm(self, user_prompt):
        system_prompt = "You are a helpful assistant.\n"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = self.llm.create(messages=messages, max_tokens=self.max_tokens)
        response = response.choices[0].message.content

        if self.verbose:
            print("LLM response:\n", response)
        return response

import trace

@model
class PredictPaper(LLMCallable):
    def __init__(self):
        super().__init__()

        self.demos = []
        self.prompt_template = dedent(
            """
        Given the fields `question`, produce the fields `answer`.

        ---

        Follow the following format.

        Question: question
        Reasoning : Let ’s think step by step in order to produce the answer . We ...
        Answer: answer

        ---
        Question: {}
        """
        )
        self.prompt_template = node(self.prompt_template, trainable=True,
                                    description="[ ParameterNode ] This is the Prompt Template to the LLM ...")

    @bundle(trainable=True, catch_execution_error=True, allow_external_dependencies=True)
    def extract_answer(self, prompt_template, question, response):
        """
        Need to read in the response , which can contain additional thought , delibration and an answer .
        Use code to process the response and find where the answer is.
        Can use self.call_llm ("Return the answer from this text: " + response) again to refine the answer if necessary.

        Args:
            response: LLM returned a string response
                      Process it and return the answer in the exact format that the evaluator wants to see .
                      Be mindful of the type of answer you need to produce .
                      It can be (A) /( B) , a number like 8, or a string , or Yes / No .
            question: Question has a text describing the question but also " Options "
        """
        answer = response.split("Answer:")[1].strip()
        return answer

    @bundle(trainable=True, catch_execution_error=True, allow_external_dependencies=True)
    def create_prompt(self, prompt_template, question):
        """
        The function takes in a question and then add to the prompt for LLM to answer .
        The prompt should instruct the LLM to reason , think .
        Args:
            prompt_template: some guidance / hints / suggestions for LLM
            question: the question for the LLM to answer
        """
        return prompt_template.format(question)

    def forward(self, question):
        """
        question : text
        
        We read in a question and produces a resposne
        """
        user_prompt = self.create_prompt(self.prompt_template, question)
        response = self.call_llm(user_prompt)
        answer = self.extract_answer(self.prompt_template, question, response)
        return answer

@model
class Predict(LLMCallable):
    def __init__(self):
        super().__init__()

        self.demos = []
        self.prompt_template = dedent(
            """
        Given the fields `question`, produce the fields `answer`.

        ---

        Follow the following format.

        Question: 
        Answer: 

        ---
        Question: {}
        Answer:
        """
        )
        self.prompt_template = ParameterNode(self.prompt_template, trainable=True,
                                             description="This is the Prompt Template to the LLM. " + \
                                                         "Need to include information about what the format of answers LLM should output. " + \
                                                         "They can be (A)/(B), a number like 8, or a string, or Yes/No.")

    @bundle(trainable=True, catch_execution_error=True, allow_external_dependencies=True)
    def extract_answer(self, prompt_template, question, response):
        answer = response.split("Answer:")[1].strip()
        return answer

    @bundle(trainable=True, catch_execution_error=True, allow_external_dependencies=True)
    def create_prompt(self, prompt_template, question):
        return prompt_template.format(question)

    def forward(self, question):
        user_prompt = self.create_prompt(self.prompt_template, question)
        response = self.call_llm(user_prompt)
        answer = self.extract_answer(self.prompt_template, question, response)
        return answer

def learn_predict(dp, optimizer, examples):
    for step, example in enumerate(examples):
        GRAPH.clear()
        try:
            response = dp.forward(example['question'])
            correctness = eval_metric(example['answer'], response)
            feedback = "The answer is correct! No need to change anything." if correctness else f"The answer is wrong. We expect the output of your answer to be \"{example['answer']}\". Please modify the prompt and relevant parts of the program to help LLM produce the right answer."
        except ExecutionError as e:
            response = e.exception_node
            feedback = response.data
            correctness = False
            
        print("Question:", example["question"])
        print("Expected answer:", example["answer"])
        print("Answer:", response)

        if correctness:
            continue

        optimizer.zero_feedback()
        optimizer.backward(response, feedback)

        print(f"Output: {response}, Feedback: {feedback}, Variables:")  # Logging
        for p in optimizer.parameters:
            print(p.name, p.data)
        optimizer.step(verbose=True)
    


print(sys.argv[1:3])

if sys.argv[2] == "paper":
    dp = PredictPaper()
else:
    dp = Predict()

task = sys.argv[1]



train = load_dataset("maveriq/bigbenchhard", task)["train"]
examples = [{"question": r["input"], "answer": r["target"]} for r in train]

optimizer = OptoPrime(dp.parameters() + [dp.prompt_template],
                                    config_list=autogen.config_list_from_json("OAI_CONFIG_LIST_gpt-4-turbo"))

print("Training on a few examples:")
learn_predict(dp, optimizer, examples[:15])
    
print("\nTesting on new examples:")
eval_dict = []
for example in examples[15:]:
    try:
        response = dp.forward(example["question"])
        question = example["question"]
        golden = example["answer"]
        predicted =  response.data
        eval_dict.append({"question": question, "golden": golden, "predicted": predicted})
    except ExecutionError as e:
        pass
        print("Question:", example["question"])
        print("Expected answer:", example["answer"])
        print("Error:", e.exception_node.data)
        eval_dict.append({"question": example["question"] , "golden": example["answer"], "predicted": 'None'})

eval_df = pd.DataFrame(eval_dict)
eval_df.to_csv(f"bbh_eval_{task}_{'paper' if sys.argv[1]=='paper' else 'tutorial'}.csv", index=False)
print(f"done {task}")
