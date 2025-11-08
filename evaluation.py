import json
from agent import app, llm # Import the app and llm from our agent file
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import pandas as pd

# Define the Judge's Output Structure
class Evaluation(BaseModel):
    relevance: int = Field(description="Score 1-5 for how relevant the answer is to the question. 1=Not relevant, 5=Highly relevant.")
    conciseness: int = Field(description="Score 1-5 for how concise the answer is. 1=Too verbose, 5=Perfectly concise.")
    comment: str = Field(description="The judge's brief justification for the scores.")

# Setup the Judge 
print("Setting up the LLM-as-a-Judge...")

judge_llm = llm.with_structured_output(Evaluation) 

judge_prompt_template = """
You are an expert evaluator for a RAG chatbot.
Your goal is to assess the quality of the 'Answer' provided for the given 'Question'.
Provide scores for 'relevance' and 'conciseness' (1-5) and a brief 'comment'.

Question:
{question}

Answer:
{answer}
"""
judge_prompt = PromptTemplate.from_template(judge_prompt_template)
parser = JsonOutputParser(pydantic_object=Evaluation)
judge_chain = judge_prompt | judge_llm

# Define Evaluation Questions
eval_questions = [
    "What are the side effects of Aspirin?",
    "What is a common cold?",
    "How does paracetamol (acetaminophen) work?",
    "Hello, how are you?", 
    "What is the capital of France?"
]

# Run the Evaluation 
results = []
print(f"Running evaluation on {len(eval_questions)} questions...")

for i, question in enumerate(eval_questions):
    print(f"\n--- Question {i+1}/{len(eval_questions)} ---")
    print(f"Question: {question}")
    
    # 1. Run the RAG agent
    try:
        agent_input = {"question": question, "answer_attempts": 0}
        final_state = app.invoke(agent_input)
        answer = final_state.get("answer", "No answer generated.")
        print(f"Answer: {answer}")
        
        # 2. Run the Judge
        judge_result = judge_chain.invoke({"question": question, "answer": answer})
        print(f"Judge Score (Relevance): {judge_result.relevance}/5")
        print(f"Judge Score (Conciseness): {judge_result.conciseness}/5")
        print(f"Judge Comment: {judge_result.comment}")
        
        results.append({
            "question": question,
            "answer": answer,
            "relevance_score": judge_result.relevance,
            "conciseness_score": judge_result.conciseness,
            "judge_comment": judge_result.comment
        })

    except Exception as e:
        print(f"Error during evaluation of question: {question}")
        print(f"Error: {e}")
        results.append({
            "question": question,
            "answer": "ERROR",
            "relevance_score": 0,
            "conciseness_score": 0,
            "judge_comment": str(e)
        })

# Display Final Report
print("\n\n--- EVALUATION COMPLETE ---")
df = pd.DataFrame(results)
print(df)

# Calculate and print averages
if not df.empty:
    avg_relevance = df["relevance_score"].mean()
    avg_conciseness = df["conciseness_score"].mean()
    print("\n--- Average Scores ---")
    print(f"Average Relevance: {avg_relevance:.2f} / 5")
    print(f"Average Conciseness: {avg_conciseness:.2f} / 5")
