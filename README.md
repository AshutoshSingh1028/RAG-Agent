# Medical RAG Agent

This project implements a Retrieval-Augmented Generation (RAG) agent using LangGraph, Groq, and Pinecone. It features a Streamlit UI for interaction, LangSmith for tracing, and an "LLM as a Judge" script for evaluation.

## How the Agent Works

The agent is built as a "graph" or state machine using LangGraph. This allows for complex, multi-step reasoning. When a user asks a question, the agent proceeds through the following steps:

1. **Plan**: The agent first uses an LLM (Groq Llama 3.1) to analyze the user's question. It decides if the question is medical (requiring document retrieval) or a simple conversational query (like "hello"). It routes to retrieve or generate_answer accordingly.

2. **Retrieve**: If the question is medical, this node queries a Pinecone vector store (which has been populated with embeddings from a medical PDF) to find the top 3 most relevant text chunks.

3. **Generate Answer**: An LLM generates an answer. If documents were retrieved, it uses them as context (RAG). If not (for conversational queries), it answers directly. It also tracks how many times it has tried to answer this question.

4. **Reflect**: After generating an answer, another LLM call acts as a "reflector." It checks if the generated answer is actually relevant to the original question, outputting a simple `relevant` or `not_relevant`.

5. **Evaluate & Loop**: This final node acts as a conditional edge. If the reflection was relevant, the agent's run is finished and the answer is returned. If it was not_relevant (and the agent hasn't exceeded its maximum attempts), the graph loops back to the Generate Answer node, instructing the LLM to try again with a prompt that emphasizes the previous failure.

## Challenges Faced

Building this agent presented a few interesting challenges:

1. **Intelligent Routing**: A basic RAG bot will try to search its documents for every query, even "hello." Implementing the Plan node makes the agent more efficient and natural to interact with, preventing pointless and costly retrievals.

2. **Self-Correction**: LLMs can sometimes provide answers that are well-formed but miss the point of the question (hallucination or topic drift). The Reflect -> Evaluate & Loop cycle is a powerful pattern that gives the agent a chance to "check its own work" and try again, significantly improving the reliability of the final answer. Managing the state (like the `answer_attempts` counter) within the graph was critical to prevent this loop from running forever.

## How to Run

### Install Dependencies:
```bash
pip install -r requirements.txt
```

### Create .env File:
Copy the contents of `.env.example` into a new file named `.env` and fill in your API keys.

### Populate the Vector Database:
(You only need to do this once)
Make sure your `Medical_book.pdf` is in the same folder and run:
```bash
python agent.py
```
Wait for it to finish processing and embedding the document.

### Run the Streamlit App:
```bash
streamlit run app.py
```
Open the URL in your browser to chat with the agent.

### Run the Evaluation:
To evaluate the agent's performance on a predefined set of questions, run:
```bash
python evaluate.py
```
This will print a report to the console.
