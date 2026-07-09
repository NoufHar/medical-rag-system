import os
import json
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from rag import generate_answer

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

test_questions = [
    "What is glaucoma?",
    "What are the symptoms of asthma?",
    "How is diabetes treated?",
    "What causes high blood pressure?",
    "What is kidney failure?"
]

def evaluate_with_llm(question, answer, sources):
    prompt = f"""
You are an evaluator for a medical RAG system.

Evaluate the answer based on the retrieved context only.

Question:
{question}

Generated Answer:
{answer}

Retrieved Context:
{sources}

Give scores from 1 to 5 for:
1. answer_relevance: Does the answer directly answer the question?
2. faithfulness: Is the answer supported by the retrieved context?
3. retrieval_quality: Are the retrieved chunks relevant to the question?

Return ONLY valid JSON in this format:
{{
  "answer_relevance": 0,
  "faithfulness": 0,
  "retrieval_quality": 0,
  "comments": "short explanation"
}}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300
    )

    text = response.choices[0].message.content

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "answer_relevance": None,
            "faithfulness": None,
            "retrieval_quality": None,
            "comments": text
        }


results = []

for question in test_questions:
    answer, sources = generate_answer(question)
    evaluation = evaluate_with_llm(question, answer, sources)

    results.append({
        "question": question,
        "answer": answer,
        "answer_relevance": evaluation["answer_relevance"],
        "faithfulness": evaluation["faithfulness"],
        "retrieval_quality": evaluation["retrieval_quality"],
        "comments": evaluation["comments"]
    })

df_results = pd.DataFrame(results)

print(df_results)

df_results.to_csv("evaluation_results.csv", index=False)

print("\nEvaluation saved to evaluation_results.csv")