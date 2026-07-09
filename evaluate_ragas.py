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
    "What is kidney failure?",
    "What is osteoporosis?",
    "What causes migraine?",
    "What is cataract?",
    "How is anemia diagnosed?",
    "What are the symptoms of Parkinson's disease?"
]


def evaluate_with_llm(question, answer, sources):
    prompt = f"""
You are an evaluator for a medical RAG system.

Evaluate the generated answer based ONLY on the retrieved context.

Question:
{question}

Generated Answer:
{answer}

Retrieved Context:
{sources}

Score each metric from 1 to 5:
- answer_relevance: Does the answer directly answer the question?
- faithfulness: Is the answer fully supported by the retrieved context?
- retrieval_quality: Are the retrieved chunks relevant and useful?

Return ONLY valid JSON:
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

    text = response.choices[0].message.content.strip()

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
    print(f"Evaluating: {question}")

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

df_results.to_csv("evaluation_results.csv", index=False)

summary = df_results[
    ["answer_relevance", "faithfulness", "retrieval_quality"]
].mean()

summary_df = summary.reset_index()
summary_df.columns = ["metric", "average_score"]

summary_df.to_csv("evaluation_summary.csv", index=False)

print("\nEvaluation Results:")
print(df_results)

print("\nAverage Scores:")
print(summary_df)

print("\nSaved files:")
print("- evaluation_results.csv")
print("- evaluation_summary.csv")