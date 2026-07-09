import gradio as gr
from rag import generate_answer

def rag_chat(question):
    answer, sources = generate_answer(question)
    return answer, sources

demo = gr.Interface(
    fn=rag_chat,
    inputs=gr.Textbox(
        label="Ask a medical question",
        placeholder="Example: What is glaucoma?"
    ),
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Retrieved Sources")
    ],
    title="Medical RAG System",
    description="A Retrieval-Augmented Generation system using MedQuAD dataset, LangChain, FAISS, HuggingFace embeddings, and Groq LLM.",
    examples=[
        ["What is glaucoma?"],
        ["What are the symptoms of asthma?"],
        ["How is diabetes treated?"]
    ]
)

demo.launch()