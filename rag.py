import os
import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

df = pd.read_csv("data/medquad_cleaned.csv")
df = df.head(5000)
df = df.dropna(subset=["question", "answer"]).reset_index(drop=True)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=400
)

documents = []

for _, row in df.iterrows():
    question = str(row["question"])
    answer = str(row["answer"])
    source = str(row.get("source", "Unknown"))
    focus_area = str(row.get("focus_area", "Unknown"))

    if len(answer) > 3000:
        chunks = splitter.split_text(answer)

        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=f"Question: {question}\nAnswer: {chunk}",
                    metadata={
                        "source": source,
                        "focus_area": focus_area,
                        "chunk_id": i
                    }
                )
            )
    else:
        documents.append(
            Document(
                page_content=f"Question: {question}\nAnswer: {answer}",
                metadata={
                    "source": source,
                    "focus_area": focus_area,
                    "chunk_id": 0
                }
            )
        )

print(f"Original rows: {len(df)}")
print(f"Total documents after chunking: {len(documents)}")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    show_progress=True
)

print("Embedding model loaded.")

index_path = "faiss_index"

if os.path.exists(index_path):
    print("Loading existing FAISS index...")
    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("FAISS index loaded successfully!")
else:
    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(index_path)
    print("FAISS index created and saved successfully!")

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    max_tokens=500,
    api_key=os.getenv("GROQ_API_KEY")
)

prompt_template = ChatPromptTemplate.from_template("""
You are a medical educational assistant.

Answer the question using ONLY the context below.
If the answer is not found in the context, say:
"I don't know based on the provided data."

Context:
{context}

Question:
{question}
""")

def format_docs(docs):
    return "\n\n---\n\n".join(
        [
            f"{doc.page_content}\nSource: {doc.metadata.get('source')}\nFocus Area: {doc.metadata.get('focus_area')}"
            for doc in docs
        ]
    )

def generate_answer(question):
    retrieved_docs = retriever.invoke(question)

    context = format_docs(retrieved_docs)

    prompt = prompt_template.format(
        context=context,
        question=question
    )

    response = llm.invoke(prompt)

    sources = format_docs(retrieved_docs)

    return response.content, sources