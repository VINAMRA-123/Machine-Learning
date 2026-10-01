from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled,NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

video_id = "J5_-l7WIO_w"

api = YouTubeTranscriptApi()

load_dotenv()

try:
    transcript = api.fetch(video_id, languages=["hi"])

    transcript_list = transcript.to_raw_data()

    transcript = " ".join(
        chunk["text"] for chunk in transcript_list
    )

    print("Transcript fetched successfully!")
    print(transcript[:1000])

except TranscriptsDisabled:
    print("No captions available for this video.")

except NoTranscriptFound:
    print("Hindi transcript not found.")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.create_documents([transcript])

print(f"Number of chunks: {len(chunks)}")


# -----------------------------
# 3. Create Embeddings
# -----------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# -----------------------------
# 4. Create FAISS Vector Store
# -----------------------------

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)


# -----------------------------
# 5. Create Retriever
# -----------------------------

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


# -----------------------------
# 6. LLM
# -----------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.2
)


# -----------------------------
# 7. Prompt
# -----------------------------

prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Answer ONLY from the provided transcript context.

If the context is insufficient, just say you don't know.

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)


# -----------------------------
# 8. Format Retrieved Documents
# -----------------------------

def format_docs(retrieved_docs):
    return "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )


# -----------------------------
# 9. Parallel Chain
# -----------------------------

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})


# -----------------------------
# 10. Final Chain
# -----------------------------

parser = StrOutputParser()

main_chain = (
    parallel_chain
    | prompt
    | llm
    | parser
)


# -----------------------------
# 11. Ask Question
# -----------------------------

question = """
Is the topic of augmentation discussed in this video?
If yes, what was discussed?
"""

answer = main_chain.invoke(question)

print("\nANSWER:")
print(answer)