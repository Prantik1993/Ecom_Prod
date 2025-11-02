# rag_chain.py
from operator import itemgetter
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from flipkart.config import Config


class RAGChainBuilder:
    """Builds a retrieval-augmented generation chain using AstraDB retriever + Groq LLM."""

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.model = ChatGroq(model=Config.RAG_MODEL, temperature=0.5)
        self.history_store: dict[str, BaseChatMessageHistory] = {}

    def _get_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]

    def build_chain(self):
        retriever = self.vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 40, "score_threshold": 0.25}
)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Ecommerce product expert.
Use the CONTEXT (from AstraDB) to answer the QUESTION accurately.
Do NOT invent products that aren’t listed.

CONTEXT:
{context}

QUESTION: {input}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        retriever_chain = (
            itemgetter("input")
            | retriever
            | (lambda docs: {"context": "\n\n".join(d.page_content for d in docs)})
        )

        full_chain = (
            {
                "context": retriever_chain,
                "input": itemgetter("input"),
                "chat_history": itemgetter("chat_history"),
            }
            | qa_prompt
            | self.model
            | StrOutputParser()
        )

        return RunnableWithMessageHistory(
            full_chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
