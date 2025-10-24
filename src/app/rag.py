from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, END

from models.state import GraphState


class SelfHostedRAGSystem:
    def __init__(self, model_name: str,
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.vectorstore = None
        # Используем локальную модель через Ollama
        self.llm = OllamaLLM(model=model_name, temperature=0.1)
        # Используем локальные эмбеддинги
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )

        # Создаем граф
        self.graph = self._build_graph()

    def _build_graph(self):
        # Создаем builder графа с указанием типа состояния
        workflow = StateGraph(GraphState)

        # Добавляем узлы
        workflow.add_node("retrieve", self.retrieve_documents)
        workflow.add_node("generate", self.generate_answer)
        workflow.add_node("validate", self.validate_answer)

        # Устанавливаем начальную точку
        workflow.set_entry_point("retrieve")

        # Добавляем связи
        workflow.add_edge("retrieve", "generate")
        workflow.add_conditional_edges(
            "generate",
            self.should_validate,
            {
                "validate": "validate",
                "final": END
            }
        )
        workflow.add_edge("validate", END)

        # Компилируем граф
        return workflow.compile()

    def load_documents(self, documents: list[Document]):
        """Загрузка документов в векторную базу"""
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )

    def retrieve_documents(self, state: GraphState):
        """Поиск релевантных документов"""
        question = state["question"]

        # Ищем документы
        retrieved_docs = self.vectorstore.similarity_search(question, k=3)
        return {"documents": retrieved_docs}

    def generate_answer(self, state: GraphState):
        """Генерация ответа на основе найденных документов"""
        question = state["question"]
        documents = state.get("documents", [])

        if not documents:
            return {
                "answer": "Извините, в моей базе знаний нет информации по этому вопросу.",
                "confidence": "low"
            }

        # Формируем контекст из документов
        context = "\n\n".join([doc.page_content for doc in documents])

        # Создаем промпт для LLM
        prompt = f"""Ты - полезный AI-ассистент. Ответь на вопрос пользователя на основе предоставленного контекста.

Контекст:
{context}

Вопрос: {question}

Требования к ответу:
- Будь точным и информативным
- Используй только информацию из контекста
- Если информации недостаточно, скажи об этом
- Ответ должен быть на русском языке

Ответ:"""

        # Получаем ответ от локальной LLM
        response = self.llm.invoke(prompt)

        return {
            "answer": response,
            "confidence": "medium"
        }

    def validate_answer(self, state: GraphState):
        """Проверка качества ответа"""
        answer = state["answer"]
        question = state["question"]

        # Простая проверка на наличие ключевых фраз, указывающих на неопределенность
        uncertainty_phrases = [
            "не знаю", "не найдено", "нет информации",
            "не уверен", "не могу", "извините"
        ]

        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            state["confidence"] = "low"
            state[
                "answer"] = f"{answer}\n\n⚠️ Примечание: Ответ может быть неполным или требовать проверки."
        else:
            state["confidence"] = "high"

        return state

    def should_validate(self, state: GraphState):
        """Определяет, нужно ли проверять ответ"""
        # Всегда проверяем для демонстрации
        return "validate"

    def query(self, question: str) -> dict:
        """Основной метод для запроса к системе"""
        # Инициализируем состояние
        initial_state = {"question": question}

        # Выполняем граф
        final_state = self.graph.invoke(initial_state)

        return {
            "answer": final_state["answer"],
            "confidence": final_state.get("confidence", "unknown"),
            "sources": final_state.get("documents", [])
        }