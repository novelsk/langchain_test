# main.py
import time
from langchain_core.documents import Document

from config.values import GPT_OSS
from rag.app.document import DocumentProcessor
from rag.app.monitoring import RAGMonitor
from rag.app.rag import SelfHostedRAGSystem


# Основная функция для запуска системы
def main():
    # Инициализация системы
    print("Инициализация RAG системы...")
    rag_system = SelfHostedRAGSystem(
        model_name=GPT_OSS)  # или "mistral", "codellama" и т.д.
    monitor = RAGMonitor()

    # Создаем тестовые документы (в реальном проекте загружайте из файлов)
    documents = [
        Document(
            page_content="""
            Машинное обучение - это раздел искусственного интеллекта, 
            который позволяет компьютерам обучаться на данных без явного программирования.
            Основные типы машинного обучения: supervised learning, unsupervised learning и reinforcement learning.
            """,
            metadata={"source": "ml_intro"}
        ),
        Document(
            page_content="""
            Глубокое обучение использует нейронные сети с множеством слоев.
            Популярные фреймворки: TensorFlow, PyTorch и Keras.
            CNN используются для обработки изображений, RNN - для последовательных данных.
            """,
            metadata={"source": "deep_learning"}
        ),
        Document(
            page_content="""
            LangChain - это фреймворк для создания приложений с языковыми моделями.
            Он предоставляет инструменты для работы с памятью, цепочками и агентами.
            LangGraph расширяет LangChain для создания графов выполнения.
            """,
            metadata={"source": "langchain_info"}
        )
    ]

    # Обрабатываем документы
    processor = DocumentProcessor()
    processed_docs = processor.split_documents(documents)

    # Загружаем документы в систему
    rag_system.load_documents(processed_docs)

    print("RAG система готова к работе!")
    print(
        "Доступные модели через Ollama можно посмотреть командой: ollama list")

    # Интерактивный режим
    while True:
        question = input(
            "\nВведите ваш вопрос (или 'quit' для выхода, 'stats' для статистики): ")

        if question.lower() == 'quit':
            break
        elif question.lower() == 'stats':
            monitor.print_stats()
            continue

        start_time = time.time()
        result = rag_system.query(question)
        response_time = time.time() - start_time

        # Логируем запрос
        monitor.log_query(
            question=question,
            answer=result["answer"],
            confidence=result["confidence"],
            sources=result["sources"],
            response_time=response_time
        )

        # Выводим результат
        print(f"\n{'=' * 50}")
        print(f"Вопрос: {question}")
        print(f"{'=' * 50}")
        print(f"Ответ: {result['answer']}")
        print(f"\nУверенность: {result['confidence']}")
        print(f"Время ответа: {response_time:.2f} сек")

        # Показываем источники
        if result["sources"]:
            print(f"\nИсточники:")
            for i, doc in enumerate(result["sources"], 1):
                source = doc.metadata.get('source', 'unknown')
                preview = doc.page_content[:100] + "..." if len(
                    doc.page_content) > 100 else doc.page_content
                print(f"  {i}. {source}: {preview}")


if __name__ == "__main__":
    main()
