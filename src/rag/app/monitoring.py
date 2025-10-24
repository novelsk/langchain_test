# Мониторинг работы системы
from datetime import datetime


class RAGMonitor:
    def __init__(self):
        self.queries = []

    def log_query(self, question, answer, confidence, sources, response_time):
        self.queries.append({
            "timestamp": datetime.now(),
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "sources": [doc.metadata.get('source', 'unknown') for doc in
                        sources],
            "response_time": response_time
        })

    def get_stats(self):
        total_queries = len(self.queries)
        if total_queries == 0:
            return {
                "total_queries": 0,
                "high_confidence_rate": 0,
                "average_response_time": 0
            }

        high_confidence = len(
            [q for q in self.queries if q["confidence"] == "high"])
        avg_response_time = sum(
            [q["response_time"] for q in self.queries]) / total_queries

        return {
            "total_queries": total_queries,
            "high_confidence_rate": high_confidence / total_queries,
            "average_response_time": avg_response_time
        }

    def print_stats(self):
        stats = self.get_stats()
        print(f"\n{'=' * 50}")
        print("СТАТИСТИКА СИСТЕМЫ:")
        print(f"Всего запросов: {stats['total_queries']}")
        print(
            f"Доля ответов с высокой уверенностью: {stats['high_confidence_rate']:.2%}")
        print(
            f"Среднее время ответа: {stats['average_response_time']:.2f} сек")
        print(f"{'=' * 50}")
