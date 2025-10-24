# Утилиты для работы с документами
from langchain_core.documents import Document


class DocumentProcessor:
    @staticmethod
    def load_text_files(file_paths: list[str]):
        """Загрузка текстовых файлов"""
        documents = []
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": file_path}
                    ))
            except Exception as e:
                print(f"Ошибка загрузки файла {file_path}: {e}")
        return documents

    @staticmethod
    def split_documents(
            documents: list[Document],
            chunk_size=1000,
            chunk_overlap=200
    ):
        """Разбивка документов на чанки"""
        from langchain_text_splitters  import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(documents)
