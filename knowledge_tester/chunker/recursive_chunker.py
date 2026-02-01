from chunker.chunker import Chunker
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RecursiveChunker(Chunker):
    @classmethod
    def chunk(cls, documents: list[Document]) -> list[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        return text_splitter.split_documents(documents)
