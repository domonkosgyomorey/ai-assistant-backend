from chunk_gatherer.db_gatherer import DBGatherer
from chunk_gatherer.web_gatherer import WebGatherer


class ChunkGathererOrchestrator:
    def __init__(self, db_gatherer: DBGatherer, web_gatherer: WebGatherer):
        self.db_gatherer = db_gatherer
        self.web_gatherer = web_gatherer

    def gather_chunks(self, key: str) -> str:
        db_chunks = self.db_gatherer.gather(key)
        if not db_chunks:
            return self.web_gatherer.gather(key)
        return db_chunks
