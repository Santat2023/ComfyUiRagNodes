from qdrant_client import QdrantClient


class QdrantRepository:
    def __init__(self, host="localhost", port=6333):
        self.client = QdrantClient(host=host, port=port)

    def search(self, collection, vector, limit=5):
        return self.client.search(
            collection_name=collection,
            query_vector=vector,
            limit=limit
        )

    def list_collections(self):
        return [c.name for c in self.client.get_collections().collections]