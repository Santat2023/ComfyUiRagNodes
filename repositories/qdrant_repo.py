from qdrant_client import QdrantClient


class QdrantRepository:
    def __init__(self, host="localhost", port=6333):
        self.client = QdrantClient(
            host=host,
            port=port
        )

    def search(
        self,
        collection,
        vector,
        limit=5,
        using=None
    ):
        """
        Search a named vector in Qdrant.

        Parameters
        ----------
        collection : str
            Qdrant collection name.

        vector : list/np.ndarray
            Query embedding.

        limit : int
            Number of results.

        using : str | None
            Named vector:
                "auto"
                "manual"

            None is used for legacy single-vector collections.
        """

        if using is None:
            response = self.client.query_points(
                collection_name=collection,
                query=vector,
                limit=limit,
                with_payload=True
            )
        else:
            response = self.client.query_points(
                collection_name=collection,
                query=vector,
                using=using,
                limit=limit,
                with_payload=True
            )

        return response.points

    def search_auto(
        self,
        collection,
        vector,
        limit=10
    ):
        return self.search(
            collection=collection,
            vector=vector,
            limit=limit,
            using="auto"
        )

    def search_manual(
        self,
        collection,
        vector,
        limit=10
    ):
        return self.search(
            collection=collection,
            vector=vector,
            limit=limit,
            using="manual"
        )

    def list_collections(self):
        return [
            collection.name
            for collection in self.client.get_collections().collections
        ]