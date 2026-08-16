class RetrievalService:
    def __init__(self, clip_model, qdrant_repo, s3_repo):
        self.clip = clip_model
        self.qdrant = qdrant_repo
        self.s3 = s3_repo

    def find_image(self, query: str, collection: str):
        vector = self.clip.embed_text(query)
        results = self.qdrant.search(collection, vector, limit=1)

        if not results:
            raise ValueError("No results")

        hit = results[0]
        key = f"{hit.id}_{hit.payload.get('filename')}"
        return self.s3.get_image_bytes(key)