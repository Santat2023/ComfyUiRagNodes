class RetrievalService:

    DEFAULT_AUTO_WEIGHT = 0.3
    DEFAULT_MANUAL_WEIGHT = 0.7

    def __init__(
        self,
        clip_model,
        qdrant_repo,
        s3_repo
    ):
        self.clip = clip_model
        self.qdrant = qdrant_repo
        self.s3 = s3_repo

    def find_image(
        self,
        query: str,
        collection: str,
        auto_weight: float = DEFAULT_AUTO_WEIGHT,
        manual_weight: float = DEFAULT_MANUAL_WEIGHT,
        search_limit: int = 10
    ):
        """
        Search image using automatic and manual vectors.

        If a point has both vectors:

            final =
                auto_weight * auto_score +
                manual_weight * manual_score

        If a point has no manual vector:

            final = auto_score

        This means missing manual annotation does not penalize
        the image.
        """

        if not query or not query.strip():
            raise ValueError(
                "Search query cannot be empty"
            )

        if auto_weight < 0 or manual_weight < 0:
            raise ValueError(
                "Search weights cannot be negative"
            )

        if auto_weight == 0 and manual_weight == 0:
            raise ValueError(
                "At least one search weight must be greater than zero"
            )

        # --------------------------------------------------
        # Normalize weights
        # --------------------------------------------------

        weight_sum = auto_weight + manual_weight

        normalized_auto_weight = (
            auto_weight / weight_sum
        )

        normalized_manual_weight = (
            manual_weight / weight_sum
        )

        # --------------------------------------------------
        # Query embedding
        # --------------------------------------------------

        vector = self.clip.embed_text(query)

        # --------------------------------------------------
        # AUTO search
        # --------------------------------------------------

        auto_results = []

        if auto_weight > 0:
            auto_results = self.qdrant.search_auto(
                collection=collection,
                vector=vector,
                limit=search_limit
            )

        # --------------------------------------------------
        # MANUAL search
        # --------------------------------------------------

        manual_results = []

        if manual_weight > 0:
            manual_results = self.qdrant.search_manual(
                collection=collection,
                vector=vector,
                limit=search_limit
            )

        # --------------------------------------------------
        # Merge results
        # --------------------------------------------------

        merged = {}

        for hit in auto_results:
            point_id = str(hit.id)

            merged.setdefault(
                point_id,
                {
                    "id": hit.id,
                    "payload": hit.payload or {},
                    "auto_score": None,
                    "manual_score": None
                }
            )

            merged[point_id]["auto_score"] = float(
                hit.score
            )

        for hit in manual_results:
            point_id = str(hit.id)

            merged.setdefault(
                point_id,
                {
                    "id": hit.id,
                    "payload": hit.payload or {},
                    "auto_score": None,
                    "manual_score": None
                }
            )

            merged[point_id]["manual_score"] = float(
                hit.score
            )

        if not merged:
            raise ValueError(
                "No results found"
            )

        # --------------------------------------------------
        # Calculate final score
        # --------------------------------------------------

        ranked_results = []

        for result in merged.values():

            auto_score = result["auto_score"]
            manual_score = result["manual_score"]

            # ------------------------------------------------
            # IMPORTANT:
            #
            # No manual vector means:
            #
            #     final = auto_score
            #
            # We do NOT do:
            #
            #     0.3 * auto + 0.7 * 0
            #
            # because missing manual annotation is not
            # negative information.
            # ------------------------------------------------

            if manual_score is None:

                if auto_score is None:
                    continue

                final_score = auto_score

            elif auto_score is None:

                final_score = manual_score

            else:

                final_score = (
                    normalized_auto_weight * auto_score
                    +
                    normalized_manual_weight * manual_score
                )

            result["final_score"] = float(
                final_score
            )

            ranked_results.append(result)

        # --------------------------------------------------
        # Sort by final score
        # --------------------------------------------------

        ranked_results.sort(
            key=lambda item: item["final_score"],
            reverse=True
        )

        if not ranked_results:
            raise ValueError(
                "No valid search results"
            )

        # --------------------------------------------------
        # Best result
        # --------------------------------------------------

        best = ranked_results[0]

        print(
            "[RetrievalService] "
            f"Query: {query}"
        )

        print(
            "[RetrievalService] "
            f"Auto score: {best['auto_score']}"
        )

        print(
            "[RetrievalService] "
            f"Manual score: {best['manual_score']}"
        )

        print(
            "[RetrievalService] "
            f"Final score: {best['final_score']}"
        )

        payload = best["payload"]

        key = self._build_s3_key(
            best["id"],
            payload
        )

        print(
            "[RetrievalService] "
            f"Selected image: {key}"
        )

        return self.s3.get_image_bytes(key)

    @staticmethod
    def _build_s3_key(point_id, payload):
        filename = payload.get(
            "filename",
            "unknown"
        )

        return f"{point_id}_{filename}"