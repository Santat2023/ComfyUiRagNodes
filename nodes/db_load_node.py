from ..services.retrieval_service import RetrievalService
from ..repositories.qdrant_repo import QdrantRepository
from ..repositories.s3_repo import S3Repository
from ..models.clip_model import ClipModel

import node_helpers

from PIL import Image, ImageOps, ImageSequence

import numpy as np
import torch
import io


class DB_Load_Node:

    def __init__(self):

        self.service = RetrievalService(
            clip_model=ClipModel(),
            qdrant_repo=QdrantRepository(),
            s3_repo=S3Repository(
                endpoint="http://localhost:9000",
                access_key="minioadmin",
                secret_key="minioadmin",
                bucket="images"
            )
        )

    @classmethod
    def INPUT_TYPES(cls):

        try:

            repo = QdrantRepository()

            collections = repo.list_collections()

            if not collections:
                collections = [
                    "empty_collection"
                ]

        except Exception as e:

            print(
                "⚠️ Ошибка получения коллекций "
                f"из Qdrant: {e}"
            )

            collections = [
                "no_collections_found"
            ]

        return {
            "required": {

                # -----------------------------------------
                # POSE
                # -----------------------------------------

                "pose_query": (
                    "STRING",
                    {
                        "forceInput": True
                    }
                ),

                "pose_collection": (
                    collections,
                ),

                # -----------------------------------------
                # STYLE
                # -----------------------------------------

                "style_query": (
                    "STRING",
                    {
                        "forceInput": True
                    }
                ),

                "style_collection": (
                    collections,
                ),

                # -----------------------------------------
                # SEARCH WEIGHTS
                # -----------------------------------------

                "auto_weight": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05
                    }
                ),

                "manual_weight": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05
                    }
                ),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
    )

    RETURN_NAMES = (
        "Pose image",
        "Style image",
    )

    FUNCTION = "load_from_db"

    CATEGORY = "MyNodes"

    def load_from_db(
        self,
        pose_query,
        pose_collection,
        style_query,
        style_collection,
        auto_weight,
        manual_weight
    ):

        print(
            "[DB_Load_Node] "
            f"Auto weight: {auto_weight}"
        )

        print(
            "[DB_Load_Node] "
            f"Manual weight: {manual_weight}"
        )

        # -----------------------------------------
        # POSE
        # -----------------------------------------

        print(
            "[DB_Load_Node] "
            f"Pose query: {pose_query}"
        )

        pose_file_bytes = self.service.find_image(
            query=pose_query,
            collection=pose_collection,
            auto_weight=auto_weight,
            manual_weight=manual_weight
        )

        pose_image = self.load_image(
            pose_file_bytes
        )

        # -----------------------------------------
        # STYLE
        # -----------------------------------------

        print(
            "[DB_Load_Node] "
            f"Style query: {style_query}"
        )

        style_file_bytes = self.service.find_image(
            query=style_query,
            collection=style_collection,
            auto_weight=auto_weight,
            manual_weight=manual_weight
        )

        style_image = self.load_image(
            style_file_bytes
        )

        return (
            pose_image,
            style_image
        )

    def load_image(self, file_bytes):

        img = node_helpers.pillow(
            Image.open,
            io.BytesIO(file_bytes)
        )

        output_images = []
        output_masks = []

        w, h = None, None

        for i in ImageSequence.Iterator(img):

            i = node_helpers.pillow(
                ImageOps.exif_transpose,
                i
            )

            image = i.convert("RGB")

            if len(output_images) == 0:
                w, h = image.size

            if image.size != (w, h):
                continue

            image = (
                np.array(image)
                .astype(np.float32)
                / 255.0
            )

            image = torch.from_numpy(
                image
            )[None,]

            if "A" in i.getbands():

                mask = (
                    np.array(
                        i.getchannel("A")
                    )
                    .astype(np.float32)
                    / 255.0
                )

                mask = (
                    1.
                    - torch.from_numpy(mask)
                )

            else:

                mask = torch.zeros(
                    (64, 64),
                    dtype=torch.float32
                )

            output_images.append(image)

            output_masks.append(
                mask.unsqueeze(0)
            )

        if not output_images:

            raise ValueError(
                "Не удалось загрузить изображение"
            )

        if len(output_images) > 1:

            return torch.cat(
                output_images,
                dim=0
            )

        return output_images[0]