from comfy.comfy_types import IO

from ..services.llm_service import LLMService


class LLM_Node:

    def __init__(self):

        self.llm = LLMService()

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {

                "user_initial_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default":
                            "Enter your prompt here"
                    }
                ),

                "clip": (
                    IO.CLIP,
                    {
                        "tooltip":
                            "The CLIP model used "
                            "for encoding the text."
                    }
                )
            }
        }

    RETURN_TYPES = (
        IO.CONDITIONING,
        "STRING",
        "STRING"
    )

    RETURN_NAMES = (
        "cond",
        "pose_query",
        "style_query"
    )

    FUNCTION = "call_llm"

    CATEGORY = "MyNodes"

    def call_llm(
        self,
        user_initial_prompt,
        clip
    ):

        result = self.llm.analyze_prompt(
            user_initial_prompt
        )

        final_prompt = (
            result["final_prompt"]
        )

        pose_query = (
            result["pose_query"]
        )

        style_query = (
            result["style_query"]
        )

        # -----------------------------------------------------
        # Stable Diffusion conditioning
        # -----------------------------------------------------

        tokens = clip.tokenize(
            final_prompt
        )

        conditioning = (
            clip.encode_from_tokens_scheduled(
                tokens
            )
        )

        print(
            "[LLM_Node] FINAL PROMPT:",
            final_prompt
        )

        print(
            "[LLM_Node] POSE QUERY:",
            pose_query
        )

        print(
            "[LLM_Node] STYLE QUERY:",
            style_query
        )

        return (
            conditioning,
            pose_query,
            style_query
        )