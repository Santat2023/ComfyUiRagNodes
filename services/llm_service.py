import json
import logging
import requests


class LLMService:

    """
    Service for interaction with local Ollama.

    The LLM performs three tasks:

        1. Creates an improved final image-generation prompt.
        2. Creates a short semantic query for pose retrieval.
        3. Creates a short semantic query for style retrieval.

    Result:

        {
            "final_prompt": "...",
            "pose_query": "...",
            "style_query": "..."
        }
    """

    MAX_QUERY_WORDS = 20

    def __init__(
        self,
        base_url="http://localhost:11434",
        model="qwen2:7b",
        timeout=120
    ):

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

        # ==================================================
        # LOGGER
        # ==================================================

        self.logger = logging.getLogger(
            "LLMService"
        )

        if not self.logger.handlers:

            handler = logging.StreamHandler()

            formatter = logging.Formatter(
                "[LLMService] %(levelname)s: %(message)s"
            )

            handler.setFormatter(
                formatter
            )

            self.logger.addHandler(
                handler
            )

            self.logger.setLevel(
                logging.INFO
            )

        self.logger.info(
            "LLM service initialized. "
            "Provider=Ollama, Model=%s",
            self.model
        )

    # ======================================================
    # MAIN PUBLIC METHOD
    # ======================================================

    def analyze_prompt(
        self,
        user_prompt: str
    ) -> dict:

        """
        Analyze user prompt.

        Returns:

            {
                "final_prompt": str,
                "pose_query": str,
                "style_query": str
            }
        """

        user_prompt = str(
            user_prompt
        ).strip()

        self.logger.info(
            "================================================"
        )

        self.logger.info(
            "USER PROMPT:"
        )

        self.logger.info(
            "%s",
            user_prompt
        )

        self.logger.info(
            "================================================"
        )

        try:

            # ==================================================
            # REQUEST OLLAMA
            # ==================================================

            response = requests.post(

                f"{self.base_url}/api/generate",

                json={
                    "model": self.model,

                    "prompt": self._build_prompt(
                        user_prompt
                    ),

                    "stream": False,

                    "options": {
                        "temperature": 0.2
                    }
                },

                timeout=self.timeout
            )

            # ==================================================
            # HTTP ERROR
            # ==================================================

            if response.status_code != 200:

                self.logger.error(
                    "Ollama HTTP error: %s",
                    response.status_code
                )

                self.logger.error(
                    "Ollama response: %s",
                    response.text
                )

                return self._fallback_result(
                    user_prompt
                )

            # ==================================================
            # OLLAMA RESPONSE
            # ==================================================

            data = response.json()

            raw_text = data.get(
                "response",
                ""
            ).strip()

            self.logger.info(
                "RAW LLM RESPONSE:"
            )

            self.logger.info(
                "%s",
                raw_text
            )

            if not raw_text:

                self.logger.error(
                    "Ollama returned empty response"
                )

                return self._fallback_result(
                    user_prompt
                )

            # ==================================================
            # PARSE JSON
            # ==================================================

            result = self._parse_response(
                raw_text
            )

            # ==================================================
            # VALIDATE RETRIEVAL QUERIES
            # ==================================================

            result["pose_query"] = (
                self._validate_query_length(
                    result["pose_query"]
                )
            )

            result["style_query"] = (
                self._validate_query_length(
                    result["style_query"]
                )
            )

            # ==================================================
            # LOG RESULTS
            # ==================================================

            self.logger.info(
                "================================================"
            )

            self.logger.info(
                "FINAL PROMPT:"
            )

            self.logger.info(
                "%s",
                result["final_prompt"]
            )

            self.logger.info(
                "POSE QUERY:"
            )

            self.logger.info(
                "%s",
                result["pose_query"]
            )

            self.logger.info(
                "STYLE QUERY:"
            )

            self.logger.info(
                "%s",
                result["style_query"]
            )

            self.logger.info(
                "================================================"
            )

            return result

        # ==================================================
        # TIMEOUT
        # ==================================================

        except requests.exceptions.Timeout:

            self.logger.error(
                "Ollama request timeout after %s seconds",
                self.timeout
            )

            return self._fallback_result(
                user_prompt
            )

        # ==================================================
        # CONNECTION ERROR
        # ==================================================

        except requests.exceptions.ConnectionError:

            self.logger.error(
                "Cannot connect to Ollama at %s",
                self.base_url
            )

            return self._fallback_result(
                user_prompt
            )

        # ==================================================
        # OTHER ERROR
        # ==================================================

        except Exception as e:

            self.logger.exception(
                "Unexpected error in LLMService: %s",
                e
            )

            return self._fallback_result(
                user_prompt
            )

    # ======================================================
    # BACKWARD COMPATIBILITY
    # ======================================================

    def generate_prompt(
        self,
        user_prompt: str
    ) -> dict:

        """
        Backward-compatible alias.

        Old code may call:

            generate_prompt(...)

        New code should preferably use:

            analyze_prompt(...)
        """

        return self.analyze_prompt(
            user_prompt
        )

    # ======================================================
    # BUILD LLM PROMPT
    # ======================================================

    def _build_prompt(
        self,
        user_prompt: str
    ) -> str:

        return f"""
You are an AI assistant for an illustrator working
inside a Retrieval-Augmented Generation pipeline.

Analyze the user's original image-generation request.

Return ONLY valid JSON.

The JSON MUST contain exactly these fields:

{{
    "final_prompt": "...",
    "pose_query": "...",
    "style_query": "..."
}}


==================================================
1. FINAL PROMPT
==================================================

Create a high-quality English prompt for
Stable Diffusion / SDXL.

Preserve the meaning of the original request.

Improve the prompt by describing relevant visual
characteristics such as:

- subject
- action
- pose
- environment
- composition
- camera
- lighting
- colors
- artistic style
- important visual details

The final prompt can be detailed.

Do NOT explain your reasoning.

Do NOT use Markdown.

Do NOT add additional JSON fields.


==================================================
2. POSE QUERY
==================================================

Create a SHORT English semantic search query.

This query will be encoded using CLIP and used
to search a Qdrant database containing pose
reference images for ControlNet.

Maximum length: 20 words.

Describe ONLY visual pose information:

- body position
- limb positions
- body orientation
- action
- viewpoint
- framing

DO NOT describe:

- artistic style
- colors
- lighting
- environment
- character identity
- story
- objects

Use compact visual keywords.

GOOD:

"multiple women, dynamic yoga poses, full body, varied body positions"


==================================================
3. STYLE QUERY
==================================================

Create a SHORT English semantic search query.

This query will be encoded using CLIP and used
to search a Qdrant database containing style
reference images for IP-Adapter.

Maximum length: 20 words.

Describe ONLY visual style information:

- artistic style
- medium
- rendering technique
- linework
- color palette
- lighting aesthetics
- texture
- visual atmosphere

DO NOT describe:

- characters
- poses
- actions
- story
- specific objects
- environment

Use compact visual keywords.

GOOD:

"Victorian ink illustration, watercolor washes, detailed linework, surreal fantasy"


==================================================
IMPORTANT
==================================================

The pose_query and style_query are NOT prompts
for image generation.

They are semantic retrieval queries for CLIP.

Keep them extremely concise.

Maximum 20 words each.

Return ONLY valid JSON.

No Markdown.

No explanations.


USER PROMPT:

{user_prompt}
"""

    # ======================================================
    # PARSE RESPONSE
    # ======================================================

    def _parse_response(
        self,
        raw_text: str
    ) -> dict:

        text = raw_text.strip()

        # --------------------------------------------------
        # Remove ```json ... ```
        # --------------------------------------------------

        if text.startswith("```"):

            lines = text.splitlines()

            if lines:

                lines = lines[1:]

            if (
                lines
                and lines[-1].strip() == "```"
            ):

                lines = lines[:-1]

            text = "\n".join(
                lines
            ).strip()

        # --------------------------------------------------
        # Find JSON object
        # --------------------------------------------------

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1:

            text = text[
                start:end + 1
            ]

        # --------------------------------------------------
        # Parse JSON
        # --------------------------------------------------

        try:

            result = json.loads(
                text
            )

        except json.JSONDecodeError as e:

            self.logger.error(
                "Cannot parse LLM JSON: %s",
                e
            )

            self.logger.error(
                "Raw response: %s",
                raw_text
            )

            raise ValueError(
                "LLM returned invalid JSON"
            )

        # --------------------------------------------------
        # Extract fields
        # --------------------------------------------------

        final_prompt = str(
            result.get(
                "final_prompt",
                ""
            )
        ).strip()

        pose_query = str(
            result.get(
                "pose_query",
                ""
            )
        ).strip()

        style_query = str(
            result.get(
                "style_query",
                ""
            )
        ).strip()

        # --------------------------------------------------
        # Validate
        # --------------------------------------------------

        if not final_prompt:

            raise ValueError(
                "LLM returned empty final_prompt"
            )

        if not pose_query:

            raise ValueError(
                "LLM returned empty pose_query"
            )

        if not style_query:

            raise ValueError(
                "LLM returned empty style_query"
            )

        return {
            "final_prompt": final_prompt,
            "pose_query": pose_query,
            "style_query": style_query
        }

    # ======================================================
    # QUERY LENGTH PROTECTION
    # ======================================================

    def _validate_query_length(
        self,
        text: str
    ) -> str:

        words = text.split()

        if len(words) <= self.MAX_QUERY_WORDS:

            return text

        self.logger.warning(
            "Retrieval query contains %d words. "
            "Truncating to %d words.",
            len(words),
            self.MAX_QUERY_WORDS
        )

        return " ".join(
            words[:self.MAX_QUERY_WORDS]
        )

    # ======================================================
    # FALLBACK
    # ======================================================

    def _fallback_result(
        self,
        user_prompt: str
    ) -> dict:

        """
        Fallback if Ollama is unavailable.

        The final prompt remains unchanged.

        The original prompt is also used as a retrieval
        query to keep the pipeline operational.
        """

        query = self._validate_query_length(
            user_prompt
        )

        self.logger.warning(
            "Using fallback result"
        )

        return {
            "final_prompt": user_prompt,

            "pose_query": query,

            "style_query": query
        }

