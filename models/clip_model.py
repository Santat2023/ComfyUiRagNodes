import torch
import clip


class ClipModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load("ViT-B/32", device=self.device)

    def embed_text(self, text: str):
        with torch.no_grad():
            tokens = clip.tokenize([text]).to(self.device)
            emb = self.model.encode_text(tokens)
            return emb.cpu().numpy()[0]