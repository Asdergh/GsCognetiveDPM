import torch 
import numpy as np
import tyro
import torch.nn.functional as F
from FlagEmbedding import FlagModel as flModel
from ..models.blocks import InteractionAttention



# txt_embedder = flModel(
#     model_name_or_path="BAAI/bge-base-en-v1.5",
#     normalize_embeddings=True,
#     devices="cpu",
#     batch_size=32,
# )
# sentence = [
#     "fidn the the cat on the image",
#     "theres the cat on the image, fidn it",
#     "serach the hole image and find me a cat",
#     "need to find cat on this image, can you help ?",
#     "find cat",
#     "cat"
# ]
# embedding = txt_embedder.encode(sentence, convert_to_numpy=False)
# print(embedding.size()
# print(embedding.min(), embedding.mean(), embedding.max())
# print(embedding @ embedding.T)


test = torch.rand((10, 256, 8, 8))
test = torch.rand((10, 64, 256))
# test1 = torch.rand((10, 256)).repeat(test.size(-2), test.size(-1), 1, 1)
# print(test1.size())
# test1 = test1.permute(-2, -1, 0, 1)
# print(test1.size())
# print(test.size(), test1.size())
# tokens = torch.cat([test, test1], dim=1)
# print(tokens.size())

attention = InteractionAttention(
    input_dim=256,
    first_dim=64,
    hiden_dim=32,
    last_dim=268,
    patch_n_pr=(8, 8),
    pooling_size=3,
    mode="self",
    format="sequence"
)
print(attention(test).size())

# test = torch.normal(0, 1, (10, 256, 268))
# print(attention(test).size())


