from typing import Literal, TypeAlias


EmbeddingModel: TypeAlias = str | Literal["hashed_bag_of_words"]
AnsSimilarityFunc: TypeAlias = str | Literal["jaccard", "hamming"]
