from functools import lru_cache
from src.inference.predictor import Predictor

@lru_cache(maxsize=1)
def get_predictor():
    return Predictor()

