from pydantic import BaseModel


class PredictionResponse(BaseModel):

    crop: str

    disease: str

    confidence: float

    message: str