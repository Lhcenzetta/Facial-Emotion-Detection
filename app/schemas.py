from pydantic import BaseModel


class PredictionBase(BaseModel):
    emotion : str
    score : float
    create_at_date : str
class PredictionCreate(PredictionBase):
    pass

class Prediction(PredictionBase):
    id : int

    class config:
        orm_model = True
