from database import Base
from sqlalchemy import Column, Integer, String, Float
import datetime
class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    emotion = Column(String)
    score = Column(Float)
    create_at_date = Column(String)