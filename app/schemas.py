from pydantic import BaseModel
from typing import Optional, Dict

class StudentProfile(BaseModel):
    user_id: str
    name: Optional[str]
    gpa: Optional[float]
    interests: Optional[Dict]
