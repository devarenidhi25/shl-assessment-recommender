from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=0)


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1)

    @field_validator("messages")
    @classmethod
    def last_message_must_be_user(cls, v: List[Message]) -> List[Message]:
        if not v:
            raise ValueError("messages must not be empty")
        return v


class RecommendationItem(BaseModel):
    name: str
    url: str
    test_type: str = ""


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[RecommendationItem] = Field(default_factory=list)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"
