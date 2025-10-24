from pydantic import BaseModel


class ResponseFormat(BaseModel):
    """Response schema for the agent."""
    punny_response: str
    weather_conditions: str | None = None
