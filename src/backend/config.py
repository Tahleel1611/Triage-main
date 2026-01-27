import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    app_name: str = "AI Triage HMS"
    
    # Use SQLite for development if DATABASE_URL not set
    database_url: str = Field(
        default="sqlite:///./triage_dev.db",
        alias="DATABASE_URL",
    )
    jwt_secret_key: str = Field(default="change-this-secret", alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=60)

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
