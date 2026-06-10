from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CREDIT_", case_sensitive=False)

    APP_NAME: str = "Credit Risk Scoring API"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = ""  # asyncpg URL, empty means no DB
    DB_ECHO: bool = False

    # Model
    MODEL_PATH: str = "models/best_model.pkl"
    PREPROCESSOR_PATH: str = "models/preprocessor.pkl"

    # API
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list[str] = ["*"]
    RATE_LIMIT: str = "100/minute"

    # Security
    API_KEY_ENABLED: bool = False
    API_KEY: str = ""


settings = Settings()
