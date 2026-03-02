from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # API Settings
    PROJECT_NAME: str = "AI Colorizer API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # CORS
    BACKEND_CORS_ORIGINS: list[str] | str = ["*"]
    
    # Upload Settings
    MAX_FILE_SIZE_MB: int = 10
    MAX_IMAGE_DIMENSION: int = 1024
    ALLOWED_MIME_TYPES: list[str] = ["image/jpeg", "image/png", "image/webp"]
    
    # Model Settings
    MODEL_WEIGHTS_PATH: str = "checkpoints/colorizer_resnet18_v3.pth"
    USE_HALF_PRECISION: bool = False
    
    # Admin Settings
    ADMIN_API_KEY: str = "hackathon-secret-key-123"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

settings = Settings()
