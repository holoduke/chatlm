from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=("settings_",),
    )

    ollama_host: str = "http://localhost:11434"
    model_name: str = "gemma4:e2b"
    request_timeout: int = 120


settings = Settings()
