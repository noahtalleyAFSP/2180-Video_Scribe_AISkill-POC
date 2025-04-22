from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr, model_validator, BaseModel, Field, HttpUrl, field_validator
from typing import Optional, List


class GPTVision(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AZURE_OPENAI_GPT_VISION_",
        extra="ignore",
    )

    endpoint: str
    api_key: SecretStr
    api_version: str
    deployment: str

    @model_validator(mode="before")
    def check_missing_fields(cls, values):
        missing_fields = [field for field in cls.model_fields if field not in values]
        if missing_fields:
            missing_with_prefix = [
                f"{cls.model_config['env_prefix']}{field.upper()}"
                for field in missing_fields
            ]
            raise ValueError(
                f"Missing environment variables: {', '.join(missing_with_prefix)}. Please set these environment variables before proceeding."
            )
        return values


class AzureSpeech(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AZURE_SPEECH_",
        extra="ignore",
    )

    key: SecretStr
    region: str

    @model_validator(mode="before")
    def check_missing_fields(cls, values):
        missing_fields = [field for field in cls.model_fields if field not in values]
        if missing_fields:
            missing_with_prefix = [
                f"{cls.model_config['env_prefix']}{field.upper()}"
                for field in missing_fields
            ]
            raise ValueError(
                f"Missing environment variables: {', '.join(missing_with_prefix)}. Please set these environment variables before proceeding."
            )
        return values


class AzureFace(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FACE_",
        extra="ignore",
    )

    endpoint: str
    apikey: SecretStr

    @model_validator(mode="before")
    def check_missing_fields(cls, values):
        missing_fields = [field for field in cls.model_fields if field not in values]
        if missing_fields:
            missing_with_prefix = [
                f"{cls.model_config['env_prefix']}{field.upper()}"
                for field in missing_fields
            ]
            raise ValueError(
                f"Missing environment variables: {', '.join(missing_with_prefix)}. Please set these environment variables before proceeding."
            )
        return values


class BlobStorageConfig(BaseSettings):
    """Azure Blob Storage configuration."""
    model_config = SettingsConfigDict(
        extra="ignore",
    )

    account_name: Optional[str] = Field(
        default=None, validation_alias="AZURE_STORAGE_ACCOUNT_NAME"
    )
    container_name: Optional[str] = Field(
        default=None, validation_alias="AZURE_STORAGE_CONTAINER_NAME"
    )
    # Use one of the following for authentication:
    connection_string: Optional[SecretStr] = Field(
        default=None, validation_alias="AZURE_STORAGE_CONNECTION_STRING"
    )
    sas_token: Optional[SecretStr] = Field(
        default=None, validation_alias="AZURE_STORAGE_SAS_TOKEN" # Container or Account SAS
    )

    @field_validator("account_name", "container_name")
    def check_required_for_batch(cls, v, info):
        # These might become mandatory if batch transcription is used
        # For now, allow None but the transcription function will check
        return v

    @field_validator("sas_token")
    def check_auth(cls, v, info):
        if v is None and info.data.get("connection_string") is None:
            # We need at least one auth method if account_name/container are provided
            if info.data.get("account_name") and info.data.get("container_name"):
                 print("Warning: AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_SAS_TOKEN should be set for Blob Storage access.")
        return v


class CobraEnvironment(BaseModel):
    """Environment variables for Cobra services."""
    vision: GPTVision = Field(default_factory=GPTVision)
    speech: AzureSpeech = Field(default_factory=AzureSpeech)
    face: AzureFace = Field(default_factory=AzureFace)
    blob_storage: BlobStorageConfig = Field(default_factory=BlobStorageConfig)

    class Config:
        # Enable reading from environment variables automatically
        # Note: Pydantic v2 uses settings_config, v1 used this Config class
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Support nested environment variables if needed (adjust based on actual env var names)
        # env_nested_delimiter = '__' # Example if using VISION__API_KEY etc.
        extra = "ignore" # Ignore extra fields not defined in the model
        # Optional: Allow reading directly from environment
        # Requires Pydantic v2's PydanticSettings, or manual loading in older versions
