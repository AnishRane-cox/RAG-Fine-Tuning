from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Mongo Atlas connections
    MONGO_URI: str = "mongodb+srv://anishrane2000_db_user:WROBJRsitZ4Dwmsd@cluster0.hxka3jx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    MONGO_DB: str = "rag"
    MONGO_COLLECTION: str = "documents"

    # Embedding model (sentence-transformers by default)
    MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Batch sizes / chunk sizes (configurable)
    BATCH_SIZE: int = 32

    # If set True, attempts to use FAISS local fallback (not implemented fully here)
    USE_FAISS_FALLBACK: bool = False

    class Config:
        env_file = ".env"
        extra = "ignore"

def get_settings() -> Settings:
    return Settings()




