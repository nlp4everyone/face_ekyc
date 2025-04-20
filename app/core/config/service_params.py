from dotenv import load_dotenv
import os

# Load from .env file
load_dotenv()

# Params
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_PORT = os.getenv("MINIO_PORT")