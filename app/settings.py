import os


MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/kws_resnet.pt")
DB_PATH = os.getenv("DB_PATH", "storage/requests.db")
