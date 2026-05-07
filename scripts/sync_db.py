from src.core.database import engine, Base
from src.models import leads, chat_history, ingest_log

print("Syncing database tables...")
Base.metadata.create_all(bind=engine)
print("Done!")
