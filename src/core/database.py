from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

# Bikin file database bernama cliste_ai.db di root folder
SQLALCHEMY_DATABASE_URL = "sqlite:///./cliste_ai.db"

# check_same_thread=False wajib untuk SQLite di FastAPI biar nggak error saat diakses banyak user
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# Fungsi Dependency Injection untuk mengambil sesi database
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
