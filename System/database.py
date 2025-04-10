import os
import sys
from contextlib import contextmanager
from typing import Generator, Optional, Dict, Any
from dotenv import load_dotenv
import logging
from cryptography.fernet import Fernet, InvalidToken

from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# Initialize environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('database.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure SQLAlchemy logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)

class DatabaseConfig:
    """Centralized database configuration"""
    def __init__(self):
        self.db_type = os.getenv('DB_TYPE', 'sqlite').lower()
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate required configuration"""
        if self.db_type in ('postgresql', 'mysql'):
            required_vars = ['DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_NAME']
            missing = [var for var in required_vars if not os.getenv(var)]
            if missing:
                logger.critical(f"Missing required DB config: {', '.join(missing)}")
                sys.exit(1)

    @property
    def uri(self) -> str:
        """Get database connection URI"""
        if self.db_type == 'postgresql':
            return (
                f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
                f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT', '5432')}"
                f"/{os.getenv('DB_NAME')}"
            )
        elif self.db_type == 'mysql':
            return (
                f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
                f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT', '3306')}"
                f"/{os.getenv('DB_NAME')}?charset=utf8mb4"
            )
        else:  # sqlite
            db_file = os.getenv('DB_FILE', 'riaor.db')
            return f"sqlite:///{db_file}"

    @property
    def connect_args(self) -> Dict[str, Any]:
        """Database-specific connection arguments"""
        return {'check_same_thread': False} if self.db_type == 'sqlite' else {}

    @property
    def pool_config(self) -> Dict[str, Any]:
        """Connection pool configuration"""
        return {
            'pool_size': int(os.getenv('DB_POOL_SIZE', 5)),
            'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', 10)),
            'pool_pre_ping': os.getenv('DB_POOL_PRE_PING', 'true').lower() == 'true',
            'pool_recycle': int(os.getenv('DB_POOL_RECYCLE', 3600)),
            'pool_timeout': int(os.getenv('DB_POOL_TIMEOUT', 30))
        }

class EncryptionService:
    """Handles all cryptographic operations"""
    def __init__(self):
        self.cipher = self._initialize_cipher()

    def _initialize_cipher(self) -> Fernet:
        """Initialize Fernet cipher with encryption key"""
        key = os.getenv('ENCRYPTION_KEY')
        if not key:
            key = Fernet.generate_key().decode()
            logger.warning("No ENCRYPTION_KEY in env - using generated key (not secure for production!)")
        return Fernet(key.encode())

    def encrypt(self, data: str) -> str:
        """Encrypt data using Fernet"""
        if not data:
            raise ValueError("No data provided for encryption")
        try:
            return self.cipher.encrypt(data.encode()).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            raise ValueError("Data encryption failed") from e

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data using Fernet"""
        if not encrypted_data:
            raise ValueError("No data provided for decryption")
        try:
            return self.cipher.decrypt(encrypted_data.encode()).decode()
        except InvalidToken:
            logger.error("Decryption failed - invalid token (possible key mismatch)")
            raise ValueError("Invalid encryption key") from None
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise ValueError("Data decryption failed") from e

    def rotate_key(self, new_key: str) -> bool:
        """Rotate encryption key"""
        try:
            self.cipher = Fernet(new_key.encode())
            logger.info("Encryption key rotated successfully")
            return True
        except Exception as e:
            logger.error(f"Key rotation failed: {str(e)}")
            return False


def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable foreign key constraints for SQLite"""
    if db_config.db_type == 'sqlite':
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

# Initialize services
db_config = DatabaseConfig()
encryption = EncryptionService()

# Create database engine
engine = create_engine(
    db_config.uri,
    echo=False,  # Set to True for debugging
    **db_config.pool_config,
    connect_args=db_config.connect_args,
)

event.listen(engine, "connect", set_sqlite_pragma)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
)

# Base for declarative models
Base = declarative_base()

@contextmanager
def db_session() -> Generator:
    """Provide a transactional scope around a series of operations"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
        logger.debug("Transaction committed successfully")
    except Exception as e:
        session.rollback()
        logger.error(f"Transaction failed: {str(e)}")
        raise
    finally:
        session.close()
        logger.debug("Session closed")

def init_db() -> bool:
    """Initialize database tables with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables initialized successfully")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Database initialization attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                logger.critical("Max retries reached for database initialization")
                raise RuntimeError("Database initialization failed") from e
            time.sleep(1)
    return False

def health_check() -> Dict[str, bool]:
    """Check database health status"""
    status = {
        'database_connection': False,
        'encryption_service': False,
        'schema_initialized': False
    }
    
    # Test database connection
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        status['database_connection'] = True
    except Exception as e:
        logger.error(f"Database connection check failed: {str(e)}")

    # Test encryption service
    try:
        test_str = "test_encryption"
        encrypted = encryption.encrypt(test_str)
        decrypted = encryption.decrypt(encrypted)
        status['encryption_service'] = decrypted == test_str
    except Exception as e:
        logger.error(f"Encryption service check failed: {str(e)}")

    try:
        with engine.connect() as conn:
            if engine.dialect.has_table(conn, "customers"):
                status['schema_initialized'] = True
    except Exception as e:
        logger.error(f"Schema check failed: {str(e)}")

    return status

def get_encryption_service() -> EncryptionService:
    """Get encryption service instance"""
    return encryption

def encrypt_data(data: str) -> str:
    """Encrypt data (convenience wrapper)"""
    return encryption.encrypt(data)

def decrypt_data(encrypted_data: str) -> str:
    """Decrypt data (convenience wrapper)"""
    return encryption.decrypt(encrypted_data)

def verify_db_connection() -> bool:
    """Verify database connection is working"""
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        logger.info("Starting database initialization...")
        if init_db():
            logger.info("Database setup completed successfully")
            health = health_check()
            logger.info(f"Health check results: {health}")
    except Exception as e:
        logger.critical(f"Critical database initialization error: {str(e)}")
        sys.exit(1)