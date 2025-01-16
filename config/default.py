import os
from pathlib import Path
import dotenv

dotenv.load_dotenv()

# Database configuration
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = int(os.getenv('DB_PORT', '5432'))
DB_NAME = os.getenv('DB_NAME', 'anomalies')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')

# Construct database URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"

# Determine base directory for the application
BASE_DIR = Path(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Data persistence configuration
STORAGE_PATH = Path(os.getenv('STORAGE_PATH', str(BASE_DIR / 'tmp')))

# Server configuration
PORT = int(os.getenv('PORT', '8787'))
HOST = os.getenv('HOST', '0.0.0.0')

# Anthropic configuration
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
CLAUDE_MODEL = os.getenv('CLAUDE_MODEL', 'claude-3-sonnet-20240229')
MAX_TOKENS = int(os.getenv('MAX_TOKENS', '100000'))
MAX_RECORDS_PER_BATCH = int(os.getenv('MAX_RECORDS_PER_BATCH', '50'))

# Anomaly detection configuration
HISTORICAL_RETENTION_DAYS = int(os.getenv('RETENTION_DAYS', '14'))
ANOMALY_RETENTION_DAYS = int(os.getenv('ANOMALY_RETENTION_DAYS', '90'))
MODEL_RETENTION_DAYS = int(os.getenv('ANOMALY_RETENTION_DAYS', '360'))
ANOMALY_THRESHOLD = float(os.getenv('ANOMALY_THRESHOLD', '0.1'))
MINIMUM_TRAINING_RECORDS = int(os.getenv('MINIMUM_TRAINING_RECORDS', '100'))
MAX_HISTORICAL_RECORDS = int(os.getenv('MAX_HISTORICAL_RECORDS', '100000'))

# Ensure data directory exists
STORAGE_PATH.mkdir(parents=True, exist_ok=True)
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
