from datetime import datetime

from flask import Flask
from app.api.routes import create_routes
from app.utils.logging import setup_logging
from app.core.hybrid_detector import EfficientHybridDetector
from app.core.db_storage import DatabaseStorage
import config.default as config
import argparse
import os
import logging

logger = logging.getLogger(__name__)


def create_app(config):
    app = Flask(__name__)
    app.config.from_object(config)

    # Initialize database storage with configured retention periods
    db_storage = DatabaseStorage(
        connection_url=app.config['DATABASE_URL'],
        historical_retention_days=app.config.get('HISTORICAL_RETENTION_DAYS', 14),
        anomaly_retention_days=app.config.get('ANOMALY_RETENTION_DAYS', 30)
    )

    # Initialize services
    anomaly_service = EfficientHybridDetector(
        db_storage=db_storage,
        anthropic_api_key=app.config['ANTHROPIC_API_KEY']
    )

    # Create routes with services
    app = create_routes(app, anomaly_service)

    # Add cleanup endpoint or scheduled task
    @app.cli.command("cleanup-data")
    def cleanup_data():
        """Command to clean up old data"""
        try:
            result = db_storage.cleanup_all_old_data()
            print(f"Cleanup completed: {result}")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    return app


def main():
    parser = argparse.ArgumentParser(description='Anomaly Detection Service')
    parser.add_argument('--host',
                        default=config.HOST,
                        help='Host to bind to')
    parser.add_argument('--port',
                        type=int,
                        default=config.PORT,
                        help='Port to bind to (default: 8787)')
    parser.add_argument('--debug',
                        action='store_true',
                        help='Enable debug mode')
    args = parser.parse_args()

    port = int(os.getenv('PORT', args.port))

    app = create_app(config)
    logger.info(f"Starting server on {args.host}:{port}")
    app.run(host=args.host, port=port, debug=args.debug)


if __name__ == '__main__':
    main()
