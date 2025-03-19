import sys
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
import signal
import socket
from werkzeug.serving import make_server
import atexit
import threading

logger = logging.getLogger(__name__)

class ServerManager:
    def __init__(self):
        self.server = None
        self._setup_signal_handlers()
        self._shutdown_event = threading.Event()
        self._shutdown_timeout = 10  # seconds

    def _setup_signal_handlers(self):
        # Use just one approach for handling signals
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}")
        # Just set the event and call cleanup once
        if not self._shutdown_event.is_set():
            self._shutdown_event.set()
            # Use threading to avoid hanging in signal handler
            threading.Thread(target=self.cleanup).start()

    def cleanup(self, *args):
        if self.server:
            logger.info("Shutting down server...")
            try:
                # Set a timeout for the shutdown process
                shutdown_thread = threading.Thread(target=self._shutdown_server)
                shutdown_thread.daemon = True
                shutdown_thread.start()

                # Wait with timeout
                shutdown_thread.join(timeout=self._shutdown_timeout)

                if shutdown_thread.is_alive():
                    logger.warning(f"Server shutdown took longer than {self._shutdown_timeout}s. Forcing exit.")
                    # Force cleanup if still running after timeout
                    sys.exit(1)

                logger.info("Server shutdown complete")
            except Exception as e:
                logger.error(f"Error during server shutdown: {e}")
                sys.exit(1)

    def _shutdown_server(self):
        """Actual server shutdown logic, run in a separate thread"""
        try:
            if hasattr(self.server, 'shutdown'):
                # First stop accepting new requests
                self.server.shutdown()

            # Then properly close the server
            if hasattr(self.server, 'server_close'):
                self.server.server_close()
        except Exception as e:
            logger.error(f"Error during server shutdown sequence: {e}")

    def run_server(self, app, host, port, debug):
        try:
            if debug:
                # Register cleanup for debug mode
                atexit.register(self.cleanup)
                # Use werkzeug's debug server
                self.server = make_server(host, port, app, threaded=True)
                logger.info(f"Starting debug server on {host}:{port}")
                # Add check for shutdown event
                while not self._shutdown_event.is_set():
                    self.server.handle_request()
            else:
                # In production, use regular werkzeug server
                self.server = make_server(host, port, app)
                logger.info(f"Starting production server on {host}:{port}")
                self.server.serve_forever()
        except socket.error as e:
            logger.error(f"Socket error: {e}")
            raise
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            # Ensure cleanup is called
            self.cleanup()

def create_app(config):
    app = Flask(__name__)
    app.config.from_object(config)

    # Initialize database storage with configured retention periods
    db_storage = DatabaseStorage(
        connection_url=app.config['DATABASE_URL'],
        connect_args=app.config['DB_CONNECT_ARGS'],
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
    server_manager = ServerManager()
    try:
        server_manager.run_server(app, args.host, port, args.debug)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        server_manager.cleanup()
    except Exception as e:
        logger.error(f"Server error: {e}")
        server_manager.cleanup()
        raise

if __name__ == '__main__':
    main()
