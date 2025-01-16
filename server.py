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

    def _setup_signal_handlers(self):
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}")
        self._shutdown_event.set()
        self.cleanup()

    def cleanup(self, *args):
        if self.server:
            logger.info("Shutting down server...")
            try:
                # Close all active connections
                if hasattr(self.server, 'socket'):
                    self.server.socket.close()
                self.server.shutdown()
            except Exception as e:
                logger.error(f"Error during server shutdown: {e}")

            try:
                # Force socket cleanup
                for sock in socket.socket._defaultsock:
                    try:
                        sock.close()
                    except:
                        pass
            except:
                pass
            logger.info("Server shutdown complete")

    def run_server(self, app, host, port, debug):
        try:
            if debug:
                # Register cleanup for debug mode
                atexit.register(self.cleanup)
                # Use werkzeug's debug server
                self.server = make_server(host, port, app, threaded=True)
                self.server._BaseServer__is_shut_down = threading.Event()  # Reset shutdown event
                logger.info(f"Starting debug server on {host}:{port}")
                self.server.serve_forever()
            else:
                # In production, use regular werkzeug server
                self.server = make_server(host, port, app)
                logger.info(f"Starting production server on {host}:{port}")
                self.server.serve_forever()
        except socket.error as e:
            logger.error(f"Socket error: {e}")
            # Try to force-close the port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind((host, port))
            except:
                pass
            finally:
                sock.close()
            raise
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            self.cleanup()

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
