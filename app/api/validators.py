from functools import wraps
from flask import request, jsonify
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Logs to console
    ]
)

logger = logging.getLogger(__name__)


def require_headers(f):
    """Decorator to require necessary headers"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        required_headers = ['X-Hasura-Role']
        missing_headers = [h for h in required_headers if h not in request.headers]

        if missing_headers:
            logger.error(f"Missing required headers: {missing_headers}")
            return jsonify({
                "error": "Missing required headers",
                "missing": missing_headers
            }), 400

        return f(*args, **kwargs)
    return decorated_function


def validate_request(f):
    """Decorator to validate request body"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            request_data = request.get_json()

            if not request_data:
                logger.error("Empty request body")
                return jsonify({"error": "Request body is required"}), 400

            if 'rawRequest' not in request_data:
                logger.error("Missing rawRequest field")
                return jsonify({"error": "rawRequest is required"}), 400

            if 'response' not in request_data:  # Changed from 'data'
                logger.error("Missing response field")
                return jsonify({"error": "response is required"}), 400

            raw_request = request_data['rawRequest']
            required_raw_fields = ['query', 'variables', 'operationName']
            missing_raw_fields = [f for f in required_raw_fields if f not in raw_request]

            if missing_raw_fields:
                logger.error(f"Missing required fields in rawRequest: {missing_raw_fields}")
                return jsonify({
                    "error": "Invalid rawRequest format",
                    "missing": missing_raw_fields
                }), 400

            # Reject introspection queries
            if raw_request['operationName'] == "IntrospectionQuery":
                logger.error("Introspection queries are not allowed")
                return jsonify({
                    "error": "Introspection queries are not allowed"
                }), 400

            if not isinstance(request_data['response'], dict):  # Changed from 'data'
                logger.error("response field must be an object")
                return jsonify({
                    "error": "response must be an object with dataset keys"
                }), 400

            return f(*args, **kwargs)

        except Exception as e:
            logger.error(f"Request validation error: {str(e)}")
            return jsonify({"error": "Invalid request format"}), 400

    return decorated_function
