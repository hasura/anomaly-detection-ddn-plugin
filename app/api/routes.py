import datetime
import hashlib
import os
import re
from flask import request, jsonify
from app.api.validators import require_headers, validate_request
import logging

from app.core.storage import AnomalyStorage
from app.core.db_storage import DatabaseStorage

logger = logging.getLogger(__name__)


def generate_query_hash(query: str) -> str:
    """Generate MD5 hash of query string"""
    return hashlib.md5(query.encode('utf-8')).hexdigest()


def should_exclude_dataset(dataset_key: str, exclusion_patterns: list) -> bool:
    """
    Check if dataset should be excluded based on regex patterns

    Args:
        dataset_key: The dataset key to check
        exclusion_patterns: List of compiled regex patterns

    Returns:
        bool: True if dataset should be excluded, False otherwise
    """
    if not exclusion_patterns:
        return False

    for pattern in exclusion_patterns:
        if pattern.search(dataset_key):
            logger.info(f"Excluding dataset {dataset_key} based on pattern {pattern.pattern}")
            return True

    return False


def create_routes(app, anomaly_service):
    """Create API routes"""

    # Initialize storage handlers
    file_storage = AnomalyStorage(app.config['STORAGE_PATH'])
    db_storage = DatabaseStorage(app.config['DATABASE_URL'],app.config['DB_CONNECT_ARGS'])

    # Get and compile exclusion patterns from environment variable
    excluded_datasets_env = os.environ.get('EXCLUDED_DATASETS', '')
    excluded_patterns = []

    if excluded_datasets_env:
        patterns = [p.strip() for p in excluded_datasets_env.split(',') if p.strip()]
        for pattern in patterns:
            try:
                excluded_patterns.append(re.compile(pattern))
                logger.info(f"Added dataset exclusion pattern: {pattern}")
            except re.error as e:
                logger.error(f"Invalid regex pattern '{pattern}': {str(e)}")

    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "excluded_patterns": [p.pattern for p in excluded_patterns] if excluded_patterns else []
        })

    @app.route('/anomalies', methods=['POST'])
    @require_headers
    @validate_request
    def detect_anomalies():
        """Main anomaly detection endpoint"""
        timestamp = datetime.datetime.now().isoformat()

        try:
            request_data = request.get_json()
            raw_request = request_data['rawRequest']
            datasets = request_data['response']['data']

            # Generate query hash as primary key
            query_hash = generate_query_hash(raw_request['query'])

            context = {
                'query': raw_request['query'],
                'query_hash': query_hash,
                'variables': raw_request['variables'],
                'operation_name': raw_request['operationName'],
                'user': request.headers.get('X-Hasura-User'),
                'role': request.headers['X-Hasura-Role'],
                'timestamp': timestamp,
                'rawRequest': raw_request
            }

            results = {}
            skipped_datasets = []
            total_records = 0
            total_anomalies = 0

            for dataset_key, records in datasets.items():
                if not records:  # Skip empty datasets
                    continue

                # Check if dataset should be excluded
                if should_exclude_dataset(dataset_key, excluded_patterns):
                    skipped_datasets.append(dataset_key)
                    logger.info(f"Skipping excluded dataset: {dataset_key}")
                    continue

                total_records += len(records)
                storage_key = f"{query_hash}_{context['operation_name']}_{dataset_key}"

                try:
                    analysis_result = anomaly_service.analyze_dataset(
                        query_id=query_hash,
                        dataset=records,
                        context=context
                    )

                    results[dataset_key] = analysis_result
                    total_anomalies += analysis_result.get('statistical_flags', 0)

                    # Store analysis result in file using storage service
                    try:
                        file_path = file_storage.store_anomaly(
                            query_id=query_hash,
                            storage_key=storage_key,
                            query_result={'data': records},
                            context=context,
                            analysis_result=analysis_result
                        )
                        if file_path:
                            results[dataset_key]['file_path'] = file_path
                            logger.info(f"Stored analysis result to file: {file_path}")
                    except Exception as file_error:
                        logger.error(f"Error storing analysis to file: {str(file_error)}")

                    # Store in database
                    try:
                        db_record_id = db_storage.store_anomaly(
                            query_id=query_hash,
                            query_result={'data': records},
                            context=context,
                            analysis_result=analysis_result
                        )
                        if db_record_id:
                            results[dataset_key]['db_record_id'] = db_record_id
                            logger.info(f"Stored analysis result in database with ID: {db_record_id}")
                    except Exception as db_error:
                        logger.error(f"Error storing analysis in database: {str(db_error)}")

                except Exception as e:
                    logger.error(f"Error analyzing dataset {dataset_key}: {str(e)}")
                    results[dataset_key] = {
                        "error": str(e),
                        "timestamp": timestamp,
                        "status": "error",
                        "dataset": dataset_key
                    }

            # Only save query pattern if any datasets were analyzed
            if results:
                db_storage.save_query_pattern(
                    query_text=context.get('query'),
                    variables_json=context.get('variables'),
                    user_role=context.get('role'),
                    is_anomalous=total_anomalies > 0,
                )

            response = {
                "results": results,
                "metadata": {
                    "timestamp": timestamp,
                    "query_hash": query_hash,
                    "operation_name": context['operation_name'],
                    "total_records": total_records,
                    "total_anomalies": total_anomalies,
                    "datasets_processed": len(results),
                    "datasets_skipped": skipped_datasets,
                    "status": "completed"
                }
            }

            return jsonify(response)

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            return jsonify({
                "error": "Internal server error",
                "details": str(e),
                "timestamp": timestamp,
                "status": "error"
            }), 500

    @app.route('/history/<query_id>', methods=['GET'])
    def get_history(query_id):
        """Get historical data for a query"""
        try:
            historical_data = anomaly_service.get_historical_data(query_id)
            return jsonify({
                "query_hash": query_id,
                "record_count": len(historical_data),
                "data": historical_data,
                "timestamp": datetime.datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error retrieving history for {query_id}: {str(e)}")
            return jsonify({
                "error": f"Error retrieving history: {str(e)}",
                "timestamp": datetime.datetime.now().isoformat(),
                "status": "error"
            }), 500

    @app.route('/history/<query_id>', methods=['DELETE'])
    def clear_history(query_id):
        """Clear historical data for a query"""
        try:
            success = anomaly_service.clear_history(query_id)
            return jsonify({
                "success": success,
                "query_hash": query_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "message": "History cleared successfully" if success else "Error clearing history"
            })
        except Exception as e:
            logger.error(f"Error clearing history for {query_id}: {str(e)}")
            return jsonify({
                "error": f"Error clearing history: {str(e)}",
                "timestamp": datetime.datetime.now().isoformat(),
                "status": "error"
            }), 500

    @app.route('/model/<query_id>', methods=['GET'])
    def get_model_info(query_id):
        """Get model information for a query"""
        try:
            model_info = anomaly_service.get_model_info(query_id)
            return jsonify({
                "query_hash": query_id,
                "model_info": model_info,
                "timestamp": datetime.datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error retrieving model info for {query_id}: {str(e)}")
            return jsonify({
                "error": f"Error retrieving model info: {str(e)}",
                "timestamp": datetime.datetime.now().isoformat(),
                "status": "error"
            }), 500

    @app.route('/analyze/<query_id>', methods=['POST'])
    def analyze_single(query_id):
        """Analyze a single record"""
        timestamp = datetime.datetime.now().isoformat()

        try:
            record = request.get_json()
            analysis_result = anomaly_service.analyze_dataset(
                query_id=query_id,
                dataset=[record],
                context={"type": "single_record", "timestamp": timestamp}
            )

            # Store analysis result in file using storage service
            try:
                file_path = file_storage.store_anomaly(
                    query_id=query_id,
                    storage_key=f"{query_id}_single",
                    query_result={'data': [record]},
                    context={"type": "single_record", "timestamp": timestamp},
                    analysis_result=analysis_result
                )
                if file_path:
                    analysis_result['file_path'] = file_path
                    logger.info(f"Stored analysis result to file: {file_path}")
            except Exception as file_error:
                logger.error(f"Error storing analysis to file: {str(file_error)}")

            # Store in database
            try:
                db_record_id = db_storage.store_anomaly(
                    query_id=query_id,
                    query_result={'data': [record]},
                    context={"type": "single_record", "timestamp": timestamp},
                    analysis_result=analysis_result
                )
                if db_record_id:
                    analysis_result['db_record_id'] = db_record_id
                    logger.info(f"Stored analysis result in database with ID: {db_record_id}")
            except Exception as db_error:
                logger.error(f"Error storing analysis in database: {str(db_error)}")

            return jsonify({
                "query_hash": query_id,
                "analysis": analysis_result,
                "timestamp": timestamp
            })
        except Exception as e:
            logger.error(f"Error analyzing record for {query_id}: {str(e)}")
            return jsonify({
                "error": f"Error analyzing record: {str(e)}",
                "timestamp": timestamp,
                "status": "error"
            }), 500

    @app.route('/stats/<query_id>', methods=['GET'])
    def get_statistics(query_id):
        """Get statistical summaries for a query"""
        try:
            historical_data = anomaly_service.get_historical_data(query_id)
            model_info = anomaly_service.get_model_info(query_id)

            return jsonify({
                "query_hash": query_id,
                "historical_records": len(historical_data),
                "model_info": model_info,
                "timestamp": datetime.datetime.now().isoformat(),
                "status": "active"
            })
        except Exception as e:
            logger.error(f"Error retrieving statistics for {query_id}: {str(e)}")
            return jsonify({
                "error": f"Error retrieving statistics: {str(e)}",
                "timestamp": datetime.datetime.now().isoformat(),
                "status": "error"
            }), 500

    return app
