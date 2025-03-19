from typing import Dict, List
import logging
from datetime import datetime
from app.core.query_detector import QueryAnomalyDetector
from app.core.statistical_detector import PersistentAnomalyDetector
from app.core.db_storage import DatabaseStorage
from config.default import HISTORICAL_RETENTION_DAYS

# Create logger for this module
logger = logging.getLogger(__name__)


class EfficientHybridDetector:
    def __init__(self,
                 db_storage: DatabaseStorage,
                 anthropic_api_key: str,
                 max_historical_days: int = HISTORICAL_RETENTION_DAYS):
        """Initialize hybrid detector with database storage

        Args:
            db_storage: DatabaseStorage instance for persistence
            anthropic_api_key: API key for Anthropic services
            max_historical_days: Number of days to retain historical data
        """
        self.statistical_detector = PersistentAnomalyDetector(
            db_storage=db_storage,
            retrain_threshold=max_historical_days
        )
        self.query_detector = QueryAnomalyDetector(
            api_key=anthropic_api_key
        )
        logger.debug("[hybrid_detector.py] Initialized with max_historical_days: %d",
                     max_historical_days)

    def analyze_dataset(self,
                        query_id: str,  # This is now the query hash
                        dataset: List[Dict],
                        context: Dict) -> Dict:
        """Analyze dataset focusing on query patterns"""
        logger.debug("[hybrid_detector.py] Starting analysis for query_hash: %s with dataset size: %d",
                     query_id, len(dataset))
        logger.debug("[hybrid_detector.py] Context: %s", context)

        # Get historical query results using query hash
        historical_data = self.statistical_detector._load_historical_data(query_id)
        logger.debug("[hybrid_detector.py] Loaded historical data count: %d", len(historical_data))

        # First check for basic statistical anomalies
        statistical_flags = []
        errors = []

        for idx, record in enumerate(dataset):
            try:
                is_final = idx == len(dataset) - 1  # True for last record
                logger.debug("[hybrid_detector.py] Processing record %d: %s", idx, record)

                # Handle both nested and flat data structures
                data_record = record.get('data', record)
                if not isinstance(data_record, dict):
                    logger.error("[hybrid_detector.py] Record %d does not contain valid data: %s",
                                 idx, type(data_record))
                    continue


                result = self.statistical_detector.check_anomaly(query_id, record, historical_data, is_final_record=is_final)
                logger.debug("[hybrid_detector.py] Statistical analysis result for record %d: %s",
                             idx, result)

                if result and result.get("is_anomaly"):
                    statistical_flags.append({
                        "record": data_record,
                        "score": result.get("score", 0),
                        "index": idx,
                        "features": result.get("features", []),
                        "timestamp": record.get("timestamp", datetime.now().isoformat())
                    })

            except Exception as e:
                error_msg = f"Error processing record {idx}: {str(e)}"
                logger.error("[hybrid_detector.py] %s", error_msg, exc_info=True)
                errors.append(error_msg)

        logger.debug("[hybrid_detector.py] Found %d statistical anomalies", len(statistical_flags))

        try:
            # Analyze query patterns with focus on the entire result set
            logger.debug("[hybrid_detector.py] Starting query pattern analysis")
            query_analysis = self.query_detector.analyze_query_results(
                historical_data=historical_data,
                new_data=dataset,
                context=context
            )
            logger.debug("[hybrid_detector.py] Query analysis complete: %s", query_analysis)

            result = {
                "query_analysis": query_analysis,
                "query_hash": query_id,  # Include query hash in result
                "operation_name": context.get("operation_name"),  # Include operation name
                "statistical_flags": len(statistical_flags),
                "statistical_details": statistical_flags,
                "processed_records": len(dataset),
                "anomalous_records": len(statistical_flags),
                "recommendations": query_analysis.get("query_pattern_analysis", {}).get("recommendations", []),
                "security_concerns": [
                    record for record in query_analysis.get("records", [])
                    if record.get("category") == "security_concern"
                ],
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "query_hash": query_id,
                    "operation_name": context.get("operation_name"),
                    "historical_data_size": len(historical_data),
                    "errors": errors if errors else None
                }
            }

            logger.debug("[hybrid_detector.py] Analysis complete: %s", result)
            return result

        except Exception as e:
            logger.error("[hybrid_detector.py] Error in query pattern analysis: %s", str(e), exc_info=True)
            # Return partial results if we have them
            return {
                "error": str(e),
                "error_details": errors,
                "statistical_flags": len(statistical_flags),
                "statistical_details": statistical_flags,
                "processed_records": len(dataset),
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "query_hash": query_id,
                    "operation_name": context.get("operation_name"),
                    "historical_data_size": len(historical_data),
                    "error": str(e)
                }
            }

    def get_historical_data(self, query_id: str) -> List[Dict]:
        """Get historical data for a query hash"""
        try:
            historical_data = self.statistical_detector._load_historical_data(query_id)
            logger.debug("[hybrid_detector.py] Retrieved %d historical records for query_hash %s",
                         len(historical_data), query_id)
            return historical_data
        except Exception as e:
            logger.error("[hybrid_detector.py] Error retrieving historical data: %s", str(e))
            return []

    def clear_history(self, query_id: str = None) -> bool:
        """Clear historical data for a query hash"""
        try:
            if query_id:
                # Clear specific query history
                self.statistical_detector._save_historical_data(query_id, [])
                logger.info("[hybrid_detector.py] Cleared history for query_hash %s", query_id)
            else:
                # Clear all histories (implement based on your storage structure)
                logger.warning("[hybrid_detector.py] Clearing all historical data")
                # Implementation depends on your storage structure
            return True
        except Exception as e:
            logger.error("[hybrid_detector.py] Error clearing history: %s", str(e))
            return False

    def get_model_info(self, query_id: str) -> Dict:
        """Get information about the statistical model for a query hash"""
        try:
            model_data = self.statistical_detector.db_storage.load_model(query_id)
            if not model_data:
                return {"status": "no_model"}

            return {
                "status": "active",
                "model_type": "isolation_forest",
                "query_hash": query_id,
                "features": len(model_data['scaler'].mean_) if model_data.get('scaler') else 0,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error("[hybrid_detector.py] Error getting model info: %s", str(e))
            return {"status": "error", "error": str(e)}
