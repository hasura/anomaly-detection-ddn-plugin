import logging
from datetime import datetime
from typing import Dict, List

import random
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app.core.db_storage import DatabaseStorage
from config.default import HISTORICAL_RETENTION_DAYS, MAX_HISTORICAL_RECORDS

# Create logger for this module
logger = logging.getLogger(__name__)


class PersistentAnomalyDetector:
    def __init__(self,
                 db_storage: DatabaseStorage,
                 retrain_threshold: int = HISTORICAL_RETENTION_DAYS):
        """
        Initialize the statistical detector with database persistence

        Args:
            db_storage: DatabaseStorage instance for persistence
            retrain_threshold: Number of new records before triggering a retrain
        """
        self.db_storage = db_storage
        self.retrain_threshold = retrain_threshold
        self.model_cache = {}
        self.pending_updates = {}

        logger.debug("[statistical_detector.py] Initialized with historical_retention_days: %d, "
                     "retrain_threshold: %d",
                     db_storage.historical_retention_days, retrain_threshold)

    @staticmethod
    def _extract_features(record: Dict) -> List[float]:
        """Extract numerical features from record"""
        try:
            logger.debug("[statistical_detector.py] Extracting features from record type: %s", type(record))
            logger.debug("[statistical_detector.py] Record content: %s", record)

            # Handle nested data structure
            actual_data = record.get('data', record)
            logger.debug("[statistical_detector.py] Actual data for processing: %s", actual_data)

            if not isinstance(actual_data, dict):
                logger.error("[statistical_detector.py] Data is not a dictionary: %s", type(actual_data))
                return []

            features = []

            # Process dates into numerical features
            for key, value in actual_data.items():
                logger.debug("[statistical_detector.py] Processing field %s of type %s with value %s",
                             key, type(value), value)

                if isinstance(value, (int, float)):
                    features.append(float(value))
                elif isinstance(value, str):
                    # Try to parse dates
                    if 'date' in key.lower():
                        try:
                            dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                            # Convert to timestamp for numerical analysis
                            features.append(float(dt.timestamp()))
                            logger.debug("[statistical_detector.py] Converted date %s to timestamp %f",
                                         value, dt.timestamp())
                        except ValueError:
                            logger.debug("[statistical_detector.py] Could not parse date: %s", value)
                    # Try to parse numeric strings
                    elif value.replace('.', '').isdigit():
                        features.append(float(value))

            logger.debug("[statistical_detector.py] Extracted features: %s", features)
            return features

        except Exception as e:
            logger.error("[statistical_detector.py] Feature extraction error: %s", str(e), exc_info=True)
            return []

    def _train_model(self, query_id: str, historical_data: List[Dict]) -> Dict:
        logger.debug("[statistical_detector.py] Training new model for %s with %d records",
                     query_id, len(historical_data))

        # Extract features from all records
        features_list = []
        for record in historical_data:
            features = self._extract_features(record)
            if features:
                features_list.append(features)

        if not features_list:
            logger.error("[statistical_detector.py] No valid features extracted for training")
            raise ValueError("No valid features for training")

        try:
            # Check if the number of historical records exceeds the limit
            if len(historical_data) > MAX_HISTORICAL_RECORDS:
                logger.info("[statistical_detector.py] Number of historical records (%d) exceeds the limit (%d), sampling records.",
                            len(historical_data), MAX_HISTORICAL_RECORDS)
                # Calculate the number of records to remove
                overage = len(historical_data) - MAX_HISTORICAL_RECORDS
                # Sample the records to remove
                records_to_remove = random.choices(historical_data, k=overage)
                # Remove the sampled records from the historical data
                historical_data = [record for record in historical_data if record not in records_to_remove]
                # Recalculate the features_list
                features_list = []
                for record in historical_data:
                    features = self._extract_features(record)
                    if features:
                        features_list.append(features)

                # Remove the sampled records from the database
                self._remove_historical_data(query_id, records_to_remove)

            # Create and fit scaler
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_list)

            # Train isolation forest
            model = IsolationForest(random_state=42, contamination=0.1)
            model.fit(features_scaled)

            # Create model data dictionary
            model_data = {
                'model': model,
                'scaler': scaler,
                'timestamp': datetime.now().isoformat(),
                'num_records': len(features_list),
                'num_features': len(features_list[0]) if features_list else 0
            }

            self.db_storage.save_model(query_id, model_data)
            logger.debug("[statistical_detector.py] Saved new model to db for %s", query_id)

            # Update cache
            self.model_cache[query_id] = model_data
            return model_data

        except Exception as e:
            logger.error("[statistical_detector.py] Error training model for %s: %s", query_id, str(e))
            raise

    def _remove_historical_data(self, query_id: str, records_to_remove: List[Dict]):
        """Remove specified records from the historical data in the database"""
        try:
            self.db_storage.remove_historical_data(query_id, records_to_remove)
            logger.debug("[statistical_detector.py] Removed %d records for query_hash %s",
                         len(records_to_remove), query_id)
        except Exception as e:
            logger.error("[statistical_detector.py] Error removing historical data for %s: %s",
                         query_id, str(e))
            raise

    def _should_retrain(self, query_id: str) -> bool:
        """Check if model should be retrained based on pending updates"""
        pending = self.pending_updates.get(query_id, 0)
        return pending >= self.retrain_threshold

    def finalize_training(self, query_id: str = None):
        """
        Perform final training step for any pending records.
        If query_id is provided, only train that specific query's model.
        If query_id is None, train all models with pending updates.
        """
        try:
            queries_to_train = [query_id] if query_id else list(self.pending_updates.keys())

            for qid in queries_to_train:
                pending_count = self.pending_updates.get(qid, 0)
                if pending_count > 0:
                    logger.info("[statistical_detector.py] Final training step for %s with %d pending records",
                                qid, pending_count)

                    # Load historical data
                    historical_data = self._load_historical_data(qid)

                    # Train model with all data
                    try:
                        self._train_model(qid, historical_data)
                        self._reset_pending_updates(qid)
                        logger.info("[statistical_detector.py] Successfully completed final training for %s", qid)
                    except Exception as e:
                        logger.error("[statistical_detector.py] Error during final training for %s: %s",
                                     qid, str(e))
                        raise
        except Exception as e:
            logger.error("[statistical_detector.py] Error during finalize_training: %s", str(e))
            raise

    def check_anomaly(self, query_id: str, record: Dict, is_final_record: bool = False) -> Dict:
        logger.debug("[statistical_detector.py] Starting check_anomaly for query_id: %s", query_id)
        logger.debug("[statistical_detector.py] Input record: %s", record)

        # Load historical data
        historical_data = self._load_historical_data(query_id)
        logger.debug("[statistical_detector.py] Loaded historical data count: %d", len(historical_data))

        # Add timestamp to record
        timestamp = datetime.now().isoformat()
        logger.debug("[statistical_detector.py] Created timestamp: %s", timestamp)

        # Ensure we're storing the full record structure
        record_with_meta = {
            'data': record.get('data', record),
            'timestamp': record.get('timestamp', timestamp)
        }
        logger.debug("[statistical_detector.py] Created record_with_meta: %s", record_with_meta)

        # Add to historical data and save
        historical_data.append(record_with_meta)
        self._save_historical_data(query_id, historical_data)

        # Extract features
        features = self._extract_features(record_with_meta)
        logger.debug("[statistical_detector.py] Extracted features: %s", features)

        if not features:
            logger.warning("[statistical_detector.py] No features extracted from record")
            return {
                "is_anomaly": False,
                "reason": "No numerical features extracted",
                "score": 0.0,
                "features": []
            }

        # Increment pending updates counter
        self._increment_pending_updates(query_id)

        # Check if we should retrain (either threshold reached or final record)
        should_retrain = self._should_retrain(query_id) or is_final_record
        if should_retrain:
            logger.debug("[statistical_detector.py] Retraining triggered for %s (threshold: %d, is_final: %s)",
                         query_id, self.retrain_threshold, is_final_record)
            try:
                model_data = self._train_model(query_id, historical_data)
                self._reset_pending_updates(query_id)
            except Exception as e:
                logger.error("[statistical_detector.py] Error retraining model: %s", str(e))
                # Continue with existing model if retraining fails
                model_data = self.db_storage.load_model(query_id)
        else:
            # Check if the model is cached
            if query_id in self.model_cache:
                logger.debug("[statistical_detector.py] Using cached model for %s", query_id)
                model_data = self.model_cache[query_id]
            else:
                # Load existing model
                model_data = self.db_storage.load_model(query_id)
                if not model_data:
                    logger.debug("[statistical_detector.py] No existing model found, training initial model")
                    try:
                        model_data = self._train_model(query_id, historical_data)
                    except Exception as e:
                        logger.error("[statistical_detector.py] Error training initial model: %s", str(e))
                        raise

        try:
            # Scale features
            features_scaled = model_data['scaler'].transform([features])
            logger.debug("[statistical_detector.py] Scaled features shape: %s", features_scaled.shape)

            # Get predictions
            is_anomaly = model_data['model'].predict(features_scaled)[0] == -1
            anomaly_score = model_data['model'].score_samples(features_scaled)[0]

            result = {
                "is_anomaly": is_anomaly,
                "score": float(anomaly_score),
                "features": features,
                "model_details": {
                    "model_type": "isolation_forest",
                    "features_used": len(features),
                    "pending_updates": self.pending_updates.get(query_id, 0),
                    "retrain_threshold": self.retrain_threshold
                }
            }

            logger.debug("[statistical_detector.py] Analysis result: %s", result)
            return result

        except Exception as e:
            logger.error("[statistical_detector.py] Error during prediction: %s", str(e))
            raise

    def _load_historical_data(self, query_id: str) -> List[Dict]:
        """Load historical data from database"""
        try:
            return self.db_storage.load_historical_data(query_id)
        except Exception as e:
            logger.error("[statistical_detector.py] Error loading historical data for %s: %s",
                         query_id, str(e))
            return []

    def _save_historical_data(self, query_id: str, data: List[Dict]):
        """Save historical data to database"""
        try:
            self.db_storage.save_historical_data(query_id, data)
            logger.debug("[statistical_detector.py] Saved %d records for query_hash %s",
                         len(data), query_id)
        except Exception as e:
            logger.error("[statistical_detector.py] Error saving historical data for %s: %s",
                         query_id, str(e))
            raise

    def clear_history(self, query_id: str = None) -> bool:
        """Clear historical data using database storage"""
        try:
            success = self.db_storage.clear_historical_data(query_id)
            if success:
                logger.info("[statistical_detector.py] Cleared history for query_hash %s",
                            query_id if query_id else "all queries")
            return success
        except Exception as e:
            logger.error("[statistical_detector.py] Error clearing history: %s", str(e))
            return False

    def _increment_pending_updates(self, query_id: str):
        """Increment the pending updates counter for a query"""
        self.pending_updates[query_id] = self.pending_updates.get(query_id, 0) + 1

    def _reset_pending_updates(self, query_id: str):
        """Reset the pending updates counter for a query"""
        self.pending_updates[query_id] = 0

