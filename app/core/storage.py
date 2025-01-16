from pathlib import Path
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional
import numpy as np


class AnomalyStorage:
    def __init__(self, base_directory: str):
        """Initialize storage manager with required directories"""
        self.base_dir = Path(base_directory)
        self.anomalies_dir = self.base_dir / 'anomalies'
        self.models_dir = self.base_dir / 'models'

        # Create necessary directories
        self.anomalies_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _calculate_percentile(self, value: float, field_stats: Dict) -> float:
        """Calculate percentile for a value based on field statistics"""
        try:
            mean = field_stats["stats"]["mean"]
            std = field_stats["stats"]["std"]
            if std == 0:
                return 50.0  # Default to median if no variation

            z_score = (value - mean) / std
            return float(100 * 0.5 * (1 + np.erf(z_score / np.sqrt(2))))
        except (KeyError, TypeError):
            return None

    def _get_enum_rank(self, value: Any, field_stats: Dict) -> Optional[int]:
        """Get rank of enum value based on frequency"""
        try:
            frequencies = field_stats.get("distribution", {}).get("frequencies", {})
            sorted_values = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
            for rank, (val, _) in enumerate(sorted_values, 1):
                if str(val) == str(value):
                    return rank
            return None
        except (KeyError, TypeError):
            return None

    def store_anomaly(self,
                      query_id: str,
                      storage_key: str,
                      query_result: Dict,
                      context: Dict,
                      analysis_result: Dict) -> Optional[str]:
        """Store anomaly results with enhanced statistical analysis"""
        # Check if there were any anomalies
        if not (analysis_result.get("query_analysis", {}).get("anomalies_detected", False) or
                analysis_result.get("statistical_flags", 0) > 0 or
                len(analysis_result.get("statistical_analysis", {})
                            .get("significant_changes", [])) > 0):
            return None

        timestamp = datetime.now()

        # Create date-based directory structure
        date_path = timestamp.strftime("%Y/%m/%d")
        save_dir = self.anomalies_dir / date_path
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create the enhanced storage record
        storage_record = {
            "timestamp": timestamp.isoformat(),
            "query_hash": query_id,  # Store query hash
            "operation_name": context.get("operation_name"),  # Store operation name separately
            "context": {
                "raw_request": context.get("rawRequest", {}),
                "user": context.get("user"),
                "role": context.get("role"),
                "operation_name": context.get("operation_name"),
                "query_hash": context.get("query_hash")  # Include query hash in context
            },
            "statistical_analysis": analysis_result.get("statistical_analysis", {}),
            "anomalies": {
                "query_pattern_analysis": analysis_result.get("query_analysis", {})
                .get("query_pattern_analysis", {}),
                "anomalous_records": []
            }
        }

        # Include full anomalous records with their analysis
        if "records" in analysis_result.get("query_analysis", {}):
            for record_analysis in analysis_result["query_analysis"]["records"]:
                record_index = record_analysis.get("record_index")
                if record_index is not None and record_index < len(query_result.get("data", [])):
                    storage_record["anomalies"]["anomalous_records"].append({
                        **record_analysis,
                        "record": query_result["data"][record_index],
                        "statistical_context": self._get_record_statistical_context(
                            record=query_result["data"][record_index],
                            stats=analysis_result["statistical_analysis"]["new_data_stats"]
                        )
                    })

        # Generate filename using storage_key
        filename = f"{timestamp.strftime('%H%M%S')}_{storage_key}.json"
        file_path = save_dir / filename

        try:
            with open(file_path, 'w') as f:
                json.dump(storage_record, f, indent=2)
            self.logger.info(f"Stored anomaly analysis at {file_path}")
            return str(file_path)
        except Exception as e:
            self.logger.error(f"Error storing anomaly analysis: {str(e)}")
            raise

    def _get_record_statistical_context(self, record: Dict, stats: Dict) -> Dict:
        """Get statistical context for a specific record"""
        context = {}
        for field, value in record.items():
            field_stats = stats.get("fields", {}).get(field)
            if not field_stats:
                continue

            if field_stats["type"] == "numeric" and isinstance(value, (int, float)):
                mean = field_stats["stats"].get("mean")
                std = field_stats["stats"].get("std")
                if mean is not None and std is not None and std > 0:
                    z_score = (value - mean) / std
                    context[field] = {
                        "z_score": z_score,
                        "percentile": self._calculate_percentile(value, field_stats)
                    }
            elif field_stats["type"] == "enum":
                context[field] = {
                    "frequency": field_stats.get("distribution", {})
                    .get("frequencies", {}).get(str(value)),
                    "rank": self._get_enum_rank(value, field_stats)
                }

        return context

    def get_anomaly_history(self,
                            query_id: Optional[str] = None,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            role: Optional[str] = None) -> List[Dict]:
        """Retrieve anomaly history with filters"""
        anomalies = []

        try:
            # Convert dates if provided
            start_dt = datetime.fromisoformat(start_date) if start_date else None
            end_dt = datetime.fromisoformat(end_date) if end_date else None

            # Walk through anomalies directory
            for anomaly_file in self.anomalies_dir.rglob("*.json"):
                try:
                    with open(anomaly_file, 'r') as f:
                        anomaly = json.load(f)

                    # Apply filters
                    if query_id and anomaly['query_hash'] != query_id:  # Updated to use query_hash
                        continue

                    if role and anomaly['context']['role'] != role:
                        continue

                    anomaly_dt = datetime.fromisoformat(anomaly['timestamp'])
                    if start_dt and anomaly_dt < start_dt:
                        continue
                    if end_dt and anomaly_dt > end_dt:
                        continue

                    anomalies.append(anomaly)

                except Exception as e:
                    self.logger.error(f"Error reading anomaly file {anomaly_file}: {str(e)}")
                    continue

        except Exception as e:
            self.logger.error(f"Error retrieving anomaly history: {str(e)}")
            raise

        return anomalies

    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """Clean up anomaly records older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        removed_count = 0

        try:
            for file_path in self.anomalies_dir.rglob("*.json"):
                try:
                    # Check file modification time
                    if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_date:
                        file_path.unlink()
                        removed_count += 1
                except Exception as e:
                    self.logger.error(f"Error removing file {file_path}: {str(e)}")
                    continue

            # Remove empty directories
            for dir_path in self.anomalies_dir.rglob("*"):
                if dir_path.is_dir() and not any(dir_path.iterdir()):
                    dir_path.rmdir()

            return removed_count

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise
