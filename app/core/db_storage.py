import json
from typing import Dict, List, Optional, Type
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Float, ForeignKey, Boolean, Text, Enum, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import func
import enum
import logging
import pickle
from sqlalchemy import LargeBinary

Base = declarative_base()


class RiskLevel(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class TrainedModel(Base):
    __tablename__ = 'trained_models'

    id = Column(Integer, primary_key=True)
    query_id = Column(String(255), nullable=False, index=True)
    model_data = Column(LargeBinary, nullable=False)  # Pickled model data
    features_count = Column(Integer)
    records_count = Column(Integer)
    last_updated = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), default=func.now())

    __table_args__ = (
        Index('idx_model_query', 'query_id', 'last_updated'),
    )

class AnomalyAnalysis(Base):
    __tablename__ = 'anomaly_analyses'

    id = Column(Integer, primary_key=True)
    query_id = Column(String(255), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), default=func.now(), index=True)
    operation_name = Column(String(255))
    user_id = Column(String(255), index=True)
    role = Column(String(100))
    processed_records = Column(Integer)
    anomalous_records_count = Column(Integer)
    historical_data_size = Column(Integer)
    analysis_mode = Column(String(50))
    anomalies_detected = Column(Boolean)
    status = Column(String(50))

    # Relationships
    records = relationship("AnomalyRecord", back_populates="analysis")
    concerns = relationship("QueryConcern", back_populates="analysis")
    recommendations = relationship("Recommendation", back_populates="analysis")
    statistical_flags = relationship("StatisticalFlag", back_populates="analysis")

    __table_args__ = (
        Index('idx_query_time', 'query_id', 'timestamp'),
        Index('idx_user_time', 'user_id', 'timestamp'),
    )


class AnomalyRecord(Base):
    __tablename__ = 'anomaly_records'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('anomaly_analyses.id'), nullable=False)
    category = Column(String(100))
    reason = Column(Text)
    record_index = Column(Integer)
    risk_level = Column(Enum(RiskLevel))

    # Relationships
    analysis = relationship("AnomalyAnalysis", back_populates="records")
    field_stats = relationship("FieldStatistic", back_populates="record")


class QueryConcern(Base):
    __tablename__ = 'query_concerns'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('anomaly_analyses.id'), nullable=False)
    description = Column(Text, nullable=False)

    # Relationships
    analysis = relationship("AnomalyAnalysis", back_populates="concerns")


class Recommendation(Base):
    __tablename__ = 'recommendations'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('anomaly_analyses.id'), nullable=False)
    description = Column(Text, nullable=False)

    # Relationships
    analysis = relationship("AnomalyAnalysis", back_populates="recommendations")


class FieldStatistic(Base):
    __tablename__ = 'field_statistics'

    id = Column(Integer, primary_key=True)
    record_id = Column(Integer, ForeignKey('anomaly_records.id'), nullable=False)
    field_name = Column(String(255), nullable=False)
    field_type = Column(String(50))
    avg_length = Column(Float)
    unique_count = Column(Integer)
    uniqueness_ratio = Column(Float)
    min_value = Column(Float)
    max_value = Column(Float)
    mean_value = Column(Float)
    median_value = Column(Float)
    std_dev = Column(Float)

    # Relationships
    record = relationship("AnomalyRecord", back_populates="field_stats")
    value_distribution = relationship("ValueDistribution", back_populates="field_statistic")


class ValueDistribution(Base):
    __tablename__ = 'value_distributions'

    id = Column(Integer, primary_key=True)
    field_statistic_id = Column(Integer, ForeignKey('field_statistics.id'), nullable=False)
    value = Column(String(255))
    count = Column(Integer)

    # Relationships
    field_statistic = relationship("FieldStatistic", back_populates="value_distribution")


class StatisticalFlag(Base):
    __tablename__ = 'statistical_flags'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('anomaly_analyses.id'), nullable=False)
    flag_type = Column(String(100))
    description = Column(Text)
    severity = Column(Float)

    # Relationships
    analysis = relationship("AnomalyAnalysis", back_populates="statistical_flags")


class DatabaseStorage:
    def __init__(self,
                 connection_url: str,
                 historical_retention_days: int = 14,
                 anomaly_retention_days: int = 30,
                 model_retention_days: int = 7,
                 retrain_threshold: int = 100):
        """Initialize database storage with configurable parameters

        Args:
            connection_url: Database connection URL
            historical_retention_days: Days to retain historical data
            anomaly_retention_days: Days to retain anomaly records
            model_retention_days: Days to retain trained models
            retrain_threshold: Number of records before model retraining
        """
        self.engine = create_engine(connection_url, pool_size=20, max_overflow=0)
        self.Session = sessionmaker(bind=self.engine)
        self.logger = logging.getLogger(__name__)
        self.historical_retention_days = historical_retention_days
        self.anomaly_retention_days = anomaly_retention_days
        self.model_retention_days = model_retention_days
        self.retrain_threshold = retrain_threshold
        Base.metadata.create_all(self.engine)

    def save_model(self, query_id: str, model_data: Dict) -> bool:
        """Save trained model to database"""
        session = self.Session()
        try:
            # Pickle the model data
            pickled_data = pickle.dumps(model_data)

            # Check for existing model
            model_record = session.query(TrainedModel).filter_by(query_id=query_id).first()

            if model_record:
                # Update existing model
                model_record.model_data = pickled_data
                model_record.features_count = model_data.get('num_features', 0)
                model_record.records_count = model_data.get('num_records', 0)
                model_record.last_updated = func.now()
            else:
                # Create new model record
                model_record = TrainedModel(
                    query_id=query_id,
                    model_data=pickled_data,
                    features_count=model_data.get('num_features', 0),
                    records_count=model_data.get('num_records', 0)
                )
                session.add(model_record)

            session.commit()
            self.logger.debug(f"Saved model for query_hash {query_id}")
            return True

        except Exception as e:
            session.rollback()
            self.logger.error(f"Error saving model for {query_id}: {str(e)}")
            raise
        finally:
            session.close()

    def load_model(self, query_id: str) -> Optional[Dict]:
        """Load trained model from database"""
        session = self.Session()
        try:
            model_record: Optional[Type[TrainedModel]] = session.query(TrainedModel).filter_by(query_id=query_id).first()
            if model_record:
                try:
                    model_data = pickle.loads(model_record.model_data)
                    return model_data
                except Exception as e:
                    self.logger.error(f"Error unpickling model data for {query_id}: {str(e)}")
                    return None
            return None

        except Exception as e:
            self.logger.error(f"Error loading model for {query_id}: {str(e)}")
            return None
        finally:
            session.close()

    def cleanup_old_models(self) -> int:
        """Clean up old models based on model retention period"""
        session = self.Session()
        try:
            cutoff_date = datetime.now() - timedelta(days=self.model_retention_days)
            result = session.query(TrainedModel).filter(
                TrainedModel.last_updated < cutoff_date
            ).delete()
            session.commit()
            self.logger.info(f"Cleaned up {result} models older than {self.model_retention_days} days")
            return result
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error cleaning up old models: {str(e)}")
            raise
        finally:
            session.close()

    def cleanup_all_old_data(self) -> Dict[str, int]:
        """Clean up all old data including models"""
        try:
            historical_count = self.cleanup_historical_data()
            anomaly_count = self.cleanup_anomaly_records()
            model_count = self.cleanup_old_models()

            cleanup_summary = {
                "historical_records_removed": historical_count,
                "anomaly_records_removed": anomaly_count,
                "models_removed": model_count,
                "retention_periods": {
                    "historical_days": self.historical_retention_days,
                    "anomaly_days": self.anomaly_retention_days,
                    "model_days": self.model_retention_days
                }
            }

            self.logger.info("Cleanup completed: %s", cleanup_summary)
            return cleanup_summary
        except Exception as e:
            self.logger.error(f"Error during complete data cleanup: {str(e)}")
            raise

    def remove_historical_data(self, query_id: str, records_to_remove: List[Dict]):
        """Remove specified records from the historical data in the database"""
        session = self.Session()
        try:
            # Get the historical data record for the given query_id
            historical_data = session.query(HistoricalData).filter_by(query_id=query_id).first()

            if historical_data:
                # Load the existing data
                data = json.loads(historical_data.data)

                # Remove the specified records
                data = [record for record in data if record not in records_to_remove]

                # Update the historical data record with the modified data
                historical_data.data = json.dumps(data)
                historical_data.record_count = len(data)
                historical_data.last_updated = func.now()

                session.commit()
                self.logger.debug(f"Removed {len(records_to_remove)} records for query_hash {query_id}")
            else:
                self.logger.warning(f"No historical data found for query_id: {query_id}")

        except Exception as e:
            session.rollback()
            self.logger.error(f"Error removing historical data for {query_id}: {str(e)}")
            raise
        finally:
            session.close()

    def cleanup_historical_data(self) -> int:
        """Clean up historical data based on retention period"""
        session = self.Session()
        try:
            cutoff_date = datetime.now() - timedelta(days=self.historical_retention_days)
            result = session.query(HistoricalData).filter(
                HistoricalData.last_updated < cutoff_date
            ).delete()
            session.commit()
            self.logger.info(f"Cleaned up {result} historical records older than {self.historical_retention_days} days")
            return result
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error during historical data cleanup: {str(e)}")
            raise
        finally:
            session.close()

    def cleanup_anomaly_records(self) -> int:
        """Clean up anomaly records based on retention period"""
        session = self.Session()
        try:
            cutoff_date = datetime.now() - timedelta(days=self.anomaly_retention_days)
            # First get IDs of old analyses
            old_analysis_ids = [id[0] for id in session.query(AnomalyAnalysis.id).filter(
                AnomalyAnalysis.timestamp < cutoff_date
            ).all()]

            if not old_analysis_ids:
                return 0

            # Delete related records first
            for table in [StatisticalFlag, Recommendation, QueryConcern, AnomalyRecord]:
                session.query(table).filter(
                    table.analysis_id.in_(old_analysis_ids)
                ).delete(synchronize_session=False)

            # Finally delete the analyses
            result = session.query(AnomalyAnalysis).filter(
                AnomalyAnalysis.id.in_(old_analysis_ids)
            ).delete(synchronize_session=False)

            session.commit()
            self.logger.info(f"Cleaned up {result} anomaly analyses older than {self.anomaly_retention_days} days")
            return result
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error during anomaly record cleanup: {str(e)}")
            raise
        finally:
            session.close()

    @staticmethod
    def _get_severity_value(risk_level: str) -> float:
        """Convert risk level to numerical severity value"""
        severity_map = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.9,
            'critical': 1.0
        }
        return severity_map.get(risk_level.lower(), 0.5)

    def _store_field_statistics(self, session, record_id: int, field_stats: Dict):
        """Store field statistics and value distributions for a record"""
        self.logger.debug(f"Starting _store_field_statistics for record_id: {record_id}")
        self.logger.debug(f"Field stats keys: {list(field_stats.keys())}")

        for field_name, field_data in field_stats.items():
            stats = field_data.get('stats', {})
            if not stats:
                self.logger.debug(f"No stats found for field: {field_name}")
                continue

            self.logger.debug(f"Processing field: {field_name}, type: {field_data.get('type')}")
            self.logger.debug(f"Stats for {field_name}: {stats}")

            # Create field statistics record
            field_stat = FieldStatistic(
                record_id=record_id,
                field_name=field_name,
                field_type=field_data.get('type'),
                avg_length=stats.get('avg_length'),
                unique_count=stats.get('unique_count'),
                uniqueness_ratio=stats.get('uniqueness_ratio'),
                min_value=field_data.get('numeric_stats', {}).get('stats', {}).get('min'),
                max_value=field_data.get('numeric_stats', {}).get('stats', {}).get('max'),
                mean_value=field_data.get('numeric_stats', {}).get('stats', {}).get('mean'),
                median_value=field_data.get('numeric_stats', {}).get('stats', {}).get('median'),
                std_dev=field_data.get('numeric_stats', {}).get('stats', {}).get('std')
            )
            session.add(field_stat)
            session.flush()
            self.logger.debug(f"Created field statistic record for {field_name} with id: {field_stat.id}")

            # Store value distribution for this field
            value_dist = stats.get('value_distribution', {})
            self.logger.debug(f"Value distribution for {field_name}: {value_dist}")

            for value, count in value_dist.items():
                dist_record = ValueDistribution(
                    field_statistic=field_stat,
                    value=str(value),
                    count=count
                )
                session.add(dist_record)
                self.logger.debug(f"Added value distribution record for {field_name}: {value}={count}")

        self.logger.debug("Completed _store_field_statistics")

    def store_anomaly(self, query_id: str, query_result: Dict, context: Dict, analysis_result: Dict) -> Optional[int]:
        """Store anomaly results in normalized database tables"""
        query_analysis = analysis_result.get("query_analysis", {})
        has_anomalies = (
                query_analysis.get("anomalies_detected", False) or
                analysis_result.get("statistical_flags", 0) > 0 or
                len(analysis_result.get("statistical_analysis", {})
                    .get("significant_changes", [])) > 0 or
                len(query_analysis.get("records", [])) > 0
        )

        if not has_anomalies:
            return None

        session = self.Session()
        try:
            # Create main analysis record
            analysis = AnomalyAnalysis(
                query_id=query_id,
                operation_name=context.get('operation_name'),
                user_id=context.get('user'),
                role=context.get('role'),
                processed_records=len(query_result.get("data", [])),
                anomalous_records_count=len(query_analysis.get("records", [])),
                historical_data_size=analysis_result.get("metadata", {}).get("historical_data_size"),
                analysis_mode=query_analysis.get("analysis_mode"),
                anomalies_detected=query_analysis.get("anomalies_detected"),
                status=analysis_result.get("metadata", {}).get("status")
            )
            session.add(analysis)
            session.flush()
            self.logger.debug(f"Created main analysis record with id: {analysis.id}")

            # Create a base record for overall statistics
            base_record = AnomalyRecord(
                analysis=analysis,
                category="overall_statistics",
                reason="Dataset-wide statistics",
                record_index=-1,
                risk_level=RiskLevel.LOW
            )
            session.add(base_record)
            session.flush()
            self.logger.debug(f"Created base record for statistics with id: {base_record.id}")

            # Store field statistics and value distributions
            # Fix: Get field stats from the correct path in query_analysis
            field_stats = query_analysis.get("statistical_analysis", {}).get("new_data_stats", {}).get("fields", {})
            if field_stats:
                self.logger.debug(f"Found field stats with keys: {list(field_stats.keys())}")
                self._store_field_statistics(session, base_record.id, field_stats)
            else:
                self.logger.debug("No field stats found in analysis result")

            # Store anomaly records
            anomaly_records = query_analysis.get("records", [])
            self.logger.debug(f"Processing {len(anomaly_records)} anomaly records")

            for record in anomaly_records:
                anomaly_record = AnomalyRecord(
                    analysis=analysis,
                    category=record.get('category'),
                    reason=record.get('reason'),
                    record_index=record.get('record_index'),
                    risk_level=RiskLevel(record.get('risk_level', 'low'))
                )
                session.add(anomaly_record)
                session.flush()
                self.logger.debug(f"Created anomaly record: {anomaly_record.id}, category: {record.get('category')}")

                # Create statistical flags for each anomaly
                session.add(StatisticalFlag(
                    analysis=analysis,
                    flag_type=record.get('category', 'unknown'),
                    description=record.get('reason', ''),
                    severity=self._get_severity_value(record.get('risk_level', 'low'))
                ))
                self.logger.debug(f"Added statistical flag for record {anomaly_record.id}")

            # Store concerns
            concerns = query_analysis.get("query_pattern_analysis", {}).get("concerns", [])
            self.logger.debug(f"Processing {len(concerns)} concerns")
            for concern in concerns:
                session.add(QueryConcern(
                    analysis=analysis,
                    description=concern
                ))

            # Store recommendations
            recommendations = analysis_result.get("recommendations", [])
            self.logger.debug(f"Processing {len(recommendations)} recommendations")
            for recommendation in recommendations:
                session.add(Recommendation(
                    analysis=analysis,
                    description=recommendation
                ))

            session.commit()
            self.logger.info(f"Successfully stored anomaly record with ID: {analysis.id}")
            return analysis.id

        except Exception as e:
            session.rollback()
            self.logger.error(f"Error storing anomaly record: {str(e)}", exc_info=True)
            raise
        finally:
            session.close()

    def load_historical_data(self, query_id: str) -> List[Dict]:
        """Load historical data for a query"""
        session = self.Session()
        try:
            record = session.query(HistoricalData).filter_by(query_id=query_id).first()
            if record:
                return json.loads(record.data)
            return []
        except Exception as e:
            self.logger.error(f"Error loading historical data for {query_id}: {str(e)}")
            raise
        finally:
            session.close()

    def save_historical_data(self, query_id: str, data: List[Dict]) -> bool:
        """Save historical data for a query"""
        session = self.Session()
        try:
            record = session.query(HistoricalData).filter_by(query_id=query_id).first()
            if record:
                record.data = json.dumps(data)
                record.record_count = len(data)
                record.last_updated = func.now()
            else:
                record = HistoricalData(
                    query_id=query_id,
                    data=json.dumps(data),
                    record_count=len(data)
                )
                session.add(record)

            session.commit()
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error saving historical data for {query_id}: {str(e)}")
            raise
        finally:
            session.close()

    def clear_historical_data(self, query_id: str = None) -> bool:
        """Clear historical data for a specific query or all queries"""
        session = self.Session()
        try:
            if query_id:
                session.query(HistoricalData).filter_by(query_id=query_id).delete()
            else:
                session.query(HistoricalData).delete()
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error clearing historical data: {str(e)}")
            raise
        finally:
            session.close()

    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """Clean up historical data older than specified days"""
        session = self.Session()
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            result = session.query(HistoricalData).filter(
                HistoricalData.last_updated < cutoff_date
            ).delete()
            session.commit()
            return result
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise
        finally:
            session.close()

class HistoricalData(Base):
    __tablename__ = 'historical_data'

    id = Column(Integer, primary_key=True)
    query_id = Column(String(255), nullable=False, index=True)
    data = Column(Text, nullable=False)  # JSON blob
    last_updated = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), default=func.now())
    record_count = Column(Integer, default=0)

    __table_args__ = (
        Index('idx_query_updated', 'query_id', 'last_updated'),
    )
