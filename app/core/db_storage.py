import enum
import hashlib
import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Type

from graphql import parse, visit, Visitor, GraphQLError
from sqlalchemy import LargeBinary
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Float, ForeignKey, Boolean, Text, Enum, Index
from sqlalchemy import text, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import func

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
    field_statistics = relationship("FieldStatistic", back_populates="record")


class QueryFeature(Base):
    __tablename__ = 'query_features'

    id = Column(Integer, primary_key=True)
    pattern_id = Column(Integer, ForeignKey('query_patterns.id'), nullable=False)
    feature_name = Column(String(50), nullable=False)
    feature_value = Column(Float, nullable=False)
    feature_type = Column(String(10), nullable=False)  # 'int', 'bool', or 'float'

    pattern = relationship("QueryPattern", back_populates="features")

    __table_args__ = (
        Index('idx_query_features_pattern_name', 'pattern_id', 'feature_name'),
        Index('idx_query_features_name_value', 'feature_name', 'feature_value'),
    )

    def get_typed_value(self):
        """Return the feature value converted to its proper type"""
        if self.feature_type == 'int':
            return int(self.feature_value)
        elif self.feature_type == 'bool':
            return bool(self.feature_value)
        return self.feature_value


class QueryPattern(Base):
    __tablename__ = 'query_patterns'

    id = Column(Integer, primary_key=True)
    query_hash = Column(String(32), nullable=False, index=True)
    operation_name = Column(String(255), nullable=True)
    query_text = Column(Text, nullable=False)
    variables_json = Column(Text, nullable=True)
    complexity_score = Column(Float, nullable=False)
    user_role = Column(String(50), nullable=False)
    is_anomalous = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)

    features = relationship("QueryFeature", back_populates="pattern", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_query_patterns_hash_role', 'query_hash', 'user_role'),
        Index('idx_query_patterns_created', 'created_at'),
        Index('idx_query_patterns_anomaly', 'is_anomalous'),
    )

    class GraphQLAnalyzer(Visitor):
        def __init__(self):
            super().__init__()  # Call parent class constructor
            self.depth = 0
            self.max_depth = 0
            self.fields = set()
            self.arguments = []
            self.variables = set()
            self.operation_name = None

        def enter(self, node, *args):
            if hasattr(node, 'kind'):
                # Handle operation definition
                if node.kind == 'operation_definition':
                    if hasattr(node, 'name') and node.name:
                        self.operation_name = node.name.value
                    if hasattr(node, 'variable_definitions') and node.variable_definitions:
                        self.variables.update(var.variable.name.value for var in node.variable_definitions)

                # Handle field
                elif node.kind == 'field':
                    self.depth += 1
                    self.max_depth = max(self.max_depth, self.depth)

                    if hasattr(node, 'name') and not node.name.value.startswith('__'):
                        self.fields.add(node.name.value)

                    if hasattr(node, 'arguments') and node.arguments:
                        self.arguments.extend(arg.name.value for arg in node.arguments)

        def leave(self, node, *args):
            if hasattr(node, 'kind') and node.kind == 'field':
                self.depth -= 1

    @classmethod
    def create_and_store_from_query(cls, session, query_text: str, variables_json: Optional[str],
                                    user_role: str, is_anomalous: bool) -> 'QueryPattern':
        """Create, analyze, and store a QueryPattern instance from a GraphQL query"""
        try:
            # Parse the query
            ast = parse(query_text)
            analyzer = cls.GraphQLAnalyzer()
            visit(ast, analyzer)

            # Parse variables JSON
            variables = variables_json if variables_json else {}

            # Compute pagination/filtering/sorting flags
            pagination_args = {'first', 'last', 'limit', 'offset', 'page', 'pageSize'}
            filter_args = {'filter', 'where', 'search', 'query', 'conditions'}
            sort_args = {'orderBy', 'sortBy', 'sort', 'order', 'orderByDirection'}

            arg_set = set(analyzer.arguments)
            has_pagination = bool(pagination_args & arg_set)
            has_filtering = bool(filter_args & arg_set)
            has_sorting = bool(sort_args & arg_set)

            # Calculate complexity score
            complexity_score = (
                    len(analyzer.fields) * 2.0 +  # Base score from fields
                    analyzer.max_depth * 3.0 +  # Depth multiplier
                    len(analyzer.arguments) * 1.5  # Argument complexity
            )

            # Create pattern instance
            pattern = cls(
                query_hash=hashlib.md5(f"{query_text}{variables_json}".encode()).hexdigest(),
                operation_name=analyzer.operation_name,
                query_text=query_text,
                variables_json=json.dumps(variables_json),
                complexity_score=round(complexity_score, 2),
                user_role=user_role,
                is_anomalous=is_anomalous
            )

            # Create features
            pattern.features = [
                QueryFeature(
                    feature_name='query_length',
                    feature_value=float(len(query_text)),
                    feature_type='int'
                ),
                QueryFeature(
                    feature_name='depth',
                    feature_value=float(analyzer.max_depth),
                    feature_type='int'
                ),
                QueryFeature(
                    feature_name='fields',
                    feature_value=float(len(analyzer.fields)),
                    feature_type='int'
                ),
                QueryFeature(
                    feature_name='arguments',
                    feature_value=float(len(analyzer.arguments)),
                    feature_type='int'
                ),
                QueryFeature(
                    feature_name='variables',
                    feature_value=float(len(analyzer.variables)),
                    feature_type='int'
                ),
                QueryFeature(
                    feature_name='has_pagination',
                    feature_value=float(has_pagination),
                    feature_type='bool'
                ),
                QueryFeature(
                    feature_name='has_filtering',
                    feature_value=float(has_filtering),
                    feature_type='bool'
                ),
                QueryFeature(
                    feature_name='has_sorting',
                    feature_value=float(has_sorting),
                    feature_type='bool'
                ),
                QueryFeature(
                    feature_name='complexity_score',
                    feature_value=complexity_score,
                    feature_type='float'
                )
            ]

            # Store in database
            session.add(pattern)
            session.commit()

            return pattern

        except GraphQLError as e:
            session.rollback()
            raise ValueError(f"Invalid GraphQL query: {str(e)}")
        except json.JSONDecodeError:
            session.rollback()
            raise ValueError("Invalid variables JSON")
        except Exception as e:
            session.rollback()
            raise ValueError(f"Error analyzing query: {str(e)}")

    def get_feature_value(self, feature_name: str):
        """Helper method to get a specific feature value"""
        feature = next((f for f in self.features if f.feature_name == feature_name), None)
        if feature:
            return feature.get_typed_value()
        return None

    def __repr__(self):
        return (f"<QueryPattern(id={self.id}, "
                f"operation_name='{self.operation_name}', "
                f"complexity_score={self.complexity_score}, "
                f"is_anomalous={self.is_anomalous})>")


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
    record = relationship("AnomalyRecord", back_populates="field_statistics")  # Changed from field_stats
    value_distributions = relationship("ValueDistribution", back_populates="field_statistic")  # Changed to plural


class ValueDistribution(Base):
    __tablename__ = 'value_distributions'

    id = Column(Integer, primary_key=True)
    field_statistic_id = Column(Integer, ForeignKey('field_statistics.id'), nullable=False)
    value = Column(String(255))
    count = Column(Integer)

    # Relationships
    field_statistic = relationship("FieldStatistic", back_populates="value_distributions")  # Changed to plural


class StatisticalFlag(Base):
    __tablename__ = 'statistical_flags'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('anomaly_analyses.id'), nullable=False)
    flag_type = Column(String(100))
    description = Column(Text)
    severity = Column(Float)

    # Relationships
    analysis = relationship("AnomalyAnalysis", back_populates="statistical_flags")


from sqlalchemy import text


def ensure_schema_exists(engine, connect_args):
    """
    Ensures schema exists based on connect_args for the specific database dialect.

    Args:
        engine: SQLAlchemy engine
        connect_args: The connect_args dictionary passed to create_engine
    """
    dialect_name = engine.dialect.name
    schema_name = None

    # Extract schema based on dialect and connect_args
    if dialect_name == 'postgresql':
        options = connect_args.get('options', '')
        if options and 'search_path=' in options:
            # Extract first schema from search_path
            if '-c search_path=' in options:
                search_path = options.split('-c search_path=')[1].split()[0]
            else:
                search_path = options.split('search_path=')[1].split()[0]

            # Get first schema in the path
            schema_name = search_path.split(',')[0].strip()

            # Don't create 'public' schema as it exists by default
            if schema_name == 'public':
                return

    elif dialect_name in ('mysql', 'mariadb'):
        init_command = connect_args.get('init_command', '')
        if init_command and 'default_schema=' in init_command:
            schema_name = init_command.split('default_schema=')[1].split(';')[0].strip()

    elif dialect_name == 'oracle':
        # For Oracle, schema creation usually requires separate admin privileges
        # Just log that we don't handle this automatically
        print("Note: Oracle schemas typically need to be created by a DBA")
        return

    elif dialect_name == 'mssql':
        # For SQL Server, schema might be in the connect_args
        schema_name = connect_args.get('schema', None)

    # Skip if no schema name found or it's a default schema
    if not schema_name or schema_name.lower() in ('public', 'dbo', 'default'):
        return

    # Create schema based on dialect
    with engine.connect() as conn:
        try:
            if dialect_name == 'postgresql':
                conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))
                conn.commit()
                print(f"Created schema '{schema_name}' in PostgreSQL database")

            elif dialect_name in ('mysql', 'mariadb'):
                conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {schema_name}"))
                conn.commit()
                print(f"Created database '{schema_name}' in MySQL/MariaDB")

            elif dialect_name == 'mssql':
                conn.execute(text(
                    f"IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = '{schema_name}') "
                    f"EXEC('CREATE SCHEMA [{schema_name}]')"))
                conn.commit()
                print(f"Created schema '{schema_name}' in SQL Server database")

            # SQLite doesn't support schemas in the same way

        except Exception as e:
            print(f"Error creating schema: {e}")
            # Log error but don't raise to prevent application startup failure


class DatabaseStorage:
    def __init__(self,
                 connection_url: str,
                 connect_args: Optional[Dict] = None,
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
        if connect_args:
            self.engine = create_engine(connection_url, pool_size=20, max_overflow=0, connect_args=connect_args)
            ensure_schema_exists(self.engine, connect_args)
        else:
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
            for table_class in [StatisticalFlag, Recommendation, QueryConcern, AnomalyRecord]:
                session.query(table_class).filter(
                    getattr(table_class, 'analysis_id').in_(old_analysis_ids)
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

        try:
            for field_name, field_data in field_stats.items():
                stats = field_data.get('stats', {})
                if not stats:
                    self.logger.debug(f"No stats found for field: {field_name}")
                    continue

                # Create field statistics record
                field_stat = FieldStatistic(
                    record_id=record_id,
                    field_name=field_name,
                    field_type=field_data.get('type'),
                    avg_length=stats.get('avg_length'),
                    unique_count=stats.get('unique_count'),
                    uniqueness_ratio=stats.get('uniqueness_ratio'),
                    min_value=field_data.get('numeric_stats', {}).get('min'),
                    max_value=field_data.get('numeric_stats', {}).get('max'),
                    mean_value=field_data.get('numeric_stats', {}).get('mean'),
                    median_value=field_data.get('numeric_stats', {}).get('median'),
                    std_dev=field_data.get('numeric_stats', {}).get('std')
                )
                session.add(field_stat)
                session.flush()

                # Store value distributions for this field
                value_dist = stats.get('value_distribution', {})
                for value, count in value_dist.items():
                    dist_record = ValueDistribution(
                        field_statistic_id=field_stat.id,
                        value=str(value),
                        count=count
                    )
                    session.add(dist_record)

            session.flush()
            self.logger.debug("Successfully stored field statistics and distributions")

        except Exception as e:
            self.logger.error(f"Error in _store_field_statistics: {str(e)}", exc_info=True)
            raise

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
            session.flush()  # Flush to get the analysis.id
            self.logger.debug(f"Created main analysis record with id: {analysis.id}")

            # Create a base record for overall statistics
            base_record = AnomalyRecord(
                analysis_id=analysis.id,  # Use analysis_id instead of analysis relationship
                category="overall_statistics",
                reason="Dataset-wide statistics",
                record_index=-1,
                risk_level=RiskLevel.LOW
            )
            session.add(base_record)
            session.flush()
            self.logger.debug(f"Created base record for statistics with id: {base_record.id}")

            # Store field statistics and value distributions
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
                    analysis_id=analysis.id,  # Use analysis_id instead of analysis relationship
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
                    analysis_id=analysis.id,  # Use analysis_id instead of analysis relationship
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
                    analysis_id=analysis.id,  # Use analysis_id instead of analysis relationship
                    description=concern
                ))

            # Store recommendations
            recommendations = analysis_result.get("recommendations", [])
            self.logger.debug(f"Processing {len(recommendations)} recommendations")
            for recommendation in recommendations:
                session.add(Recommendation(
                    analysis_id=analysis.id,  # Use analysis_id instead of analysis relationship
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

    def save_historical_data(self, query_id: str, data: List[Dict]) -> bool:
        """Save historical data for a query"""
        session = self.Session()
        try:
            # Find or create historical data record
            historical_data = session.query(HistoricalData).filter_by(query_id=query_id).first()
            if not historical_data:
                historical_data = HistoricalData(
                    query_id=query_id,
                    record_count=len(data)
                )
                session.add(historical_data)
                session.flush()

            # Delete existing records if updating
            if historical_data.historical_records:
                for record in historical_data.historical_records:
                    session.delete(record)

            # Create new historical records
            for item in data:
                record = HistoricalRecord(
                    historical_data_id=historical_data.id,
                    record_data=json.dumps(item)
                )
                session.add(record)

            historical_data.record_count = len(data)
            historical_data.last_updated = func.now()

            session.commit()
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error saving historical data for {query_id}: {str(e)}")
            raise
        finally:
            session.close()

    def load_historical_data(self, query_id: str) -> List[Dict]:
        """Load historical data for a query"""
        session = self.Session()
        try:
            historical_data = session.query(HistoricalData).filter_by(query_id=query_id).first()
            if historical_data and historical_data.historical_records:
                return [
                    json.loads(record.record_data)
                    for record in historical_data.historical_records
                ]
            return []
        except Exception as e:
            self.logger.error(f"Error loading historical data for {query_id}: {str(e)}")
            raise
        finally:
            session.close()

    def remove_historical_data(self, query_id: str, records_to_remove: List[Dict]):
        """Remove specified records from the historical data in the database"""
        session = self.Session()
        try:
            historical_data = session.query(HistoricalData).filter_by(query_id=query_id).first()
            if historical_data:
                records_json = [json.dumps(record) for record in records_to_remove]
                removed_count = 0

                # Remove matching records
                for record in historical_data.historical_records[:]:  # Create a copy of the list
                    if record.record_data in records_json:
                        session.delete(record)
                        removed_count += 1

                # Update record count
                historical_data.record_count -= removed_count
                historical_data.last_updated = func.now()

                session.commit()
                self.logger.debug(f"Removed {removed_count} records for query_hash {query_id}")
            else:
                self.logger.warning(f"No historical data found for query_id: {query_id}")

        except Exception as e:
            session.rollback()
            self.logger.error(f"Error removing historical data for {query_id}: {str(e)}")
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

    def save_query_pattern(self, query_text: str, variables_json: Optional[str],
                                    user_role: str, is_anomalous: bool):
        QueryPattern.create_and_store_from_query(
            self.Session(),
            query_text=query_text,
            variables_json=variables_json,
            user_role=user_role,
            is_anomalous=is_anomalous
    )

class HistoricalData(Base):
    __tablename__ = 'historical_data'

    id = Column(Integer, primary_key=True)
    query_id = Column(String(255), nullable=False, index=True)
    record_count = Column(Integer, default=0)
    last_updated = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), default=func.now())

    # Relationship to HistoricalRecords
    historical_records = relationship("HistoricalRecord", back_populates="historical_data", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_query_updated', 'query_id', 'last_updated'),
    )

class HistoricalRecord(Base):
    __tablename__ = 'historical_records'

    id = Column(Integer, primary_key=True)
    historical_data_id = Column(Integer, ForeignKey('historical_data.id'), nullable=False)
    record_data = Column(Text, nullable=False)  # JSON blob for individual record
    last_updated = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), default=func.now())

    # Relationship to HistoricalData
    historical_data = relationship("HistoricalData", back_populates="historical_records")

    __table_args__ = (
        Index('idx_historical_data', 'historical_data_id'),
    )

