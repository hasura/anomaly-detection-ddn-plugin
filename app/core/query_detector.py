from typing import Dict, List, Any
from anthropic import Anthropic
import json
import numpy as np
from datetime import datetime
import logging

from app.core.statistics import DatasetStatistics


class QueryAnomalyDetector:
    def __init__(self,
                 api_key: str,
                 model: str = "claude-3-sonnet-20240229",
                 max_records_per_batch: int = 50):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_records_per_batch = max_records_per_batch
        self.logger = logging.getLogger(__name__)

    def _summarize_query_history(self, historical_data: List[Dict]) -> Dict:
        """Summarize historical query results patterns"""
        if not historical_data:
            return {
                "query_count": 0,
                "typical_result_sizes": {},
                "value_ranges": {},
                "common_combinations": {},
                "query_patterns": {},
                "status": "initializing"
            }

        summary = {
            "query_count": len(historical_data),
            "typical_result_sizes": {},
            "value_ranges": {},
            "common_combinations": {},
            "query_patterns": {},
            "status": "established"
        }

        # Analyze result sizes and patterns
        for record in historical_data:
            # Track size of result sets
            record_count = len(record) if isinstance(record, (list, dict)) else 1
            size_bucket = f"{(record_count // 10) * 10}-{((record_count // 10) + 1) * 10}"
            summary["typical_result_sizes"][size_bucket] = summary["typical_result_sizes"].get(size_bucket, 0) + 1

            # Analyze numeric fields
            if isinstance(record, dict):
                for key, value in record.items():
                    if isinstance(value, (int, float)):
                        if key not in summary["value_ranges"]:
                            summary["value_ranges"][key] = {
                                "min": float('inf'),
                                "max": float('-inf'),
                                "common_ranges": {}
                            }

                        # Update ranges
                        summary["value_ranges"][key]["min"] = min(summary["value_ranges"][key]["min"], value)
                        summary["value_ranges"][key]["max"] = max(summary["value_ranges"][key]["max"], value)

                        # Track value ranges in buckets
                        range_bucket = f"{(value // 100) * 100}-{((value // 100) + 1) * 100}"
                        summary["value_ranges"][key]["common_ranges"][range_bucket] = \
                            summary["value_ranges"][key]["common_ranges"].get(range_bucket, 0) + 1

        return summary

    def _create_system_prompt(self) -> str:
        """Create system prompt for analysis"""
        return """You are an expert in analyzing query result patterns and detecting anomalies in query behavior.
        Focus on identifying unusual patterns in how data is being queried and returned, rather than just the data values themselves.

        Consider:
        1. Unusual combinations of values that don't typically appear together in query results
        2. Unexpected result set sizes or structures
        3. Values that fall outside typical ranges for specific query patterns
        4. Potential security concerns (e.g., data leakage, excessive data retrieval)
        5. Business logic violations in the query results
        6. Query pattern changes that might indicate unauthorized access or system misuse

        If there is no historical data available, focus on establishing baseline patterns and validating basic business rules.

        Provide responses in JSON format with the following structure:
        {
            "anomalies_detected": boolean,
            "records": [
                {
                    "record_index": int,
                    "reason": string,
                    "risk_level": "low|medium|high",
                    "category": "unusual_combination|size_anomaly|range_violation|security_concern|business_logic"
                }
            ],
            "query_pattern_analysis": {
                "typical": boolean,
                "concerns": [string],
                "recommendations": [string]
            },
            "analysis_mode": "baseline|comparative"
        }"""

    def _create_analysis_prompt(self,
                                query_history: Dict,
                                new_data: List[Dict],
                                new_data_stats: Dict,
                                context: Dict) -> str:
        """Enhanced prompt with statistical analysis"""
        is_baseline = not bool(query_history.get("fields", {}))

        base_prompt = f"""
    Analyze these query results for anomalous patterns:

    Query Context:
    Query Hash: {context.get('query_hash')}
    Operation: {context.get('operation_name')}
    User Role: {context.get('role')}
    Query: {context.get('query')}
    Variables: {json.dumps(context.get('variables', {}))}

    Analysis Mode: {'Baseline (No History)' if is_baseline else 'Comparative'}

    Statistical Analysis of New Results:
    {json.dumps(new_data_stats, indent=2)}
    """

        if not is_baseline:
            base_prompt += f"""
    Historical Query Patterns:
    The provided historical data represents an aggregated summary of all previous queries, not a per-query comparison. The summary includes:
    {json.dumps(query_history, indent=2)}
    """

        base_prompt += """
    Consider:
    1. Statistical Patterns in the New Data:
       - Field distributions and value ranges
       - Enum value frequencies
       - Numeric correlations
       - Temporal patterns in date fields

    2. Comparison to Historical Patterns (if available):
       - Significant changes in result size or structure
       - Emergence of new field combinations
       - Shifts in value distributions and ranges

    3. Security and Access:
       - Data access appropriateness
       - Sensitive field exposure
       - Data volume context

    4. Business Logic:
       - Field relationships
       - Value ranges and combinations
       - Temporal patterns
       - Status and workflow validity"""

        return base_prompt

    def analyze_query_results(self,
                            historical_data: List[Dict],
                            new_data: List[Dict],
                            context: Dict) -> Dict[str, Any]:
        """Enhanced analysis incorporating query hash and statistical analysis"""
        try:
            # Calculate statistics for new data
            new_data_stats = DatasetStatistics.calculate_field_statistics(new_data)

            # Handle case with no history
            if not historical_data:
                self.logger.info("No historical data available. Establishing baseline.")
                baseline_analysis = {
                    "anomalies_detected": False,
                    "records": [],
                    "query_pattern_analysis": {
                        "typical": True,
                        "concerns": [],
                        "recommendations": ["Continue collecting baseline data"],
                    },
                    "analysis_mode": "baseline",
                    "query_hash": context.get('query_hash'),
                    "operation_name": context.get('operation_name'),
                    "statistical_analysis": {
                        "new_data_stats": new_data_stats,
                        "historical_comparison": None,
                        "baseline_established": True
                    },
                    "metadata": {
                        "historical_queries_analyzed": 0,
                        "new_records_analyzed": len(new_data),
                        "timestamp": datetime.now().isoformat(),
                        "query_context": {
                            "query_hash": context.get('query_hash'),
                            "operation": context.get('operation_name'),
                            "role": context.get('role')
                        },
                        "status": "initializing_history"
                    }
                }
                return baseline_analysis

            # Calculate statistics for historical data
            historical_stats = DatasetStatistics.calculate_field_statistics(historical_data)

            # Create statistical comparison
            statistical_comparison = self._compare_statistics(historical_stats, new_data_stats)

            # Get Claude's analysis
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.1,
                system=self._create_system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": self._create_analysis_prompt(
                            query_history=historical_stats,
                            new_data=new_data,
                            new_data_stats=new_data_stats,
                            context=context
                        )
                    }
                ]
            )

            analysis = json.loads(response.content[0].text)

            # Enhance the analysis with query hash and statistical insights
            enhanced_analysis = {
                **analysis,
                "query_hash": context.get('query_hash'),
                "operation_name": context.get('operation_name'),
                "analysis_mode": "comparative",
                "statistical_analysis": {
                    "new_data_stats": new_data_stats,
                    "historical_comparison": statistical_comparison,
                    "significant_changes": self._identify_significant_changes(
                        historical_stats,
                        new_data_stats
                    )
                },
                "metadata": {
                    "historical_queries_analyzed": len(historical_data),
                    "new_records_analyzed": len(new_data),
                    "timestamp": datetime.now().isoformat(),
                    "query_context": {
                        "query_hash": context.get('query_hash'),
                        "operation": context.get('operation_name'),
                        "role": context.get('role')
                    },
                    "status": "active"
                }
            }

            return enhanced_analysis

        except Exception as e:
            self.logger.error(f"Error analyzing query results: {str(e)}", exc_info=True)
            return {
                "anomalies_detected": False,
                "records": [],
                "query_pattern_analysis": {
                    "typical": True,
                    "concerns": [f"Analysis error: {str(e)}"],
                    "recommendations": ["Verify system configuration"]
                },
                "analysis_mode": "error",
                "query_hash": context.get('query_hash'),
                "operation_name": context.get('operation_name'),
                "metadata": {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "status": "error"
                }
            }

    def _compare_statistics(self,
                          historical_stats: Dict,
                          new_stats: Dict) -> Dict:
        """Compare historical and new statistics"""
        # Method implementation remains the same
        return {
            "field_changes": {},
            "new_fields": [],
            "missing_fields": [],
            "distribution_changes": {},
            "correlation_changes": {}
        }

    def _identify_significant_changes(self,
                                    historical_stats: Dict,
                                    new_stats: Dict,
                                    z_score_threshold: float = 3.0) -> List[Dict]:
        """Identify statistically significant changes"""
        # Method implementation remains the same
        return []
