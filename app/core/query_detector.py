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
       
    4. Obfuscation:
       - Operation name is meaningful
       - Operation name is an English language word or phrase
       - Operation name indicates business intent of transaction details

    5. Business Logic:
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
        """Compare historical and new statistics to identify changes in data patterns.

        Args:
            historical_stats: Statistics from historical query data
            new_stats: Statistics from new query data

        Returns:
            Dict containing comparison results including field changes, new/missing fields,
            distribution changes, and correlation changes
        """
        comparison = {
            "field_changes": {},
            "new_fields": [],
            "missing_fields": [],
            "distribution_changes": {},
            "correlation_changes": {}
        }

        # Check for new and missing fields
        historical_fields = set(historical_stats.get("fields", {}).keys())
        new_fields = set(new_stats.get("fields", {}).keys())

        comparison["new_fields"] = list(new_fields - historical_fields)
        comparison["missing_fields"] = list(historical_fields - new_fields)

        # Compare common fields
        common_fields = historical_fields.intersection(new_fields)
        for field in common_fields:
            hist_field = historical_stats["fields"].get(field, {})
            new_field = new_stats["fields"].get(field, {})

            # Initialize field comparison
            comparison["field_changes"][field] = {
                "type_changed": False,
                "value_range_changed": False,
                "cardinality_changed": False,
                "null_ratio_changed": False,
                "distribution_changed": False,
                "changes": []
            }

            # Check type changes
            hist_type = hist_field.get("type")
            new_type = new_field.get("type")
            if hist_type != new_type:
                comparison["field_changes"][field]["type_changed"] = True
                comparison["field_changes"][field]["changes"].append(
                    f"Field type changed from {hist_type} to {new_type}"
                )

            # Check numerical fields
            if hist_type == "numeric" and new_type == "numeric":
                hist_stats = hist_field.get("stats", {})
                new_stats_field = new_field.get("stats", {})

                # Compare value ranges
                hist_min = hist_stats.get("min", 0)
                hist_max = hist_stats.get("max", 0)
                new_min = new_stats_field.get("min", 0)
                new_max = new_stats_field.get("max", 0)

                # Check for range expansion
                min_change_pct = ((new_min - hist_min) / max(abs(hist_min), 1)) * 100 if hist_min != 0 else 0
                max_change_pct = ((new_max - hist_max) / max(abs(hist_max), 1)) * 100 if hist_max != 0 else 0

                if new_min < hist_min and abs(min_change_pct) > 10:
                    comparison["field_changes"][field]["value_range_changed"] = True
                    comparison["field_changes"][field]["changes"].append(
                        f"Minimum value decreased from {hist_min} to {new_min} ({min_change_pct:.2f}%)"
                    )

                if new_max > hist_max and abs(max_change_pct) > 10:
                    comparison["field_changes"][field]["value_range_changed"] = True
                    comparison["field_changes"][field]["changes"].append(
                        f"Maximum value increased from {hist_max} to {new_max} ({max_change_pct:.2f}%)"
                    )

                # Compare means and standard deviations
                hist_mean = hist_stats.get("mean", 0)
                new_mean = new_stats_field.get("mean", 0)
                hist_std = hist_stats.get("std", 1)
                new_std = new_stats_field.get("std", 1)

                # Check for significant mean change (> 1 std dev)
                mean_change = abs(new_mean - hist_mean)
                if mean_change > hist_std:
                    comparison["field_changes"][field]["distribution_changed"] = True
                    comparison["field_changes"][field]["changes"].append(
                        f"Mean changed significantly from {hist_mean:.2f} to {new_mean:.2f} (change: {mean_change:.2f}, threshold: {hist_std:.2f})"
                    )

                    # Add to distribution changes section
                    comparison["distribution_changes"][field] = {
                        "mean_change": mean_change,
                        "std_change": abs(new_std - hist_std),
                        "z_score": mean_change / max(hist_std, 0.001),
                        "old_stats": {
                            "mean": hist_mean,
                            "std": hist_std,
                            "median": hist_stats.get("median", 0),
                            "q1": hist_stats.get("q1", 0),
                            "q3": hist_stats.get("q3", 0)
                        },
                        "new_stats": {
                            "mean": new_mean,
                            "std": new_std,
                            "median": new_stats_field.get("median", 0),
                            "q1": new_stats_field.get("q1", 0),
                            "q3": new_stats_field.get("q3", 0)
                        }
                    }

                # Compare histogram distributions if available
                if "distribution" in hist_field and "distribution" in new_field:
                    hist_dist = hist_field.get("distribution", {})
                    new_dist = new_field.get("distribution", {})

                    if "histogram" in hist_dist and "histogram" in new_dist:
                        # Simplified histogram comparison (just check if bins changed significantly)
                        hist_hist = hist_dist.get("histogram", [])
                        new_hist = new_dist.get("histogram", [])

                        if len(hist_hist) > 0 and len(new_hist) > 0 and len(hist_hist) != len(new_hist):
                            comparison["field_changes"][field]["distribution_changed"] = True
                            comparison["field_changes"][field]["changes"].append(
                                f"Histogram distribution changed (old bins: {len(hist_hist)}, new bins: {len(new_hist)})"
                            )

            # Check categorical or string fields
            elif hist_type in ["categorical", "date", "string"] and new_type in ["categorical", "date", "string"]:
                # For categorical fields, check stats and value distributions
                if "stats" in hist_field and "stats" in new_field:
                    # Compare unique value counts
                    hist_stats = hist_field.get("stats", {})
                    new_stats_field = new_field.get("stats", {})

                    hist_unique = hist_stats.get("unique_count", 0)
                    new_unique = new_stats_field.get("unique_count", 0)

                    if hist_unique > 0:
                        cardinality_change_pct = ((new_unique - hist_unique) / hist_unique) * 100

                        if abs(cardinality_change_pct) > 20:  # 20% threshold for cardinality change
                            comparison["field_changes"][field]["cardinality_changed"] = True
                            comparison["field_changes"][field]["changes"].append(
                                f"Unique value count changed from {hist_unique} to {new_unique} ({cardinality_change_pct:.2f}%)"
                            )

                    # Compare value distributions for categorical fields
                    if "value_distribution" in hist_stats and "value_distribution" in new_stats_field:
                        hist_dist = hist_stats.get("value_distribution", {})
                        new_dist = new_stats_field.get("value_distribution", {})

                        # Find new top values
                        hist_values = set(hist_dist.keys())
                        new_values = set(new_dist.keys())

                        new_values_list = list(new_values - hist_values)
                        if new_values_list:
                            comparison["distribution_changes"].setdefault(field, {})
                            comparison["distribution_changes"][field]["new_values"] = new_values_list
                            comparison["field_changes"][field]["distribution_changed"] = True
                            comparison["field_changes"][field]["changes"].append(
                                f"New unique values appeared: {new_values_list[:5]}" +
                                (f" and {len(new_values_list) - 5} more" if len(new_values_list) > 5 else "")
                            )

                        # Check for significant distribution shifts in common values
                        common_values = hist_values.intersection(new_values)
                        significant_shifts = []

                        for value in common_values:
                            hist_count = hist_dist.get(value, 0)
                            new_count = new_dist.get(value, 0)

                            hist_freq = hist_count / sum(hist_dist.values()) if sum(hist_dist.values()) > 0 else 0
                            new_freq = new_count / sum(new_dist.values()) if sum(new_dist.values()) > 0 else 0

                            # Check for significant frequency change (>20%)
                            if abs(new_freq - hist_freq) > 0.2:
                                significant_shifts.append({
                                    "value": value,
                                    "old_freq": hist_freq,
                                    "new_freq": new_freq,
                                    "change": new_freq - hist_freq
                                })

                        if significant_shifts:
                            comparison["field_changes"][field]["distribution_changed"] = True
                            comparison["field_changes"][field]["changes"].append(
                                f"Significant shifts in {len(significant_shifts)} value frequencies"
                            )
                            comparison["distribution_changes"].setdefault(field, {})
                            comparison["distribution_changes"][field]["value_shifts"] = significant_shifts

            # Check date fields specifically
            elif hist_type == "date" and new_type == "date":
                # Compare date ranges
                hist_stats = hist_field.get("stats", {})
                new_stats_field = new_field.get("stats", {})

                hist_min = hist_stats.get("min_date")
                hist_max = hist_stats.get("max_date")
                new_min = new_stats_field.get("min_date")
                new_max = new_stats_field.get("max_date")

                # Check for date range changes
                if hist_min != new_min or hist_max != new_max:
                    comparison["field_changes"][field]["value_range_changed"] = True
                    comparison["field_changes"][field]["changes"].append(
                        f"Date range changed from {hist_min} - {hist_max} to {new_min} - {new_max}"
                    )

                # Compare date components distributions
                if "components" in hist_field and "components" in new_field:
                    hist_components = hist_field.get("components", {})
                    new_components = new_field.get("components", {})

                    # Check for weekday/weekend pattern changes
                    if "days" in hist_components and "days" in new_components:
                        hist_patterns = hist_components["days"].get("patterns", {}).get("weekday_weekend", {})
                        new_patterns = new_components["days"].get("patterns", {}).get("weekday_weekend", {})

                        if hist_patterns and new_patterns:
                            hist_total = hist_patterns.get("weekday", 0) + hist_patterns.get("weekend", 0)
                            new_total = new_patterns.get("weekday", 0) + new_patterns.get("weekend", 0)

                            if hist_total > 0 and new_total > 0:
                                hist_weekend_ratio = hist_patterns.get("weekend", 0) / hist_total
                                new_weekend_ratio = new_patterns.get("weekend", 0) / new_total

                                if abs(new_weekend_ratio - hist_weekend_ratio) > 0.2:  # 20% shift
                                    comparison["field_changes"][field]["distribution_changed"] = True
                                    comparison["field_changes"][field]["changes"].append(
                                        f"Weekend/weekday pattern changed significantly (weekend ratio: {hist_weekend_ratio:.2f} → {new_weekend_ratio:.2f})"
                                    )

        # Compare correlations between fields
        if "correlations" in historical_stats and "correlations" in new_stats:
            hist_corr = historical_stats["correlations"]
            new_corr = new_stats["correlations"]

            # Process nested correlation structure
            for field1, field1_corrs in hist_corr.items():
                if field1 in new_corr:
                    for field2, hist_value in field1_corrs.items():
                        if field2 in new_corr.get(field1, {}):
                            new_value = new_corr[field1][field2]
                            corr_change = abs(new_value - hist_value)

                            # Report significant correlation changes (>0.3)
                            if corr_change > 0.3:
                                field_pair = f"{field1},{field2}"
                                comparison["correlation_changes"][field_pair] = {
                                    "fields": [field1, field2],
                                    "old_value": hist_value,
                                    "new_value": new_value,
                                    "change": corr_change,
                                    "interpretation": "strengthened" if abs(new_value) > abs(hist_value) else "weakened"
                                }

        # Add enum correlations if available
        if "enum_correlations" in historical_stats and "enum_correlations" in new_stats:
            hist_enum_corr = historical_stats["enum_correlations"]
            new_enum_corr = new_stats["enum_correlations"]

            # Similar process as regular correlations
            for field1, field1_corrs in hist_enum_corr.items():
                if field1 in new_enum_corr:
                    for field2, hist_value in field1_corrs.items():
                        if field2 in new_enum_corr.get(field1, {}):
                            new_value = new_enum_corr[field1][field2]
                            corr_change = abs(new_value - hist_value)

                            if corr_change > 0.3:
                                field_pair = f"enum:{field1},{field2}"
                                comparison["correlation_changes"][field_pair] = {
                                    "fields": [field1, field2],
                                    "type": "enum",
                                    "old_value": hist_value,
                                    "new_value": new_value,
                                    "change": corr_change
                                }

        return comparison

    def _identify_significant_changes(self,
                                      historical_stats: Dict,
                                      new_stats: Dict,
                                      z_score_threshold: float = 3.0) -> List[Dict]:
        """Identify statistically significant changes between historical and new data.

        Args:
            historical_stats: Statistics from historical query data
            new_stats: Statistics from new query data
            z_score_threshold: Z-score threshold to consider a change significant (default: 3.0)

        Returns:
            List of dictionaries containing significant changes with field, type, and severity details
        """
        significant_changes = []

        # Helper function to calculate z-score for numeric values
        def calculate_z_score(historical_value, new_value, historical_std):
            if historical_std == 0:
                return 0  # Avoid division by zero
            return abs(new_value - historical_value) / historical_std

        # Check for significant numerical field changes
        hist_fields = historical_stats.get("fields", {})
        new_fields = new_stats.get("fields", {})

        for field, hist_field in hist_fields.items():
            # Skip if field doesn't exist in new data
            if field not in new_fields:
                continue

            new_field = new_fields[field]

            # Analyze numeric fields
            if hist_field.get("type") == "numeric":
                hist_stats = hist_field.get("stats", {})
                new_stats_field = new_field.get("stats", {})

                # Check for significant mean changes
                if "mean" in hist_stats and "mean" in new_stats_field and "std" in hist_stats:
                    hist_mean = hist_stats["mean"]
                    new_mean = new_stats_field["mean"]
                    hist_std = hist_stats["std"]

                    z_score = calculate_z_score(hist_mean, new_mean, max(hist_std, 0.001))

                    if z_score > z_score_threshold:
                        significant_changes.append({
                            "field": field,
                            "type": "numeric_mean",
                            "z_score": z_score,
                            "old_value": hist_mean,
                            "new_value": new_mean,
                            "severity": "high" if z_score > 2 * z_score_threshold else "medium"
                        })

                # Check for significant variance changes
                if "std" in hist_stats and "std" in new_stats_field:
                    hist_std = hist_stats["std"]
                    new_std = new_stats_field["std"]

                    if hist_std > 0:
                        variance_ratio = new_std / max(hist_std, 0.001)

                        # A variance ratio > 2 or < 0.5 is considered significant
                        if variance_ratio > 2 or variance_ratio < 0.5:
                            significant_changes.append({
                                "field": field,
                                "type": "numeric_variance",
                                "ratio": variance_ratio,
                                "old_std": hist_std,
                                "new_std": new_std,
                                "severity": "high" if (variance_ratio > 3 or variance_ratio < 0.3) else "medium"
                            })

                # Check for range expansions
                if "min" in hist_stats and "min" in new_stats_field and "max" in hist_stats and "max" in new_stats_field:
                    hist_min = hist_stats["min"]
                    hist_max = hist_stats["max"]
                    new_min = new_stats_field["min"]
                    new_max = new_stats_field["max"]

                    hist_range = hist_max - hist_min
                    new_range = new_max - new_min

                    # Consider expansions beyond historical range
                    if hist_range > 0:
                        range_ratio = new_range / hist_range

                        if range_ratio > 1.5:  # 50% expansion in range
                            significant_changes.append({
                                "field": field,
                                "type": "numeric_range",
                                "ratio": range_ratio,
                                "old_range": [hist_min, hist_max],
                                "new_range": [new_min, new_max],
                                "severity": "high" if range_ratio > 2 else "medium"
                            })

                    # Check for new outliers
                    if new_min < hist_min or new_max > hist_max:
                        # Calculate how far outside historical range
                        outside_factor = max(
                            (hist_min - new_min) / max(hist_range, 0.001) if new_min < hist_min else 0,
                            (new_max - hist_max) / max(hist_range, 0.001) if new_max > hist_max else 0
                        )

                        if outside_factor > 0.2:  # 20% beyond historical range
                            significant_changes.append({
                                "field": field,
                                "type": "numeric_outlier",
                                "factor": outside_factor,
                                "historical_range": [hist_min, hist_max],
                                "new_range": [new_min, new_max],
                                "severity": "high" if outside_factor > 0.5 else "medium"
                            })

                # Check for distribution changes
                if "distribution" in hist_field and "distribution" in new_field:
                    hist_dist = hist_field["distribution"]
                    new_dist = new_field["distribution"]

                    # Compare histograms if available
                    if "histogram" in hist_dist and "histogram" in new_dist and "bin_edges" in hist_dist and "bin_edges" in new_dist:
                        hist_hist = hist_dist["histogram"]
                        new_hist = new_dist["histogram"]
                        hist_edges = hist_dist["bin_edges"]
                        new_edges = new_dist["bin_edges"]

                        # Simplistic comparison of histograms - check for major shifts in bin proportions
                        if len(hist_hist) > 0 and len(new_hist) > 0:
                            # Normalize histograms to proportions
                            hist_props = [h / sum(hist_hist) for h in hist_hist]
                            new_props = [h / sum(new_hist) for h in new_hist]

                            # If bin counts differ, that's already significant
                            if len(hist_props) != len(new_props):
                                significant_changes.append({
                                    "field": field,
                                    "type": "distribution_bin_count",
                                    "old_bins": len(hist_props),
                                    "new_bins": len(new_props),
                                    "severity": "medium"
                                })
                            else:
                                # Calculate sum of absolute differences in proportions
                                total_shift = sum(abs(h - n) for h, n in zip(hist_props, new_props))

                                if total_shift > 0.4:  # 40% shift in overall distribution
                                    significant_changes.append({
                                        "field": field,
                                        "type": "distribution_shift",
                                        "shift_magnitude": total_shift,
                                        "severity": "high" if total_shift > 0.6 else "medium"
                                    })

            # Analyze categorical fields
            elif hist_field.get("type") in ["categorical", "string"]:
                hist_stats = hist_field.get("stats", {})
                new_stats_field = new_field.get("stats", {})

                # Check for cardinality changes
                if "unique_count" in hist_stats and "unique_count" in new_stats_field:
                    hist_unique = hist_stats["unique_count"]
                    new_unique = new_stats_field["unique_count"]

                    if hist_unique > 0:
                        cardinality_ratio = new_unique / hist_unique

                        # Significant increase in unique values
                        if cardinality_ratio > 1.3:  # 30% increase
                            significant_changes.append({
                                "field": field,
                                "type": "categorical_cardinality",
                                "ratio": cardinality_ratio,
                                "old_unique": hist_unique,
                                "new_unique": new_unique,
                                "severity": "high" if cardinality_ratio > 1.5 else "medium"
                            })

                # Check for distribution shifts in categorical values
                if "value_distribution" in hist_stats and "value_distribution" in new_stats_field:
                    hist_dist = hist_stats["value_distribution"]
                    new_dist = new_stats_field["value_distribution"]

                    # Calculate divergence in distributions
                    total_divergence = 0
                    common_values = set(hist_dist.keys()).intersection(set(new_dist.keys()))

                    if common_values:
                        # Normalize counts to get probabilities
                        hist_total = sum(hist_dist.values())
                        new_total = sum(new_dist.values())

                        for value in common_values:
                            hist_prob = hist_dist[value] / hist_total
                            new_prob = new_dist[value] / new_total

                            # Calculate probability shift
                            total_divergence += abs(new_prob - hist_prob)

                        # Normalize by number of common values
                        if len(common_values) > 0:
                            avg_divergence = total_divergence / len(common_values)

                            # If average divergence is significant
                            if avg_divergence > 0.1:  # 10% average shift per category
                                significant_changes.append({
                                    "field": field,
                                    "type": "categorical_distribution",
                                    "avg_divergence": avg_divergence,
                                    "severity": "high" if avg_divergence > 0.2 else "medium"
                                })

                    # Check for new dominant values
                    hist_dominant = sorted(hist_dist.items(), key=lambda x: x[1], reverse=True)[:3]
                    new_dominant = sorted(new_dist.items(), key=lambda x: x[1], reverse=True)[:3]

                    hist_dom_values = [item[0] for item in hist_dominant]
                    new_dom_values = [item[0] for item in new_dominant]

                    # If top values have changed significantly
                    if len(set(hist_dom_values) - set(new_dom_values)) >= 2:
                        significant_changes.append({
                            "field": field,
                            "type": "categorical_dominant",
                            "old_top": hist_dom_values,
                            "new_top": new_dom_values,
                            "severity": "medium"
                        })

            # Analyze date fields
            elif hist_field.get("type") == "date":
                hist_stats = hist_field.get("stats", {})
                new_stats_field = new_field.get("stats", {})

                # Check for date range expansions
                if "min_date" in hist_stats and "min_date" in new_stats_field and "max_date" in hist_stats and "max_date" in new_stats_field:
                    # Note: We can't directly subtract dates as strings,
                    # but we would typically check if the new data extends the date range significantly

                    # For date fields, temporal shifts are particularly important
                    if "components" in hist_field and "components" in new_field:
                        # Check for shifts in temporal patterns
                        hist_components = hist_field.get("components", {})
                        new_components = new_field.get("components", {})

                        # Check weekday/weekend patterns
                        if "days" in hist_components and "days" in new_components:
                            if "patterns" in hist_components["days"] and "patterns" in new_components["days"]:
                                hist_patterns = hist_components["days"]["patterns"].get("weekday_weekend", {})
                                new_patterns = new_components["days"]["patterns"].get("weekday_weekend", {})

                                if hist_patterns and new_patterns:
                                    hist_weekday = hist_patterns.get("weekday", 0)
                                    hist_weekend = hist_patterns.get("weekend", 0)
                                    new_weekday = new_patterns.get("weekday", 0)
                                    new_weekend = new_patterns.get("weekend", 0)

                                    # Calculate weekday/weekend ratios
                                    hist_ratio = hist_weekend / max(hist_weekday, 1)
                                    new_ratio = new_weekend / max(new_weekday, 1)

                                    ratio_change = abs(new_ratio - hist_ratio)

                                    if ratio_change > 0.3:  # 30% change in weekday/weekend pattern
                                        significant_changes.append({
                                            "field": field,
                                            "type": "date_weekday_pattern",
                                            "old_ratio": hist_ratio,
                                            "new_ratio": new_ratio,
                                            "change": ratio_change,
                                            "severity": "high" if ratio_change > 0.5 else "medium"
                                        })

            # Check for null ratio changes across all field types
            if "null_count" in hist_stats and "null_count" in new_stats_field:
                hist_total = hist_stats.get("count", 1)
                new_total = new_stats_field.get("count", 1)

                if hist_total > 0 and new_total > 0:
                    hist_null_ratio = hist_stats["null_count"] / hist_total
                    new_null_ratio = new_stats_field["null_count"] / new_total

                    null_ratio_change = abs(new_null_ratio - hist_null_ratio)

                    # Significant change in null ratios
                    if null_ratio_change > 0.1:  # 10% change in nulls
                        significant_changes.append({
                            "field": field,
                            "type": "null_ratio",
                            "old_ratio": hist_null_ratio,
                            "new_ratio": new_null_ratio,
                            "change": null_ratio_change,
                            "severity": "high" if null_ratio_change > 0.2 else "medium"
                        })

        # Check for correlation changes
        if "correlations" in historical_stats and "correlations" in new_stats:
            hist_corr = historical_stats["correlations"]
            new_corr = new_stats["correlations"]

            # Process nested correlation structure
            for field1, field1_corrs in hist_corr.items():
                if field1 in new_corr:
                    for field2, hist_value in field1_corrs.items():
                        if field2 in new_corr.get(field1, {}):
                            new_value = new_corr[field1][field2]
                            corr_change = abs(new_value - hist_value)

                            # Significant correlation change
                            if corr_change > 0.3:
                                significant_changes.append({
                                    "field_pair": [field1, field2],
                                    "type": "correlation",
                                    "old_value": hist_value,
                                    "new_value": new_value,
                                    "change": corr_change,
                                    "direction": "strengthened" if abs(new_value) > abs(hist_value) else "weakened",
                                    "severity": "high" if corr_change > 0.5 else "medium"
                                })

        # Check for enum correlation changes if available
        if "enum_correlations" in historical_stats and "enum_correlations" in new_stats:
            hist_enum_corr = historical_stats["enum_correlations"]
            new_enum_corr = new_stats["enum_correlations"]

            # Similar process as regular correlations
            for field1, field1_corrs in hist_enum_corr.items():
                if field1 in new_enum_corr:
                    for field2, hist_value in field1_corrs.items():
                        if field2 in new_enum_corr.get(field1, {}):
                            new_value = new_enum_corr[field1][field2]
                            corr_change = abs(new_value - hist_value)

                            if corr_change > 0.3:
                                significant_changes.append({
                                    "field_pair": [field1, field2],
                                    "type": "enum_correlation",
                                    "old_value": hist_value,
                                    "new_value": new_value,
                                    "change": corr_change,
                                    "severity": "high" if corr_change > 0.5 else "medium"
                                })

        return significant_changes
