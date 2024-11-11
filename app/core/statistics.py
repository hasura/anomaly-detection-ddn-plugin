from typing import Dict, List, Any, Union, Tuple
import numpy as np
from datetime import datetime
import pandas as pd
from collections import Counter
import logging

# Create logger for this module
logger = logging.getLogger(__name__)


class DatasetStatistics:
    @staticmethod
    def _extract_record_data(record: Dict) -> Dict:
        """Extract the actual data from a potentially nested record structure"""
        if isinstance(record, dict):
            return record.get('data', record)
        return record

    @staticmethod
    def _is_likely_enum(values: List[Any], unique_threshold: float = 0.1) -> bool:
        """
        Determine if a field is likely an enum
        - If unique values are less than 10% of total values, likely an enum
        - Excludes fields with very long strings (likely descriptions/text)
        """
        if not values:
            return False

        # Convert all values to strings for comparison
        str_values = [str(v) for v in values if v is not None]
        if not str_values:
            return False

        unique_values = set(str_values)
        # Check if number of unique values is small relative to total
        if len(unique_values) / len(str_values) > unique_threshold:
            return False

        # Check if values are short strings (typical for enums)
        avg_length = sum(len(v) for v in unique_values) / len(unique_values)
        return avg_length < 50  # Arbitrary threshold for enum-like strings

    @staticmethod
    def _analyze_string_field(field_values: List[Any], field_name: str = None) -> Tuple[str, Dict]:
        """Analyze string field statistics"""
        try:
            # Convert all values to strings first
            str_values = [str(val) if val is not None else "" for val in field_values]

            # Basic string statistics
            stats = {
                "unique_count": len(set(str_values)),
                "uniqueness_ratio": len(set(str_values)) / len(str_values) if str_values else 0,
                "avg_length": sum(len(s) for s in str_values) / len(str_values) if str_values else 0
            }

            # For categorical fields, return immediately
            if isinstance(field_name, str) and field_name.lower() in ['country', 'city', 'address', 'state', 'province']:
                stats["value_distribution"] = Counter(str_values)
                return "categorical", {"type": "categorical", "stats": stats}

            # Check if values could be numeric
            try_numeric = True
            for val in str_values:
                if val and not val.replace('.', '').replace('-', '').isdigit():
                    try_numeric = False
                    break

            if try_numeric:
                try:
                    numeric_values = [float(val) for val in str_values if val]
                    return "numeric", {
                        "type": "numeric",
                        "stats": DatasetStatistics._calculate_numeric_stats(numeric_values)
                    }
                except (ValueError, TypeError):
                    pass

            # If we get here, treat as categorical
            stats["value_distribution"] = Counter(str_values)
            return "categorical", {"type": "categorical", "stats": stats}

        except Exception as e:
            logger.error(f"Error analyzing string field: {str(e)}")
            return "error", {
                "type": "error",
                "stats": {
                    "error": str(e),
                    "unique_count": 0,
                    "uniqueness_ratio": 0,
                    "avg_length": 0
                }
            }

    @staticmethod
    def _parse_date(value: Any) -> Union[datetime, None]:
        """Attempt to parse a value as a date"""
        if isinstance(value, datetime):
            return value
        if not isinstance(value, str):
            return None

        try:
            return pd.to_datetime(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _is_date_field(values: List[Any]) -> bool:
        """Check if a field contains dates"""
        # Try to parse first 10 non-null values
        sample = [v for v in values if v is not None][:10]
        if not sample:
            return False

        valid_dates = [DatasetStatistics._parse_date(v) for v in sample]
        return len([d for d in valid_dates if d is not None]) >= len(sample) * 0.8

    @staticmethod
    def calculate_field_statistics(data: List[Dict]) -> Dict:
        """Calculate comprehensive statistics for all field types"""
        if not data:
            return {}

        try:
            # Initialize statistics dictionary
            stats = {
                "record_count": len(data),
                "fields": {},
                "correlations": {},
                "enum_correlations": {}
            }

            # First pass: extract actual data and collect values
            processed_data = []
            for record in data:
                processed_record = DatasetStatistics._extract_record_data(record)
                if isinstance(processed_record, dict):
                    processed_data.append(processed_record)

            if not processed_data:
                logger.warning("No valid records found after processing")
                return stats

            # Collect values for each field
            field_values = {}
            field_types = {}
            numeric_fields = set()
            enum_fields = set()
            enum_mappings = {}

            # Analyze fields
            for record in processed_data:
                for field, value in record.items():
                    if field not in field_values:
                        field_values[field] = []

                    field_values[field].append(value)

                    if isinstance(value, (int, float)):
                        field_types[field] = "numeric"
                        numeric_fields.add(field)

            # Second pass: analyze each field
            for field, values in field_values.items():
                non_null_values = [v for v in values if v is not None]
                if not non_null_values:
                    continue

                if field in numeric_fields:
                    stats["fields"][field] = DatasetStatistics._analyze_numeric_field(non_null_values)
                elif DatasetStatistics._is_date_field(non_null_values):
                    stats["fields"][field] = DatasetStatistics._analyze_date_field(non_null_values)
                else:
                    # Analyze string/enum fields
                    field_type, field_stats = DatasetStatistics._analyze_string_field(
                        non_null_values,
                        field  # Pass field name
                    )
                    stats["fields"][field] = field_stats

                    # Store enum mappings for correlation analysis
                    if field_type == "enum":
                        enum_fields.add(field)
                        enum_mappings[field] = field_stats.get("numeric_mapping", {})

            # Calculate correlations
            if len(numeric_fields) > 1:
                stats["correlations"] = DatasetStatistics._calculate_correlations(
                    processed_data, numeric_fields)

            # Calculate enum correlations
            if enum_fields:
                enum_correlations = DatasetStatistics._calculate_enum_correlations(
                    processed_data,
                    enum_fields,
                    enum_mappings,
                    numeric_fields
                )
                if enum_correlations:
                    stats["enum_correlations"] = enum_correlations

            return stats

        except Exception as e:
            logger.error(f"Error calculating field statistics: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "record_count": len(data),
                "fields": {}
            }

    @staticmethod
    def _analyze_numeric_field(values: List[Union[int, float]]) -> Dict:
        """Analyze numeric field"""
        try:
            values = [float(v) for v in values]  # Ensure all values are float
            return {
                "type": "numeric",
                "stats": {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                    "q1": float(np.percentile(values, 25)),
                    "q3": float(np.percentile(values, 75)),
                    "unique_values": len(set(values))
                },
                "distribution": {
                    "histogram": np.histogram(values, bins='auto')[0].tolist(),
                    "bin_edges": np.histogram(values, bins='auto')[1].tolist()
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing numeric field: {str(e)}")
            return {
                "type": "numeric",
                "error": str(e)
            }

    @staticmethod
    def _analyze_date_field(values: List[Any]) -> Dict:
        """Analyze date field"""
        dates = [DatasetStatistics._parse_date(v) for v in values]
        valid_dates = [d for d in dates if d is not None]

        if not valid_dates:
            return {"type": "date", "error": "No valid dates found"}

        try:
            # Convert to epoch timestamps for numeric analysis
            timestamps = [d.timestamp() for d in valid_dates]

            # Extract components
            years = [d.year for d in valid_dates]
            months = [d.month for d in valid_dates]
            days = [d.day for d in valid_dates]
            hours = [d.hour for d in valid_dates]
            weekdays = [d.weekday() for d in valid_dates]

            return {
                "type": "date",
                "stats": {
                    "min_date": min(valid_dates).isoformat(),
                    "max_date": max(valid_dates).isoformat(),
                    "unique_dates": len(set(valid_dates))
                },
                "numeric_stats": DatasetStatistics._analyze_numeric_field(timestamps),
                "components": {
                    "years": DatasetStatistics._analyze_numeric_field(years),
                    "months": {
                        "distribution": dict(Counter(months)),
                        "most_common": Counter(months).most_common(3)
                    },
                    "days": {
                        "distribution": dict(Counter(days)),
                        "patterns": {
                            "weekday_weekend": {
                                "weekday": sum(1 for d in valid_dates if d.weekday() < 5),
                                "weekend": sum(1 for d in valid_dates if d.weekday() >= 5)
                            }
                        }
                    },
                    "hours": {
                        "distribution": dict(Counter(hours)),
                        "patterns": {
                            "business_hours": sum(1 for h in hours if 9 <= h <= 17),
                            "off_hours": sum(1 for h in hours if h < 9 or h > 17)
                        }
                    },
                    "weekdays": {
                        "distribution": dict(Counter(weekdays)),
                        "most_common": Counter(weekdays).most_common()
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing date field: {str(e)}")
            return {
                "type": "date",
                "error": str(e)
            }

    @staticmethod
    def _calculate_entropy(counter: Counter) -> float:
        """Calculate Shannon entropy of distribution"""
        try:
            total = sum(counter.values())
            probabilities = [count / total for count in counter.values()]
            return -sum(p * np.log2(p) for p in probabilities if p > 0)
        except Exception as e:
            logger.error(f"Error calculating entropy: {str(e)}")
            return 0.0

    @staticmethod
    def _calculate_correlations(data: List[Dict], numeric_fields: set) -> Dict:
        """Calculate correlations between numeric fields"""
        try:
            correlations = {}
            numeric_data = {field: [] for field in numeric_fields}

            # Collect numeric values
            for record in data:
                for field in numeric_fields:
                    numeric_data[field].append(record.get(field))

            # Calculate correlations
            fields = list(numeric_fields)
            for i, field1 in enumerate(fields):
                correlations[field1] = {}
                for field2 in fields[i + 1:]:
                    valid_values1 = [v for v in numeric_data[field1] if v is not None]
                    valid_values2 = [v for v in numeric_data[field2] if v is not None]

                    if len(valid_values1) > 1 and len(valid_values2) > 1:
                        correlation = np.corrcoef(valid_values1, valid_values2)[0, 1]
                        if not np.isnan(correlation):
                            correlations[field1][field2] = round(correlation, 3)

            return correlations
        except Exception as e:
            logger.error(f"Error calculating correlations: {str(e)}")
            return {}

    @staticmethod
    def _calculate_enum_correlations(
            data: List[Dict],
            enum_fields: set,
            enum_mappings: Dict,
            numeric_fields: set
    ) -> Dict:
        """Calculate correlations involving enum fields"""
        try:
            correlations = {}

            # Prepare numeric values for enum fields
            enum_numeric_data = {
                field: [enum_mappings[field].get(str(record.get(field)))
                        for record in data]
                for field in enum_fields
            }

            # Add regular numeric fields
            numeric_data = {
                field: [record.get(field) for record in data]
                for field in numeric_fields
            }

            all_fields = list(enum_fields) + list(numeric_fields)

            # Calculate correlations
            for i, field1 in enumerate(all_fields):
                values1 = enum_numeric_data.get(field1) or numeric_data.get(field1)
                if not values1:
                    continue

                correlations[field1] = {}
                for field2 in all_fields[i + 1:]:
                    values2 = enum_numeric_data.get(field2) or numeric_data.get(field2)
                    if not values2:
                        continue

                    # Only calculate if we have enough non-null values
                    valid_pairs = [
                        (v1, v2) for v1, v2 in zip(values1, values2)
                        if v1 is not None and v2 is not None
                    ]

                    if len(valid_pairs) > 10:  # Minimum sample size
                        v1, v2 = zip(*valid_pairs)
                        correlation = np.corrcoef(v1, v2)[0, 1]
                        if not np.isnan(correlation):
                            correlations[field1][field2] = round(correlation, 3)

            return correlations
        except Exception as e:
            logger.error(f"Error calculating enum correlations: {str(e)}")
            return {}
