"""
Feast feature definitions for fraud detection.

This file defines:
- Entities: The keys we use to look up features (merchant_category)
- Data Sources: Where the raw feature data comes from (Parquet file)
- Feature Views: The features themselves and their schemas

The key insight: These definitions are the SINGLE SOURCE OF TRUTH.
Both training and serving use these exact definitions.
"""
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64

# =============================================================================
# ENTITIES
# =============================================================================
# An entity is the "key" we use to look up features.
# For merchant-level features, the entity is merchant_category.

merchant = Entity(
    name="merchant_category",
    description="Merchant category for the transaction (for example, 'online', 'grocery')",
    value_type=ValueType.STRING,
)

# =============================================================================
# DATA SOURCES
# =============================================================================
# Data sources tell Feast where to find the raw feature data.
# For local development, we use a Parquet file.
# For production, this could be BigQuery, Snowflake, S3, etc.

merchant_stats_source = FileSource(
    name="merchant_stats_source",
    path="data/merchant_features.parquet",  # We'll create this file
    timestamp_field="event_timestamp",       # Required for point-in-time joins
)

# =============================================================================
# FEATURE VIEWS
# =============================================================================
# A Feature View defines a group of related features.
# It specifies:
# - Which entity the features are for
# - The schema (names and types of features)
# - Where the data comes from
# - How long features are valid (TTL)

merchant_stats_fv = FeatureView(
    name="merchant_stats",
    description="Aggregated statistics per merchant category",
    entities=[merchant],
    ttl=timedelta(days=7),  # Features are valid for 7 days
    schema=[
        Field(name="avg_amount", dtype=Float32, description="Average transaction amount"),
        Field(name="transaction_count", dtype=Int64, description="Number of transactions"),
        Field(name="fraud_rate", dtype=Float32, description="Historical fraud rate"),
    ],
    source=merchant_stats_source,
    online=True,  # Enable online serving (low-latency retrieval)
)