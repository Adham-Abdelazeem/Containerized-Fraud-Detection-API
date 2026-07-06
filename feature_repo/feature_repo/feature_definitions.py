from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, FeatureService
from feast.types import Float64, Int64

# ── Entity ──────────────────────────────────────────────────
# The primary key. Every transaction has a unique ID.
transaction = Entity(
    name="transaction",
    join_keys=["transaction_id"]
)

# ── Data Source ─────────────────────────────────────────────
# Where the raw feature data lives (your prepared Parquet)
transaction_source = FileSource(
    name="transaction_source",
    path="data/transactions.parquet",
    timestamp_field="event_timestamp",
)

# ── Feature View ────────────────────────────────────────────
# The schema of your features + link to the source
transaction_fv = FeatureView(
    name="transaction_features",
    entities=[transaction],
    ttl=timedelta(days=7),       # fraud features don't go stale fast
    schema=[
        Field(name="V1",     dtype=Float64),
        Field(name="V2",     dtype=Float64),
        Field(name="V3",     dtype=Float64),
        Field(name="V4",     dtype=Float64),
        Field(name="V5",     dtype=Float64),
        Field(name="V6",     dtype=Float64),
        Field(name="V7",     dtype=Float64),
        Field(name="V8",     dtype=Float64),
        Field(name="V9",     dtype=Float64),
        Field(name="V10",    dtype=Float64),
        Field(name="V11",    dtype=Float64),
        Field(name="V12",    dtype=Float64),
        Field(name="V13",    dtype=Float64),
        Field(name="V14",    dtype=Float64),
        Field(name="V15",    dtype=Float64),
        Field(name="V16",    dtype=Float64),
        Field(name="V17",    dtype=Float64),
        Field(name="V18",    dtype=Float64),
        Field(name="V19",    dtype=Float64),
        Field(name="V20",    dtype=Float64),
        Field(name="V21",    dtype=Float64),
        Field(name="V22",    dtype=Float64),
        Field(name="V23",    dtype=Float64),
        Field(name="V24",    dtype=Float64),
        Field(name="V25",    dtype=Float64),
        Field(name="V26",    dtype=Float64),
        Field(name="V27",     dtype=Float64),
        Field(name="V28",    dtype=Float64),
        Field(name="Amount", dtype=Float64),
    ],
    online=True,
    source=transaction_source,
)

# ── Feature Service ─────────────────────────────────────────
# The exact feature bundle your fraud model needs
fraud_model_v1 = FeatureService(
    name="fraud_model_v1",
    features=[transaction_fv],   # all features from the view
)