from feast import Entity
from feast import FeatureView
from feast import FileSource
from feast import Field

from feast.types import Float32

from datetime import timedelta


iris_source = FileSource(
    path="../data/iris_data_adapted_for_feast.parquet                               ",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)


iris = Entity(
    name="iris_id",
    join_keys=["iris_id"],
)


iris_feature_view = FeatureView(
    name="iris_features",
    entities=[iris],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
    ],
    source=iris_source,
)