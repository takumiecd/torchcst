"""Read one fixed PostgreSQL snapshot; ranking also accepts this dataset offline."""

from datetime import datetime, timezone

from benchmarks.database.model import digest


def read_dataset(database, request, *, page_size=1000):
    from .request import validate_request

    request = validate_request(request)
    records, after = [], None
    with database.snapshot():
        while True:
            page = database.list_records(
                **request["dataset"],
                kind="measure",
                include_declarations=True,
                after=after,
                limit=page_size,
            )
            for row in page:
                if row["started_at"] is not None:
                    stamp = datetime.fromisoformat(row["started_at"])
                    if stamp.utcoffset() is None:
                        raise ValueError("stored execution time needs a timezone")
                    row["started_at"] = stamp.astimezone(timezone.utc).isoformat()
                # PostgreSQL NUMERIC arrives as Decimal. The metric API and
                # exporter use finite JSON numbers, never repr/default=str.
                for metric in row["metrics"]:
                    metric["value"] = float(metric["value"])
            records.extend(page)
            if len(page) < page_size:
                break
            after = (page[-1]["projection_id"], page[-1]["ordinal"])
    body = {"schema_version": 1, "selection": request["dataset"], "records": records}
    return body | {"id": digest(body)}
