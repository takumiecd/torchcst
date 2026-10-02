"""Remove cached OAuth/runtime values before persisting CLI diagnostics."""

import json
import re


def private_values(paths):
    values = set()

    def visit(value):
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str) and len(value) >= 8:
            values.add(value)

    for path in paths:
        if path.is_file():
            visit(json.loads(path.read_bytes()))
    return values


def redact(content, values):
    for value in sorted(values, key=len, reverse=True):
        content = content.replace(value, "[redacted]")
    content = re.sub(r"(?i)(bearer\s+)[^\s\"']+", r"\1[redacted]", content)
    return re.sub(r"(?i)([?&](?:token|key|auth)=)[^\s&#\"']+", r"\1[redacted]", content)
