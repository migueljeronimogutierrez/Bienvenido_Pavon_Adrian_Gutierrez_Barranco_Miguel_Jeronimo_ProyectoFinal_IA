from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED


def build_results_zip(file_map: dict[str, bytes]) -> bytes:
    buffer = BytesIO()

    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as zf:
        for filename, content in file_map.items():
            zf.writestr(filename, content)

    return buffer.getvalue()
