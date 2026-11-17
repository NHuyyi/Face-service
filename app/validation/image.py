"""Kiểm tra và đọc dữ liệu ảnh bằng OpenCV."""
from pathlib import Path

import numpy as np

from ..engine.image import decode_image


def validate_image(source: bytes | str | Path) -> np.ndarray:
    """Nhận bytes hoặc đường dẫn; trả ảnh BGR, báo ValueError nếu không đọc được.

    OpenCV áp dụng EXIF một lần thông qua bộ giải mã chung của dự án.
    """
    if isinstance(source, (str, Path)):
        try:
            source = Path(source).read_bytes()
        except OSError as exc:
            raise ValueError(f"Không đọc được file ảnh: {source}") from exc
    if not isinstance(source, bytes):
        raise ValueError("Đầu vào phải là bytes hoặc đường dẫn ảnh")
    return decode_image(source)
