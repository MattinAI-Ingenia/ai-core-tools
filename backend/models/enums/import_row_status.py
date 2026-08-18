import enum


class ImportRowStatus(str, enum.Enum):
    PENDING = "PENDING"
    DOWNLOADING = "DOWNLOADING"
    DOWNLOADED = "DOWNLOADED"
    FAILED = "FAILED"
    CONFIRMED = "CONFIRMED"
    DISCARDED = "DISCARDED"
