import enum


class ImportJobStatus(str, enum.Enum):
    DOWNLOADING = "DOWNLOADING"
    REVIEW = "REVIEW"
