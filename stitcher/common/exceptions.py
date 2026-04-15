import os
import numpy as np


class StitchingError(Exception):
    def __init__(self, message, error_code=1, details=None):
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)


class ImageLoadError(StitchingError):
    def __init__(self, message, file_path=None):
        details = {'file_path': file_path} if file_path else {}
        super().__init__(message, error_code=101, details=details)


class AlignmentError(StitchingError):
    def __init__(self, message, stage=None):
        details = {'stage': stage} if stage else {}
        super().__init__(message, error_code=103, details=details)
