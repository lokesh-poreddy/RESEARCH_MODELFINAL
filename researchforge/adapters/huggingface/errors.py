"""Error types for the Hugging Face adapter layer.

All errors are subclasses of HFProviderError and follow the error codes
defined in the adapter specification.

Secret safety:
    No error message should contain credential values.
    The adapter layer must sanitize any API responses before including
    them in error messages.
"""
from __future__ import annotations


class HFProviderError(Exception):
    """Base class for all Hugging Face provider errors."""
    error_code: str = "HF_PROVIDER_ERROR"

    def __init__(self, message: str, *, model_id: str = "", task: str = "") -> None:
        super().__init__(message)
        self.model_id = model_id
        self.task = task

    def __str__(self) -> str:
        base = super().__str__()
        parts = []
        if self.model_id:
            parts.append(f"model={self.model_id!r}")
        if self.task:
            parts.append(f"task={self.task!r}")
        if parts:
            return f"{base} [{', '.join(parts)}]"
        return base


class HFTokenMissingError(HFProviderError):
    """HF_TOKEN environment variable is not set."""
    error_code = "HF_TOKEN_MISSING"

    def __init__(self) -> None:
        super().__init__(
            "HF_TOKEN environment variable is not set. "
            "Set HF_TOKEN in your environment (e.g. .env file, never in source). "
            "See .env.example for configuration reference."
        )


class HFAuthError(HFProviderError):
    """Authentication with Hugging Face API failed."""
    error_code = "HF_AUTH_FAILED"


class HFModelNotFoundError(HFProviderError):
    """Requested model was not found on Hugging Face Hub."""
    error_code = "MODEL_NOT_FOUND"


class HFModelNotSupportedError(HFProviderError):
    """Model exists but does not support the requested task."""
    error_code = "MODEL_NOT_SUPPORTED"


class HFProviderUnavailableError(HFProviderError):
    """Hugging Face Inference provider is currently unavailable."""
    error_code = "PROVIDER_UNAVAILABLE"


class HFRateLimitedError(HFProviderError):
    """Request was rate-limited by Hugging Face API."""
    error_code = "RATE_LIMITED"

    def __init__(self, message: str = "Rate limited by Hugging Face API",
                 *, model_id: str = "", task: str = "",
                 retry_after_s: float = 0.0) -> None:
        super().__init__(message, model_id=model_id, task=task)
        self.retry_after_s = retry_after_s


class HFTimeoutError(HFProviderError):
    """Request to Hugging Face API timed out."""
    error_code = "TIMEOUT"

    def __init__(self, message: str = "Request timed out",
                 *, model_id: str = "", task: str = "",
                 timeout_s: float = 0.0) -> None:
        super().__init__(message, model_id=model_id, task=task)
        self.timeout_s = timeout_s


class HFInvalidResponseError(HFProviderError):
    """Hugging Face API returned an unexpected or malformed response."""
    error_code = "INVALID_RESPONSE"


class HFTaskUnsupportedError(HFProviderError):
    """The requested inference task is not supported by this configuration."""
    error_code = "TASK_UNSUPPORTED"


class HFNetworkError(HFProviderError):
    """Network error when communicating with Hugging Face API."""
    error_code = "NETWORK_ERROR"
