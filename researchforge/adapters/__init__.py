"""ResearchForge adapter layer — public API.

Adapter API version: 1
Schema version:      1

Exports
-------
Errors:
  BackendError, BackendConnectionError, BackendNotFoundError,
  BackendSerializationError, BackendTransactionError,
  BackendCapabilityError, DimensionMismatchError

Capabilities:
  BackendCapabilities, BackendInfo, HealthStatus

Protocols:
  GraphBackend, GraphTransaction, VectorIndexBackend

Backends (reference implementations):
  InMemoryGraphBackend   — in-process, no persistence
  SQLiteGraphBackend     — file-based, persistent, transactional

Backends (documented stubs):
  Neo4jGraphBackend      — raises NotImplementedError
  PgvectorBackend        — raises NotImplementedError

Registry:
  AdapterRegistry, get_default_registry

Validation:
  validate_graph_backend, validate_vector_backend, ValidationReport, ValidationError
"""
from __future__ import annotations

ADAPTER_API_VERSION = 1
ADAPTER_SCHEMA_VERSION = 1

# Errors
from .errors import (                                     # noqa: F401
    BackendError,
    BackendConnectionError,
    BackendNotFoundError,
    BackendSerializationError,
    BackendTransactionError,
    BackendCapabilityError,
    DimensionMismatchError,
)

# Capabilities
from .capabilities import BackendCapabilities, BackendInfo, HealthStatus  # noqa: F401

# Protocols
from .protocols import GraphBackend, GraphTransaction, VectorIndexBackend  # noqa: F401

# Reference backends
from .backends.memory import InMemoryGraphBackend          # noqa: F401
from .backends.sqlite import SQLiteGraphBackend            # noqa: F401
from .backends.inprocess_vector import InProcessVectorIndex  # noqa: F401

# Documented stubs (import is safe; instantiation raises NotImplementedError)
from .backends.neo4j import Neo4jGraphBackend              # noqa: F401
from .backends.pgvector import PgvectorBackend             # noqa: F401

# Registry
from .registry import AdapterRegistry, get_default_registry  # noqa: F401

# Validation
from .validation import (                                  # noqa: F401
    validate_graph_backend,
    validate_vector_backend,
    ValidationReport,
    ValidationError,
)

# Manifest (Phase 10)
from .manifest import (                                    # noqa: F401
    AdapterCapability,
    AdapterDeclaration,
    CanonicalAdapterManifest,
    ImplementationStatus,
)

__all__ = [
    "ADAPTER_API_VERSION",
    "ADAPTER_SCHEMA_VERSION",
    # Errors
    "BackendError", "BackendConnectionError", "BackendNotFoundError",
    "BackendSerializationError", "BackendTransactionError",
    "BackendCapabilityError", "DimensionMismatchError",
    # Capabilities
    "BackendCapabilities", "BackendInfo", "HealthStatus",
    # Protocols
    "GraphBackend", "GraphTransaction", "VectorIndexBackend",
    # Reference implementations
    "InMemoryGraphBackend", "SQLiteGraphBackend", "InProcessVectorIndex",
    # Documented stubs
    "Neo4jGraphBackend", "PgvectorBackend",
    # Registry
    "AdapterRegistry", "get_default_registry",
    # Validation
    "validate_graph_backend", "validate_vector_backend",
    "ValidationReport", "ValidationError",
    # Manifest
    "CanonicalAdapterManifest", "AdapterDeclaration",
    "ImplementationStatus", "AdapterCapability",
]
