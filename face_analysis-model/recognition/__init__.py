"""Face recognition: ArcFace embeddings, the cast database and gallery matching."""

from recognition.arcface import ArcFaceEmbedder, average_embeddings  # noqa: F401
from recognition.cast_database import ActorRecord, CastDatabase  # noqa: F401
from recognition.matcher import UNKNOWN, GalleryMatcher, MatchResult  # noqa: F401
from recognition.registration import CastRegistrar, RegistrationResult  # noqa: F401

__all__ = [
    "ArcFaceEmbedder",
    "average_embeddings",
    "CastDatabase",
    "ActorRecord",
    "GalleryMatcher",
    "MatchResult",
    "UNKNOWN",
    "CastRegistrar",
    "RegistrationResult",
]
