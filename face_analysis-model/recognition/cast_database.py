"""Persistent cast store (SQLite, with a pickle mirror for portability).

Schema::

    actors(id, name UNIQUE, created_at, updated_at, dim, num_images, centroid BLOB)
    embeddings(id, actor_id -> actors.id, vector BLOB, source, quality, created_at)

Every enrolment vector is kept, so an actor can be *updated* later (add photos, drop a
bad photo) and the centroid is simply recomputed. The gallery handed to
:class:`recognition.matcher.GalleryMatcher` contains the centroid plus, optionally,
every individual sample - matching against samples with a max-reduction is noticeably
more robust when the reference photos span very different poses.

All public methods are safe to call from several threads; a single connection is shared
under a lock (SQLite serialises writes anyway, and the gallery is small).
"""

from __future__ import annotations

import pickle
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from recognition.arcface import average_embeddings
from utils.logging_utils import get_logger

log = get_logger(__name__)

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS actors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    created_at  REAL    NOT NULL,
    updated_at  REAL    NOT NULL,
    dim         INTEGER NOT NULL DEFAULT 512,
    num_images  INTEGER NOT NULL DEFAULT 0,
    centroid    BLOB
);

CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    actor_id    INTEGER NOT NULL REFERENCES actors(id) ON DELETE CASCADE,
    vector      BLOB    NOT NULL,
    source      TEXT    NOT NULL DEFAULT '',
    quality     REAL    NOT NULL DEFAULT 1.0,
    created_at  REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_embeddings_actor ON embeddings(actor_id);
"""


@dataclass
class ActorRecord:
    """One registered cast member."""

    id: int
    name: str
    centroid: np.ndarray
    num_images: int
    created_at: float
    updated_at: float
    samples: List[np.ndarray] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "num_images": self.num_images,
            "dim": int(self.centroid.shape[0]) if self.centroid.size else 0,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sources": self.sources,
        }


def _to_blob(vec: np.ndarray) -> bytes:
    return np.ascontiguousarray(vec, dtype=np.float32).tobytes()


def _from_blob(blob: Optional[bytes]) -> np.ndarray:
    if not blob:
        return np.zeros((0,), dtype=np.float32)
    return np.frombuffer(blob, dtype=np.float32).copy()


class CastDatabase:
    """SQLite-backed store of actors and their ArcFace embeddings."""

    def __init__(self, db_path: Path, pickle_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path)
        self.pickle_path = Path(pickle_path) if pickle_path else None
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.commit()
        log.info("Cast database opened: %s", self.db_path)

    # -- actors -------------------------------------------------------------
    def add_actor(self, name: str) -> int:
        """Create an actor (or return the existing id for that name)."""
        name = name.strip()
        if not name:
            raise ValueError("actor name must not be empty")
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM actors WHERE name = ?", (name,)
            ).fetchone()
            if row is not None:
                return int(row["id"])
            cur = self._conn.execute(
                "INSERT INTO actors(name, created_at, updated_at) VALUES (?, ?, ?)",
                (name, now, now),
            )
            self._conn.commit()
            log.info("Registered new actor '%s' (id=%d)", name, cur.lastrowid)
            return int(cur.lastrowid)

    def rename_actor(self, actor_id: int, new_name: str) -> None:
        """Change an actor's display name."""
        with self._lock:
            self._conn.execute(
                "UPDATE actors SET name = ?, updated_at = ? WHERE id = ?",
                (new_name.strip(), time.time(), actor_id),
            )
            self._conn.commit()

    def delete_actor(self, actor_id: int) -> None:
        """Remove an actor and every embedding they own."""
        with self._lock:
            self._conn.execute("DELETE FROM embeddings WHERE actor_id = ?", (actor_id,))
            self._conn.execute("DELETE FROM actors WHERE id = ?", (actor_id,))
            self._conn.commit()
        log.info("Deleted actor id=%d", actor_id)

    def get_actor_id(self, name: str) -> Optional[int]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM actors WHERE name = ?", (name.strip(),)
            ).fetchone()
        return int(row["id"]) if row else None

    # -- embeddings ---------------------------------------------------------
    def add_embeddings(
        self,
        actor_id: int,
        vectors: np.ndarray,
        sources: Optional[Sequence[str]] = None,
        qualities: Optional[Sequence[float]] = None,
    ) -> int:
        """Append enrolment vectors for an actor and refresh their centroid.

        Args:
            actor_id: target actor.
            vectors: ``(N, dim)`` L2-normalised embeddings.
            sources: per-vector provenance (image filename).
            qualities: per-vector quality weights in ``[0, 1]``.

        Returns:
            Number of vectors stored.
        """
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            return 0
        now = time.time()
        srcs = list(sources) if sources else [""] * vectors.shape[0]
        quals = list(qualities) if qualities else [1.0] * vectors.shape[0]
        rows = [
            (actor_id, _to_blob(vectors[i]), srcs[i], float(quals[i]), now)
            for i in range(vectors.shape[0])
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT INTO embeddings(actor_id, vector, source, quality, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()
        self.recompute_centroid(actor_id)
        return len(rows)

    def remove_embedding(self, embedding_id: int) -> None:
        """Drop one enrolment vector (e.g. a bad reference photo)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT actor_id FROM embeddings WHERE id = ?", (embedding_id,)
            ).fetchone()
            self._conn.execute("DELETE FROM embeddings WHERE id = ?", (embedding_id,))
            self._conn.commit()
        if row:
            self.recompute_centroid(int(row["actor_id"]))

    def recompute_centroid(self, actor_id: int) -> np.ndarray:
        """Recompute and store the quality-weighted centroid for one actor."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT vector, quality FROM embeddings WHERE actor_id = ?", (actor_id,)
            ).fetchall()
        if not rows:
            with self._lock:
                self._conn.execute(
                    "UPDATE actors SET centroid = NULL, num_images = 0, updated_at = ? "
                    "WHERE id = ?",
                    (time.time(), actor_id),
                )
                self._conn.commit()
            return np.zeros((0,), dtype=np.float32)

        vectors = np.stack([_from_blob(r["vector"]) for r in rows], axis=0)
        weights = np.asarray([r["quality"] for r in rows], dtype=np.float32)
        centroid = average_embeddings(vectors, weights)
        with self._lock:
            self._conn.execute(
                "UPDATE actors SET centroid = ?, num_images = ?, dim = ?, updated_at = ? "
                "WHERE id = ?",
                (_to_blob(centroid), len(rows), int(centroid.shape[0]), time.time(), actor_id),
            )
            self._conn.commit()
        return centroid

    # -- queries ------------------------------------------------------------
    def list_actors(self, with_samples: bool = False) -> List[ActorRecord]:
        """Every registered actor, optionally including their enrolment vectors."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, centroid, num_images, created_at, updated_at "
                "FROM actors ORDER BY name COLLATE NOCASE"
            ).fetchall()
            records: List[ActorRecord] = []
            for r in rows:
                rec = ActorRecord(
                    id=int(r["id"]),
                    name=str(r["name"]),
                    centroid=_from_blob(r["centroid"]),
                    num_images=int(r["num_images"]),
                    created_at=float(r["created_at"]),
                    updated_at=float(r["updated_at"]),
                )
                if with_samples:
                    srows = self._conn.execute(
                        "SELECT vector, source FROM embeddings WHERE actor_id = ?",
                        (rec.id,),
                    ).fetchall()
                    rec.samples = [_from_blob(s["vector"]) for s in srows]
                    rec.sources = [str(s["source"]) for s in srows]
                records.append(rec)
        return records

    def build_gallery(
        self, include_samples: bool = True
    ) -> Tuple[np.ndarray, List[int], Dict[int, str]]:
        """Materialise the matcher gallery.

        Returns:
            ``(vectors, owner_ids, id_to_name)`` ready for
            :meth:`recognition.matcher.GalleryMatcher.set_gallery`.
        """
        actors = self.list_actors(with_samples=include_samples)
        vectors: List[np.ndarray] = []
        owners: List[int] = []
        names: Dict[int, str] = {}
        for actor in actors:
            names[actor.id] = actor.name
            if actor.centroid.size:
                vectors.append(actor.centroid)
                owners.append(actor.id)
            if include_samples:
                for sample in actor.samples:
                    if sample.size:
                        vectors.append(sample)
                        owners.append(actor.id)
        if not vectors:
            return np.zeros((0, 512), dtype=np.float32), [], names
        return np.stack(vectors, axis=0).astype(np.float32), owners, names

    def stats(self) -> Dict[str, int]:
        """Row counts, for the GUI status line."""
        with self._lock:
            actors = self._conn.execute("SELECT COUNT(*) AS c FROM actors").fetchone()["c"]
            embs = self._conn.execute("SELECT COUNT(*) AS c FROM embeddings").fetchone()["c"]
        return {"actors": int(actors), "embeddings": int(embs)}

    # -- pickle mirror ------------------------------------------------------
    def export_pickle(self, path: Optional[Path] = None) -> Optional[Path]:
        """Write a portable ``cast_database.pkl`` snapshot.

        The pickle holds ``{"actors": [{"actor_name", "embedding", "samples", ...}]}``,
        which is the format the spec asks for and which another process can load
        without touching SQLite.
        """
        target = Path(path) if path else self.pickle_path
        if target is None:
            return None
        actors = self.list_actors(with_samples=True)
        payload = {
            "version": 1,
            "created_at": time.time(),
            "actors": [
                {
                    "actor_id": a.id,
                    "actor_name": a.name,
                    "embedding": a.centroid,
                    "samples": np.stack(a.samples) if a.samples else np.zeros((0, 512), np.float32),
                    "sources": a.sources,
                    "num_images": a.num_images,
                }
                for a in actors
            ],
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Cast pickle written: %s (%d actors)", target, len(actors))
        return target

    def import_pickle(self, path: Path) -> int:
        """Load actors from a pickle snapshot; returns the number of actors merged."""
        path = Path(path)
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        count = 0
        for entry in payload.get("actors", []):
            actor_id = self.add_actor(entry["actor_name"])
            samples = entry.get("samples")
            if samples is not None and len(samples):
                self.add_embeddings(
                    actor_id, np.asarray(samples, dtype=np.float32),
                    entry.get("sources"), None,
                )
            elif entry.get("embedding") is not None and len(entry["embedding"]):
                self.add_embeddings(
                    actor_id, np.asarray(entry["embedding"], np.float32).reshape(1, -1)
                )
            count += 1
        log.info("Imported %d actors from %s", count, path)
        return count

    # -- lifecycle ----------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            self._conn.commit()
            self._conn.close()

    def __enter__(self) -> "CastDatabase":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["CastDatabase", "ActorRecord"]
