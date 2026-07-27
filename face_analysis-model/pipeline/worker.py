"""The analysis pipeline: eight overlapping stages joined by bounded queues.

::

    decode -> sample -> detect -> track -+-> recognise (ArcFace)  -+-> timeline
                                          \\-> emotion  (HSEmotion) -/

Each arrow is a ``queue.Queue`` with a hard size cap, so a slow stage applies
back-pressure instead of growing memory without bound. Recognition and emotion run on
their own CUDA streams and pull *cross-frame* batches, so the GPU sees full batches even
when a frame contains a single face.

Frames are assembled back into order by :class:`_FrameAssembler`: results may complete
out of order (a recognition batch can finish after a later emotion batch), but the
timeline stage always receives frames in increasing index order, which is what makes
the emitted events monotonic.

Threading contract
------------------
* ``IdentityRegistry`` is touched by the track and recognition threads -> guarded by
  ``self._identity_lock``.
* ``SmootherRegistry`` is touched only by the emotion thread -> no lock.
* ``TimelineEngine`` is touched only by the timeline thread -> no lock.
* Callbacks are invoked from worker threads; the GUI adapter re-emits them as queued
  Qt signals.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from configs.config import AppConfig
from detector.base import Detection
from detector.face_quality import FaceQualityEstimator
from detector.scrfd import SCRFDDetector
from emotion.hsemotion import EmotionResult, HSEmotionClassifier
from emotion.smoothing import SmootherRegistry
from models.onnx_engine import available_providers
from pipeline.batching import BatchCollector
from pipeline.frame_source import DecodeThread, VideoMetadata, VideoSource, probe
from pipeline.sampler import AdaptiveSampler
from pipeline.types import (
    FaceObservation,
    FrameResult,
    ProgressUpdate,
    RawFrame,
    SampledFrame,
)
from recognition.arcface import ArcFaceEmbedder
from recognition.cast_database import CastDatabase
from recognition.matcher import GalleryMatcher, MatchResult
from timeline.events import AnalysisInfo, TimelineDocument, TimelineEvent, VideoInfo
from timeline.timeline_engine import TimelineEngine
from tracking.byte_tracker import ByteTracker, STrack
from tracking.track import IdentityRegistry
from utils.gpu import (
    GpuMonitor,
    StreamContext,
    apply_torch_performance_flags,
    empty_cache,
    memory_summary,
    resolve_device,
)
from utils.image_ops import align_face
from utils.logging_utils import get_logger
from utils.profiling import Profiler

log = get_logger(__name__)

_SENTINEL = None


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@dataclass
class PipelineCallbacks:
    """Hooks invoked from worker threads (keep them cheap and non-blocking)."""

    on_frame: Optional[Callable[[FrameResult], None]] = None
    on_event: Optional[Callable[[TimelineEvent], None]] = None
    on_progress: Optional[Callable[[ProgressUpdate], None]] = None
    on_finished: Optional[Callable[[TimelineDocument], None]] = None
    on_error: Optional[Callable[[BaseException], None]] = None

    def fire(self, name: str, payload) -> None:
        fn = getattr(self, name)
        if fn is None:
            return
        try:
            fn(payload)
        except Exception:  # pragma: no cover - a broken UI must not kill the pipeline
            log.exception("Callback %s raised", name)


# ---------------------------------------------------------------------------
# Work items
# ---------------------------------------------------------------------------
@dataclass
class _RecogItem:
    """One face queued for ArcFace."""

    seq: int
    track_id: int
    crop: np.ndarray
    frame_index: int
    timestamp: float


@dataclass
class _EmotionItem:
    """One face queued for HSEmotion."""

    seq: int
    track_id: int
    crop: np.ndarray
    frame_index: int
    timestamp: float


@dataclass
class _PendingFrame:
    """A frame in flight, waiting for its recognition/emotion results."""

    seq: int
    frame: SampledFrame
    observations: Dict[int, FaceObservation] = field(default_factory=dict)
    order: List[int] = field(default_factory=list)
    pending_recog: int = 0
    pending_emotion: int = 0
    activity: bool = False
    submitted_at: float = field(default_factory=time.perf_counter)

    @property
    def complete(self) -> bool:
        return self.pending_recog <= 0 and self.pending_emotion <= 0


class _FrameAssembler:
    """Re-orders asynchronous results into a monotonic frame stream."""

    def __init__(self, out_queue: "queue.Queue[Optional[_PendingFrame]]") -> None:
        self._out = out_queue
        self._lock = threading.Lock()
        self._pending: List[_PendingFrame] = []
        self._index: Dict[int, _PendingFrame] = {}

    def submit(self, pending: _PendingFrame) -> None:
        with self._lock:
            self._pending.append(pending)
            self._index[pending.seq] = pending
            ready = self._collect_ready()
        self._emit(ready)

    def complete_recognition(self, seq: int, n: int = 1) -> None:
        self._complete(seq, recog=n)

    def complete_emotion(self, seq: int, n: int = 1) -> None:
        self._complete(seq, emotion=n)

    def _complete(self, seq: int, recog: int = 0, emotion: int = 0) -> None:
        with self._lock:
            item = self._index.get(seq)
            if item is None:
                return
            item.pending_recog -= recog
            item.pending_emotion -= emotion
            ready = self._collect_ready()
        self._emit(ready)

    def _collect_ready(self) -> List[_PendingFrame]:
        """Pop the completed prefix (caller holds the lock)."""
        ready: List[_PendingFrame] = []
        while self._pending and self._pending[0].complete:
            item = self._pending.pop(0)
            self._index.pop(item.seq, None)
            ready.append(item)
        return ready

    def _emit(self, ready: Sequence[_PendingFrame]) -> None:
        for item in ready:
            self._out.put(item)

    def flush(self) -> None:
        """Force-release everything still pending (end of stream)."""
        with self._lock:
            ready = list(self._pending)
            self._pending.clear()
            self._index.clear()
        self._emit(ready)

    @property
    def depth(self) -> int:
        with self._lock:
            return len(self._pending)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class AnalysisPipeline:
    """End-to-end face-recognition + emotion-timeline pipeline.

    Typical use::

        pipeline = AnalysisPipeline(config)
        pipeline.prepare()                  # loads models, builds gallery
        document = pipeline.analyze("clip.mp4")

    or asynchronously (GUI)::

        pipeline.start("clip.mp4")
        ...
        pipeline.stop()
    """

    def __init__(
        self,
        config: AppConfig,
        callbacks: Optional[PipelineCallbacks] = None,
        database: Optional[CastDatabase] = None,
    ) -> None:
        self.cfg = config
        self.callbacks = callbacks or PipelineCallbacks()
        self.device = resolve_device(config.gpu.device, config.gpu.device_id)
        apply_torch_performance_flags()

        self.profiler = Profiler(enabled=config.logging.profile)
        self.gpu_monitor = GpuMonitor(config.gpu.device_id)

        self.db = database or CastDatabase(
            config.paths.resolve(config.paths.cast_db),
            config.paths.resolve(config.paths.cast_pickle),
        )
        self.matcher = GalleryMatcher(
            self.device,
            similarity_threshold=config.recognition.similarity_threshold,
            margin_threshold=config.recognition.margin_threshold,
        )

        # Per-stage CUDA streams so H2D copies overlap compute.
        use_streams = config.gpu.use_cuda_streams
        self._stream_detect = StreamContext(self.device, use_streams)
        self._stream_recog = StreamContext(self.device, use_streams)
        self._stream_emotion = StreamContext(self.device, use_streams)

        self.detector: Optional[SCRFDDetector] = None
        self.embedder: Optional[ArcFaceEmbedder] = None
        self.classifier: Optional[HSEmotionClassifier] = None
        self.quality = FaceQualityEstimator(config.quality)

        self.tracker = ByteTracker(config.tracker)
        self.identities = IdentityRegistry(config.recognition)
        self.smoothers = SmootherRegistry(config.emotion)
        self.timeline = TimelineEngine(config.timeline)

        self._identity_lock = threading.RLock()
        self._threads: List[threading.Thread] = []
        self._stop_event = threading.Event()
        self._running = False
        self._document: Optional[TimelineDocument] = None
        self._error: Optional[BaseException] = None
        self._done_event = threading.Event()

        self.meta: Optional[VideoMetadata] = None
        self.sampler: Optional[AdaptiveSampler] = None
        # (seq, track_id) -> smoothed emotion tuple. Written by the emotion thread,
        # consumed by the timeline thread; the assembler lock provides the ordering
        # (the entry is always written before ``complete_emotion`` releases the frame).
        self._emotion_results: Dict[Tuple[int, int], Tuple[str, float, str, float, bool]] = {}
        self._decoder_name = ""
        self._frames_decoded = 0
        self._frames_analysed = 0
        self._faces_seen = 0
        self._expression_changes = 0
        self._start_time = 0.0
        self._started_at_iso = ""
        self._prepared = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def prepare(self, force: bool = False) -> None:
        """Load models and the cast gallery. Safe to call once per process."""
        if self._prepared and not force:
            return
        t0 = time.perf_counter()
        log.info("Preparing pipeline | %s", self.cfg.describe())
        log.info("ORT providers available: %s", available_providers())

        self.detector = SCRFDDetector(
            self.cfg, self.device, self._stream_detect.cuda_stream_handle or None
        )
        self.embedder = ArcFaceEmbedder(
            self.cfg, self.device, self._stream_recog.cuda_stream_handle or None
        )
        self.classifier = HSEmotionClassifier(
            self.cfg, self.device, self._stream_emotion.cuda_stream_handle or None
        )
        self.smoothers = SmootherRegistry(self.cfg.emotion, self.classifier.labels)
        self.reload_gallery()

        with self._stream_detect:
            self.detector.warmup()
        with self._stream_recog:
            self.embedder.warmup()
        with self._stream_emotion:
            self.classifier.warmup()

        self._prepared = True
        log.info(
            "Pipeline ready in %.2fs | VRAM %s", time.perf_counter() - t0,
            memory_summary(self.cfg.gpu.device_id),
        )

    def reload_gallery(self) -> int:
        """Rebuild the matcher gallery from the cast database."""
        vectors, owners, names = self.db.build_gallery(include_samples=True)
        self.matcher.set_gallery(vectors, owners, names)
        if self.matcher.is_empty:
            log.warning("Cast database is empty - every face will be reported Unknown")
        return self.matcher.num_actors

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------
    def analyze(self, video_path: str | Path) -> TimelineDocument:
        """Run the full pipeline synchronously and return the timeline."""
        self.start(video_path)
        self._done_event.wait()
        if self._error is not None:
            raise self._error
        assert self._document is not None
        return self._document

    def start(self, video_path: str | Path) -> None:
        """Launch every stage thread and return immediately."""
        if self._running:
            raise RuntimeError("pipeline already running")
        self.prepare()
        self._reset_state()

        self.meta = probe(video_path)
        self.sampler = AdaptiveSampler(self.cfg.sampling, self.cfg.video)
        log.info(
            "Analysing %s | %dx%d @ %.2f fps, %d frames (%.1fs)",
            Path(str(video_path)).name, self.meta.width, self.meta.height,
            self.meta.fps, self.meta.frame_count, self.meta.duration,
        )

        source = VideoSource(
            video_path,
            backend=self.cfg.video.backend,
            hwaccel=self.cfg.video.hwaccel,
            threads=self.cfg.video.decode_threads,
            # Detection never needs more than the analysis resolution, and the pipeline
            # is decode-bound on most clips - so scale down inside the colour conversion.
            target_long_side=self.cfg.video.analysis_long_side,
        )
        self._decoder_name = source.decoder_name

        pcfg = self.cfg.pipeline
        self._q_raw: "queue.Queue[Optional[RawFrame]]" = queue.Queue(self.cfg.video.queue_size)
        self._q_sampled: "queue.Queue[Optional[SampledFrame]]" = queue.Queue(pcfg.detect_queue)
        self._q_detected: "queue.Queue[Optional[Tuple[SampledFrame, List[Detection]]]]" = (
            queue.Queue(pcfg.track_queue)
        )
        self._q_recog: "queue.Queue[Optional[_RecogItem]]" = queue.Queue(pcfg.infer_queue)
        self._q_emotion: "queue.Queue[Optional[_EmotionItem]]" = queue.Queue(pcfg.infer_queue)
        self._q_release: "queue.Queue[Optional[_PendingFrame]]" = queue.Queue(pcfg.result_queue)
        self._assembler = _FrameAssembler(self._q_release)

        self._running = True
        self._start_time = time.perf_counter()
        self._started_at_iso = datetime.now().isoformat(timespec="seconds")
        self._done_event.clear()

        decoder = DecodeThread(source, self._q_raw, drop_when_full=False)
        self._threads = [
            decoder,
            threading.Thread(target=self._sample_loop, name="SampleThread", daemon=True),
            threading.Thread(target=self._detect_loop, name="DetectThread", daemon=True),
            threading.Thread(target=self._track_loop, name="TrackThread", daemon=True),
            threading.Thread(target=self._recognition_loop, name="RecogThread", daemon=True),
            threading.Thread(target=self._emotion_loop, name="EmotionThread", daemon=True),
            threading.Thread(target=self._timeline_loop, name="TimelineThread", daemon=True),
            threading.Thread(target=self._monitor_loop, name="MonitorThread", daemon=True),
        ]
        self._decoder = decoder
        self._source = source
        for t in self._threads:
            t.start()
        threading.Thread(target=self._await_completion, name="PipelineJoin", daemon=True).start()

    def stop(self, timeout: float = 5.0) -> None:
        """Request shutdown and wait for the threads to unwind."""
        if not self._running:
            return
        log.info("Stopping pipeline...")
        self._stop_event.set()
        if hasattr(self, "_decoder"):
            self._decoder.stop()
        self._done_event.wait(timeout=timeout)
        self._running = False

    def wait(self, timeout: Optional[float] = None) -> Optional[TimelineDocument]:
        """Block until the run finishes; returns the document (or ``None`` on timeout)."""
        self._done_event.wait(timeout=timeout)
        return self._document

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def document(self) -> Optional[TimelineDocument]:
        return self._document

    # ------------------------------------------------------------------
    # Stage 1: sampling
    # ------------------------------------------------------------------
    def _sample_loop(self) -> None:
        assert self.sampler is not None
        try:
            while not self._stop_event.is_set():
                try:
                    frame = self._q_raw.get(timeout=0.2)
                except queue.Empty:
                    continue
                if frame is _SENTINEL:
                    break
                self._frames_decoded += 1
                self.profiler.mark("decode")
                with self.profiler.timer("sample"):
                    sampled = self.sampler.consider(frame)
                if sampled is not None:
                    self._put(self._q_sampled, sampled)
        except BaseException as exc:  # noqa: BLE001
            self._fail(exc)
        finally:
            self._q_sampled.put(_SENTINEL)

    # ------------------------------------------------------------------
    # Stage 2: detection (batched)
    # ------------------------------------------------------------------
    def _detect_loop(self) -> None:
        assert self.detector is not None
        collector = BatchCollector[SampledFrame](
            self._q_sampled, self.cfg.detector.batch_size,
            self.cfg.pipeline.max_batch_latency_ms,
        )
        try:
            with self._stream_detect:
                while not self._stop_event.is_set():
                    batch = collector.next_batch()
                    if batch:
                        with self.profiler.timer("detect", mark=len(batch)):
                            detections = self.detector.detect_batch([f.image for f in batch])
                        for frame, dets in zip(batch, detections):
                            self._put(self._q_detected, (frame, dets))
                    elif collector.finished:
                        break
            log.debug("Detector mean batch size: %.2f", collector.mean_batch)
        except BaseException as exc:  # noqa: BLE001
            self._fail(exc)
        finally:
            self._q_detected.put(_SENTINEL)

    # ------------------------------------------------------------------
    # Stage 3: tracking + crop preparation
    # ------------------------------------------------------------------
    def _track_loop(self) -> None:
        assert self.detector is not None and self.embedder is not None
        assert self.classifier is not None
        seq = 0
        try:
            while not self._stop_event.is_set():
                try:
                    item = self._q_detected.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item is _SENTINEL:
                    break
                frame, detections = item
                seq += 1
                self._frames_analysed += 1
                with self.profiler.timer("track", mark=1):
                    pending = self._process_frame(seq, frame, detections)
                self._assembler.submit(pending)
        except BaseException as exc:  # noqa: BLE001
            self._fail(exc)
        finally:
            self._q_recog.put(_SENTINEL)
            self._q_emotion.put(_SENTINEL)

    def _process_frame(
        self, seq: int, frame: SampledFrame, detections: Sequence[Detection]
    ) -> _PendingFrame:
        """Track, gate on quality, and queue the inference work for one frame."""
        pending = _PendingFrame(seq=seq, frame=frame)

        # A hard cut invalidates motion continuity: restart association cleanly.
        if frame.scene_cut:
            self.tracker.reset()
            with self._identity_lock:
                self.identities.reset()
            self.smoothers.reset()
            pending.activity = True

        for det in detections:
            det.frame_index = frame.index
        tracks: List[STrack] = self.tracker.update(list(detections))
        active_ids = [t.track_id for t in tracks]
        with self._identity_lock:
            self.identities.tick(active_ids)

        recog_items: List[_RecogItem] = []
        emotion_items: List[_EmotionItem] = []

        for track in tracks:
            bbox = track.bbox
            landmarks = track.landmarks
            aligned: Optional[np.ndarray] = None
            if landmarks is not None:
                aligned = align_face(frame.image, landmarks, self.cfg.recognition.input_size)

            report = self.quality.evaluate(track.det, aligned)
            obs = FaceObservation(
                track_id=track.track_id,
                bbox=bbox * frame.scale,
                det_score=float(track.score),
                quality=report.score,
            )
            with self._identity_lock:
                identity = self.identities.touch(
                    track.track_id, frame.index, frame.timestamp
                )
                obs.actor_id = identity.actor_id
                obs.actor_name = identity.name
                obs.similarity = identity.similarity
                obs.locked = identity.locked
                is_new = identity.embeddings_run == 0
            if is_new:
                pending.activity = True

            pending.observations[track.track_id] = obs
            pending.order.append(track.track_id)

            if not report.passed:
                continue  # gated: no embedding, no emotion, but the box is still shown

            self._faces_seen += 1

            # --- recognition scheduling ---
            if aligned is not None:
                with self._identity_lock:
                    wanted = self.identities.needs_recognition(track.track_id)
                    if wanted:
                        self.identities.mark_pending(track.track_id)
                if wanted:
                    recog_items.append(
                        _RecogItem(seq, track.track_id, aligned, frame.index, frame.timestamp)
                    )

            # --- emotion ---
            # In cast-only mode a face must first *be* somebody: HSEmotion runs only
            # once the track has matched a registered actor. Extras are tracked but
            # never classified, which is where most of the saving on crowd shots comes
            # from. Identity is resolved a frame earlier than the emotion it gates, so
            # a newly locked actor starts being classified on the next analysed frame.
            if self.cfg.recognition.cast_only and obs.actor_id < 0:
                continue

            if self.cfg.gui.overlay_boxes:
                obs.thumbnail = self._make_thumbnail(frame.image, bbox)
            emotion_items.append(
                _EmotionItem(
                    seq, track.track_id,
                    self.classifier.make_crop(frame.image, bbox),
                    frame.index, frame.timestamp,
                )
            )

        pending.pending_recog = len(recog_items)
        pending.pending_emotion = len(emotion_items)
        for item in recog_items:
            self._put(self._q_recog, item)
        for item in emotion_items:
            self._put(self._q_emotion, item)
        return pending

    def _make_thumbnail(self, image: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        size = self.cfg.gui.face_thumb_size
        h, w = image.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        crop = image[y1:y2, x1:x2]
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------
    # Stage 4: recognition (batched across frames)
    # ------------------------------------------------------------------
    def _recognition_loop(self) -> None:
        assert self.embedder is not None
        collector = BatchCollector[_RecogItem](
            self._q_recog, self.cfg.recognition.batch_size,
            self.cfg.pipeline.max_batch_latency_ms,
        )
        try:
            with self._stream_recog:
                while not self._stop_event.is_set():
                    batch = collector.next_batch()
                    if batch:
                        self._run_recognition(batch)
                    elif collector.finished:
                        break
            log.debug("ArcFace mean batch size: %.2f", collector.mean_batch)
        except BaseException as exc:  # noqa: BLE001
            self._fail(exc)

    def _run_recognition(self, batch: Sequence[_RecogItem]) -> None:
        with self.profiler.timer("recognition", mark=len(batch)):
            embeddings = self.embedder.embed_aligned([item.crop for item in batch])
            matches: List[MatchResult] = self.matcher.match(embeddings)
        with self._identity_lock:
            for item, match in zip(batch, matches):
                self.identities.update(item.track_id, match, item.frame_index, item.timestamp)
        for item in batch:
            self._assembler.complete_recognition(item.seq)

    # ------------------------------------------------------------------
    # Stage 5: emotion (batched across frames) + smoothing
    # ------------------------------------------------------------------
    def _emotion_loop(self) -> None:
        assert self.classifier is not None
        collector = BatchCollector[_EmotionItem](
            self._q_emotion, self.cfg.emotion.batch_size,
            self.cfg.pipeline.max_batch_latency_ms,
        )
        try:
            with self._stream_emotion:
                while not self._stop_event.is_set():
                    batch = collector.next_batch()
                    if batch:
                        self._run_emotion(batch)
                    elif collector.finished:
                        break
            log.debug("HSEmotion mean batch size: %.2f", collector.mean_batch)
        except BaseException as exc:  # noqa: BLE001
            self._fail(exc)

    def _run_emotion(self, batch: Sequence[_EmotionItem]) -> None:
        with self.profiler.timer("emotion", mark=len(batch)):
            results: List[EmotionResult] = self.classifier.predict_crops(
                [item.crop for item in batch]
            )
        for item, result in zip(batch, results):
            smoother = self.smoothers.get(item.track_id)
            smoothed = smoother.update(result.label, result.confidence, result.probabilities)
            self._emotion_results[item.seq, item.track_id] = (
                smoothed.label, smoothed.confidence, result.label, result.confidence,
                smoothed.changed,
            )
            self._assembler.complete_emotion(item.seq)

    # ------------------------------------------------------------------
    # Stage 6: timeline assembly
    # ------------------------------------------------------------------
    def _timeline_loop(self) -> None:
        try:
            while True:
                try:
                    pending = self._q_release.get(timeout=0.2)
                except queue.Empty:
                    if self._stop_event.is_set():
                        break
                    continue
                if pending is _SENTINEL:
                    break
                self._finalise_frame(pending)
        except BaseException as exc:  # noqa: BLE001
            self._fail(exc)

    def _finalise_frame(self, pending: _PendingFrame) -> None:
        frame = pending.frame
        faces: List[FaceObservation] = []
        activity = pending.activity

        for track_id in pending.order:
            obs = pending.observations[track_id]
            with self._identity_lock:
                identity = self.identities.get(track_id)
                obs.actor_id = identity.actor_id
                obs.actor_name = identity.name
                obs.similarity = identity.similarity
                obs.locked = identity.locked

            emotion_state = self._emotion_results.pop((pending.seq, track_id), None)
            if emotion_state is not None:
                (
                    obs.emotion, obs.emotion_confidence,
                    obs.raw_emotion, obs.raw_emotion_confidence,
                    obs.emotion_changed,
                ) = emotion_state
                if obs.emotion_changed:
                    activity = True
                    if obs.actor_id >= 0:
                        self._expression_changes += 1
                closed = self.timeline.observe(
                    track_id=track_id,
                    actor_id=obs.actor_id,
                    actor_name=obs.actor_name,
                    emotion=obs.emotion,
                    confidence=obs.emotion_confidence,
                    timestamp=frame.timestamp,
                    frame_index=frame.index,
                    similarity=obs.similarity,
                )
                if closed is not None:
                    self.callbacks.fire("on_event", closed)
            faces.append(obs)

        for event in self.timeline.close_stale(frame.timestamp):
            self.callbacks.fire("on_event", event)

        if self.sampler is not None:
            self.sampler.report_activity(activity)

        if not self.cfg.gui.show_unknown_faces:
            faces = [f for f in faces if f.actor_id >= 0]

        result = FrameResult(
            index=frame.index,
            timestamp=frame.timestamp,
            faces=faces,
            scene_cut=frame.scene_cut,
            stride_used=frame.stride_used,
            latency_ms=(time.perf_counter() - pending.submitted_at) * 1000.0,
            frame_size=(frame.image.shape[0], frame.image.shape[1]),
        )
        self.profiler.mark("analysis")
        self.profiler.observe("frame_latency", result.latency_ms)
        self.callbacks.fire("on_frame", result)

        # Periodic housekeeping: drop state for tracks that stopped being observed.
        # Staleness is measured in source frames so this needs no access to the
        # tracker's live state from this thread.
        if pending.seq % 120 == 0:
            with self._identity_lock:
                self.identities.prune_stale(frame.index, max_age=600)
                alive = [rec.track_id for rec in self.identities.all()]
            self.smoothers.prune(alive)

    # ------------------------------------------------------------------
    # Stage 7: telemetry
    # ------------------------------------------------------------------
    def _monitor_loop(self) -> None:
        interval = max(0.1, self.cfg.gui.stats_interval_ms / 1000.0)
        last_log = time.perf_counter()
        while not self._done_event.is_set() and not self._stop_event.is_set():
            time.sleep(interval)
            update = self._build_progress()
            self.callbacks.fire("on_progress", update)
            now = time.perf_counter()
            if self.cfg.logging.profile and (now - last_log) >= self.cfg.logging.profile_interval_s * 4:
                last_log = now
                log.debug(
                    "%.1f%% | decode %.1f fps | analysis %.1f fps | rt x%.2f | GPU %.0f%% %.0f MB",
                    update.percent, update.decode_fps, update.analysis_fps,
                    update.realtime_factor, update.gpu_utilization, update.gpu_memory_mb,
                )

    def _build_progress(self, finished: bool = False) -> ProgressUpdate:
        snap = self.profiler.snapshot()
        tel = self.gpu_monitor.sample()
        elapsed = max(1e-6, time.perf_counter() - self._start_time)
        meta = self.meta or VideoMetadata(path="")
        decoded_seconds = (self._frames_decoded / meta.fps) if meta.fps else 0.0

        def fps(name: str) -> float:
            return snap[name].fps if name in snap else 0.0

        return ProgressUpdate(
            frames_decoded=self._frames_decoded,
            frames_analysed=self._frames_analysed,
            total_frames=meta.frame_count,
            timestamp=decoded_seconds,
            duration=meta.duration,
            decode_fps=fps("decode"),
            analysis_fps=fps("analysis"),
            detect_fps=fps("detect"),
            recognition_fps=fps("recognition"),
            emotion_fps=fps("emotion"),
            inference_fps=fps("detect") + fps("recognition") + fps("emotion"),
            realtime_factor=decoded_seconds / elapsed,
            gpu_utilization=tel.utilization,
            gpu_memory_mb=tel.memory_used_mb,
            gpu_memory_total_mb=tel.memory_total_mb,
            queue_depths={
                "raw": self._q_raw.qsize(),
                "sampled": self._q_sampled.qsize(),
                "detected": self._q_detected.qsize(),
                "recog": self._q_recog.qsize(),
                "emotion": self._q_emotion.qsize(),
                "release": self._q_release.qsize(),
                "assembler": self._assembler.depth,
            },
            events=self.timeline.event_count,
            expression_changes=self._expression_changes,
            cast_faces=self._faces_seen,
            elapsed=elapsed,
            finished=finished,
        )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def _await_completion(self) -> None:
        """Join the stages in dependency order, then build the document."""
        try:
            names = {t.name: t for t in self._threads}
            for name in (
                "DecodeThread", "SampleThread", "DetectThread", "TrackThread",
                "RecogThread", "EmotionThread",
            ):
                thread = names.get(name)
                if thread is not None:
                    thread.join()
            # Everything upstream is done: release stragglers and close the timeline.
            self._assembler.flush()
            self._q_release.put(_SENTINEL)
            names["TimelineThread"].join()

            self._document = self._build_document()
            self.callbacks.fire("on_progress", self._build_progress(finished=True))
            self.callbacks.fire("on_finished", self._document)
        except BaseException as exc:  # noqa: BLE001
            self._fail(exc)
        finally:
            self._running = False
            self._done_event.set()
            try:
                self._source.close()
            except Exception:  # pragma: no cover
                pass
            empty_cache()

    def _build_document(self) -> TimelineDocument:
        meta = self.meta or VideoMetadata(path="")
        elapsed = max(1e-6, time.perf_counter() - self._start_time)
        sampler_stats = self.sampler.stats.as_dict() if self.sampler else {}
        processed_fps = (
            self.sampler.actual_processed_fps(meta.fps) if self.sampler and meta.fps else 0.0
        )
        info = VideoInfo(
            path=meta.path,
            duration=meta.duration,
            fps=meta.fps,
            processed_fps=processed_fps,
            frame_count=meta.frame_count,
            width=meta.width,
            height=meta.height,
            frames_analysed=self._frames_analysed,
            decoder=getattr(self, "_decoder_name", ""),
        )
        snap = self.profiler.snapshot()
        realtime = round((meta.duration / elapsed) if meta.duration else 0.0, 3)
        stats = {
            "elapsed_seconds": round(elapsed, 3),
            "realtime_factor": realtime,
            "frames_decoded": self._frames_decoded,
            "frames_analysed": self._frames_analysed,
            "faces_processed": self._faces_seen,
            "device": str(self.device),
            "fp16": self.cfg.gpu.fp16,
            "cast_only": self.cfg.recognition.cast_only,
            "decoder": getattr(self, "_decoder_name", ""),
            "sampling": sampler_stats,
            "stages": {k: v.as_dict() for k, v in snap.items()},
            "gpu": memory_summary(self.cfg.gpu.device_id),
        }
        analysis = AnalysisInfo(
            elapsed_seconds=elapsed,
            realtime_factor=realtime,
            frames_decoded=self._frames_decoded,
            frames_analysed=self._frames_analysed,
            faces_processed=self._faces_seen,
            expression_changes=self._expression_changes,
            recognition_calls=snap["recognition"].count if "recognition" in snap else 0,
            device=str(self.device),
            decoder=getattr(self, "_decoder_name", ""),
            started_at=self._started_at_iso,
            finished_at=datetime.now().isoformat(timespec="seconds"),
        )
        doc = self.timeline.build_document(info, stats, analysis)
        log.info(
            "Finished in %.2fs (%.2fx realtime) | %d events | %d expression changes | "
            "%d/%d frames analysed (%.1f%%)",
            elapsed, realtime, len(doc.events),
            # The document's count, not the raw smoother counter: same definition the
            # JSON reports, so the log and the output can never tell different stories.
            doc.analysis.expression_changes,
            self._frames_analysed, meta.frame_count,
            100.0 * self._frames_analysed / meta.frame_count if meta.frame_count else 0.0,
        )
        if self.cfg.logging.profile:
            log.info("Stage profile:\n%s", self.profiler.format_table())
        return doc

    def _reset_state(self) -> None:
        self._stop_event.clear()
        self._error = None
        self._document = None
        self._frames_decoded = 0
        self._frames_analysed = 0
        self._faces_seen = 0
        self._expression_changes = 0
        self._emotion_results: Dict[Tuple[int, int], Tuple[str, float, str, float, bool]] = {}
        self.tracker.reset()
        with self._identity_lock:
            self.identities.reset()
        self.smoothers.reset()
        self.timeline.reset()
        self.profiler.reset()

    def _fail(self, exc: BaseException) -> None:
        if self._error is None:
            self._error = exc
        log.exception("Pipeline stage failed: %s", exc)
        self._stop_event.set()
        self.callbacks.fire("on_error", exc)

    def _put(self, q: "queue.Queue", item) -> None:
        """Blocking put that still honours the stop flag."""
        while not self._stop_event.is_set():
            try:
                q.put(item, timeout=0.2)
                return
            except queue.Full:
                continue

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release models, sessions and the database handle."""
        self.stop()
        for component in (self.detector, self.embedder, self.classifier):
            if component is not None:
                component.close()
        self.detector = self.embedder = self.classifier = None
        self._prepared = False
        self.gpu_monitor.close()
        empty_cache()


__all__ = ["AnalysisPipeline", "PipelineCallbacks"]
