"""Unit tests for the CPU-side logic (no GPU or model weights required).

    python -m unittest discover -s tests -v

Covers the parts where a silent regression would corrupt output: the timeline event
rules, emotion smoothing, tracker association, gallery matching, face alignment and
config round-tripping.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from configs.config import AppConfig, EmotionConfig, RecognitionConfig, TimelineConfig, TrackerConfig  # noqa: E402
from detector.base import Detection, distance2bbox, nms  # noqa: E402
from emotion.smoothing import EmotionSmoother  # noqa: E402
from pipeline.batching import BatchCollector  # noqa: E402
from pipeline.sampler import AdaptiveSampler  # noqa: E402
from pipeline.types import RawFrame  # noqa: E402
from recognition.arcface import average_embeddings  # noqa: E402
from recognition.matcher import GalleryMatcher, cosine_similarity  # noqa: E402
from timeline.events import TimelineDocument, VideoInfo  # noqa: E402
from timeline.timeline_engine import TimelineEngine  # noqa: E402
from tracking.byte_tracker import ByteTracker, iou_matrix  # noqa: E402
from tracking.track import IdentityRegistry  # noqa: E402
from utils.image_ops import ARCFACE_TEMPLATE, align_face, estimate_norm, letterbox, umeyama  # noqa: E402


# ---------------------------------------------------------------------------
class TestTimelineEngine(unittest.TestCase):
    """The core deliverable: one event per emotion *change*, not per frame."""

    def setUp(self) -> None:
        self.cfg = TimelineConfig(
            min_event_duration=0.3, merge_gap=0.6, close_track_after=1.5,
            include_unknown=False,
        )
        self.engine = TimelineEngine(self.cfg)

    def _observe(self, emotion: str, t: float, frame: int, conf: float = 0.9):
        return self.engine.observe(
            track_id=1, actor_id=7, actor_name="John", emotion=emotion,
            confidence=conf, timestamp=t, frame_index=frame, similarity=0.6,
        )

    def test_constant_emotion_produces_one_event(self) -> None:
        for i in range(20):
            self._observe("Happy", i * 0.133, i * 4)
        events = self.engine.events(include_open=True)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].emotion, "Happy")
        self.assertAlmostEqual(events[0].start, 0.0, places=3)
        self.assertGreater(events[0].end, 2.0)
        self.assertEqual(events[0].samples, 20)

    def test_emotion_change_closes_and_opens(self) -> None:
        for i in range(10):
            self._observe("Happy", i * 0.2, i * 4)
        closed = self._observe("Sad", 2.0, 40)
        self.assertIsNotNone(closed)
        self.assertEqual(closed.emotion, "Happy")
        self.assertAlmostEqual(closed.end, 2.0, places=3)
        for i in range(1, 10):
            self._observe("Sad", 2.0 + i * 0.2, 40 + i * 4)
        events = self.engine.events(include_open=True)
        self.assertEqual([e.emotion for e in events], ["Happy", "Sad"])

    def test_short_blip_is_absorbed(self) -> None:
        for i in range(10):
            self._observe("Happy", i * 0.2, i * 4)          # 0.0 .. 1.8
        self._observe("Angry", 2.0, 40)                      # single-sample blip
        for i in range(10):
            self._observe("Happy", 2.1 + i * 0.2, 44 + i * 4)
        events = self.engine.events(include_open=True)
        self.assertTrue(all(e.emotion == "Happy" for e in events), events)

    def test_same_emotion_after_gap_is_merged(self) -> None:
        for i in range(6):
            self._observe("Happy", i * 0.2, i * 4)
        self._observe("Sad", 1.2, 24)
        self._observe("Sad", 1.4, 28)
        for i in range(6):
            self._observe("Happy", 1.6 + i * 0.2, 32 + i * 4)
        emotions = [e.emotion for e in self.engine.events(include_open=True)]
        self.assertIn("Happy", emotions)

    def test_unknown_excluded_by_default(self) -> None:
        self.engine.observe(
            track_id=2, actor_id=-1, actor_name="Unknown", emotion="Neutral",
            confidence=0.8, timestamp=1.0, frame_index=30,
        )
        self.assertEqual(len(self.engine.events(include_open=True)), 0)

    def test_document_round_trip(self) -> None:
        for i in range(12):
            self._observe("Happy", i * 0.25, i * 4)
        info = VideoInfo(path="clip.mp4", duration=10.0, fps=30.0, processed_fps=7.5)
        doc = self.engine.build_document(info)
        restored = TimelineDocument.from_json(doc.to_json())
        self.assertEqual(len(restored.events), len(doc.events))
        self.assertEqual(restored.events[0].actor, "John")
        self.assertAlmostEqual(restored.video.fps, 30.0)
        self.assertEqual(restored.actors[0].name, "John")

    def test_expression_change_count_ignores_same_emotion_resume(self) -> None:
        # Happy -> Sad is a change; Sad ... gap ... Sad is not.
        for i in range(8):
            self._observe("Happy", i * 0.25, i * 4)
        for i in range(8):
            self._observe("Sad", 2.0 + i * 0.25, 60 + i * 4)
        for i in range(8):                       # long gap, then Sad again
            self._observe("Sad", 12.0 + i * 0.25, 360 + i * 4)
        doc = self.engine.build_document(
            VideoInfo(path="clip.mp4", duration=20.0, fps=30.0)
        )
        self.assertEqual(doc.actors[0].expression_changes, 1)
        self.assertEqual(doc.analysis.expression_changes, 1)
        # The top-level total must always agree with the per-actor totals.
        self.assertEqual(
            doc.as_dict()["expression_changes"],
            sum(a.expression_changes for a in doc.actors),
        )

    def test_actor_emotion_totals(self) -> None:
        for i in range(8):
            self._observe("Happy", i * 0.25, i * 4)
        for i in range(8):
            self._observe("Angry", 2.0 + i * 0.25, 60 + i * 4)
        doc = self.engine.build_document(
            VideoInfo(path="clip.mp4", duration=10.0, fps=30.0)
        )
        totals = doc.actors[0].emotions
        self.assertIn("Happy", totals)
        self.assertIn("Angry", totals)
        self.assertGreater(totals["Happy"], 0.0)


# ---------------------------------------------------------------------------
class TestEmotionSmoothing(unittest.TestCase):
    """Flicker must not reach the timeline."""

    def _probs(self, index: int, peak: float = 0.9) -> np.ndarray:
        p = np.full(8, (1.0 - peak) / 7.0, dtype=np.float32)
        p[index] = peak
        return p

    def test_single_outlier_does_not_switch(self) -> None:
        cfg = EmotionConfig(smoothing="hybrid", window=7, min_agree=4, confidence_threshold=0.4)
        smoother = EmotionSmoother(cfg)
        for _ in range(6):
            smoother.update("Happy", 0.8, self._probs(4, 0.8))
        result = smoother.update("Sad", 0.45, self._probs(6, 0.45))
        self.assertEqual(result.label, "Happy")
        self.assertFalse(result.changed)

    def test_sustained_change_switches(self) -> None:
        cfg = EmotionConfig(smoothing="majority", window=7, min_agree=4, confidence_threshold=0.4)
        smoother = EmotionSmoother(cfg)
        for _ in range(7):
            smoother.update("Happy", 0.8, self._probs(4, 0.8))
        labels = [smoother.update("Sad", 0.75, self._probs(6, 0.75)).label for _ in range(6)]
        self.assertEqual(labels[-1], "Sad")

    def test_low_confidence_is_ignored(self) -> None:
        cfg = EmotionConfig(smoothing="majority", window=5, min_agree=3, confidence_threshold=0.5)
        smoother = EmotionSmoother(cfg)
        for _ in range(5):
            smoother.update("Neutral", 0.7, self._probs(5, 0.7))
        for _ in range(5):
            smoother.update("Fear", 0.2, self._probs(3, 0.2))
        self.assertEqual(smoother.current, "Neutral")

    def test_ema_tracks_probability_vector(self) -> None:
        cfg = EmotionConfig(smoothing="ema", ema_alpha=0.5, confidence_threshold=0.3)
        smoother = EmotionSmoother(cfg)
        smoother.update("Happy", 0.9, self._probs(4, 0.9))
        for _ in range(6):
            out = smoother.update("Surprise", 0.9, self._probs(7, 0.9))
        self.assertEqual(out.label, "Surprise")


# ---------------------------------------------------------------------------
class TestTracker(unittest.TestCase):
    def _det(self, x: float, y: float, size: float = 80.0, score: float = 0.9) -> Detection:
        return Detection(
            bbox=np.array([x, y, x + size, y + size], dtype=np.float32), score=score
        )

    def test_track_id_is_stable_under_motion(self) -> None:
        tracker = ByteTracker(TrackerConfig(min_hits=1))
        ids = set()
        for step in range(12):
            tracks = tracker.update([self._det(100 + step * 6, 100)])
            ids.update(t.track_id for t in tracks)
        self.assertEqual(len(ids), 1, f"expected one stable track, got {ids}")

    def test_two_faces_get_distinct_ids(self) -> None:
        tracker = ByteTracker(TrackerConfig(min_hits=1))
        for step in range(8):
            tracks = tracker.update(
                [self._det(80 + step * 4, 90), self._det(500 - step * 4, 120)]
            )
        self.assertEqual(len({t.track_id for t in tracks}), 2)

    def test_lost_track_is_recovered(self) -> None:
        tracker = ByteTracker(TrackerConfig(min_hits=1, max_time_lost=30))
        for step in range(6):
            tracks = tracker.update([self._det(100 + step * 5, 100)])
        first_id = tracks[0].track_id
        for _ in range(3):                       # occlusion: nothing detected
            tracker.update([])
        tracks = tracker.update([self._det(145, 100)])
        self.assertEqual(tracks[0].track_id, first_id)

    def test_iou_matrix(self) -> None:
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[0, 0, 10, 10], [10, 10, 20, 20]], dtype=np.float32)
        iou = iou_matrix(a, b)
        self.assertAlmostEqual(float(iou[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(iou[0, 1]), 0.0, places=5)


# ---------------------------------------------------------------------------
class TestMatcherAndIdentity(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.vectors = rng.standard_normal((3, 512)).astype(np.float32)
        self.vectors /= np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.matcher = GalleryMatcher(
            torch.device("cpu"), similarity_threshold=0.38, margin_threshold=0.05
        )
        self.matcher.set_gallery(
            self.vectors, [1, 2, 3], {1: "John", 2: "Mary", 3: "Sam"}
        )

    def test_exact_match(self) -> None:
        result = self.matcher.match_one(self.vectors[1])
        self.assertEqual(result.name, "Mary")
        self.assertGreater(result.similarity, 0.99)
        self.assertTrue(result.is_known)

    def test_unknown_below_threshold(self) -> None:
        rng = np.random.default_rng(99)
        query = rng.standard_normal(512).astype(np.float32)
        query /= np.linalg.norm(query)
        result = self.matcher.match_one(query)
        self.assertFalse(result.is_known)
        self.assertEqual(result.name, "Unknown")

    def test_empty_gallery_returns_unknown(self) -> None:
        matcher = GalleryMatcher(torch.device("cpu"))
        result = matcher.match_one(self.vectors[0])
        self.assertFalse(result.is_known)

    def test_average_embeddings_is_unit_norm(self) -> None:
        centroid = average_embeddings(self.vectors, [1.0, 0.5, 0.25])
        self.assertAlmostEqual(float(np.linalg.norm(centroid)), 1.0, places=5)

    def test_identity_voting_locks_after_min_votes(self) -> None:
        registry = IdentityRegistry(RecognitionConfig(vote_window=5, min_votes=3))
        self.assertTrue(registry.needs_recognition(1))
        for i in range(3):
            registry.update(1, self.matcher.match_one(self.vectors[0]), i, i * 0.1)
        record = registry.get(1)
        self.assertTrue(record.locked)
        self.assertEqual(record.name, "John")

    def test_reid_scheduling_backs_off(self) -> None:
        cfg = RecognitionConfig(vote_window=3, min_votes=2, reid_interval=45)
        registry = IdentityRegistry(cfg)
        for i in range(3):
            registry.update(1, self.matcher.match_one(self.vectors[0]), i, i * 0.1)
        self.assertFalse(registry.needs_recognition(1))
        registry.tick([1] * 1)
        for _ in range(50):
            registry.tick([1])
        self.assertTrue(registry.needs_recognition(1))

    def test_cosine_similarity_helper(self) -> None:
        self.assertAlmostEqual(cosine_similarity(self.vectors[0], self.vectors[0]), 1.0, places=5)
        self.assertAlmostEqual(cosine_similarity(np.zeros(4), np.ones(4)), 0.0, places=6)


# ---------------------------------------------------------------------------
class TestImageOps(unittest.TestCase):
    def test_umeyama_recovers_similarity_transform(self) -> None:
        src = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], dtype=np.float32)
        angle = np.deg2rad(30.0)
        rot = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        dst = (src @ rot.T) * 2.5 + np.array([7.0, -3.0])
        matrix = umeyama(src, dst)
        recovered = (np.hstack([src, np.ones((5, 1))]) @ matrix.T)[:, :2]
        np.testing.assert_allclose(recovered, dst, atol=1e-4)

    def test_estimate_norm_maps_to_template(self) -> None:
        landmarks = ARCFACE_TEMPLATE * 2.0 + 30.0
        matrix = estimate_norm(landmarks, 112)
        mapped = (np.hstack([landmarks, np.ones((5, 1))]) @ matrix.T)
        np.testing.assert_allclose(mapped, ARCFACE_TEMPLATE, atol=1e-3)

    def test_align_face_output_shape(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        crop = align_face(image, ARCFACE_TEMPLATE * 2 + 100, 112)
        self.assertEqual(crop.shape, (112, 112, 3))

    def test_letterbox_preserves_aspect(self) -> None:
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        padded, scale, _ = letterbox(image, 640)
        self.assertEqual(padded.shape, (640, 640, 3))
        self.assertAlmostEqual(scale, 1.0, places=5)
        image2 = np.zeros((1080, 1920, 3), dtype=np.uint8)
        _, scale2, _ = letterbox(image2, 640)
        self.assertAlmostEqual(scale2, 640 / 1920, places=5)

    def test_nms_removes_duplicates(self) -> None:
        boxes = np.array(
            [[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], dtype=np.float32
        )
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        keep = nms(boxes, scores, 0.4)
        self.assertEqual(len(keep), 2)
        self.assertIn(0, keep)
        self.assertIn(2, keep)

    def test_distance2bbox(self) -> None:
        points = np.array([[10.0, 10.0]], dtype=np.float32)
        distance = np.array([[2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
        box = distance2bbox(points, distance)
        np.testing.assert_allclose(box, [[8.0, 7.0, 14.0, 15.0]])


# ---------------------------------------------------------------------------
class TestSamplerAndBatching(unittest.TestCase):
    def _frame(self, index: int, value: int = 0) -> RawFrame:
        image = np.full((180, 320, 3), value, dtype=np.uint8)
        return RawFrame(index=index, timestamp=index / 30.0, image=image)

    def test_stride_selects_one_in_four(self) -> None:
        cfg = AppConfig()
        cfg.sampling.adaptive = False
        cfg.sampling.scene_cut_enabled = False
        sampler = AdaptiveSampler(cfg.sampling, cfg.video)
        selected = [i for i in range(40) if sampler.consider(self._frame(i)) is not None]
        self.assertEqual(selected, list(range(0, 40, 4)))

    def test_scene_cut_forces_immediate_sample(self) -> None:
        cfg = AppConfig()
        cfg.sampling.adaptive = False
        sampler = AdaptiveSampler(cfg.sampling, cfg.video)
        for i in range(6):
            sampler.consider(self._frame(i, value=20))
        # A drastically different frame between strides must still be analysed.
        result = sampler.consider(self._frame(6, value=230))
        self.assertIsNotNone(result)
        self.assertTrue(result.scene_cut)

    def test_adaptive_stride_relaxes_when_calm(self) -> None:
        cfg = AppConfig()
        cfg.sampling.adaptive = True
        cfg.sampling.calm_window = 3
        cfg.sampling.scene_cut_enabled = False
        sampler = AdaptiveSampler(cfg.sampling, cfg.video)
        base = sampler.stride
        for i in range(60):
            if sampler.consider(self._frame(i)) is not None:
                sampler.report_activity(False)
        self.assertGreater(sampler.stride, base)

    def test_batch_collector_groups_items(self) -> None:
        import queue

        q: "queue.Queue" = queue.Queue()
        for i in range(10):
            q.put(i)
        q.put(None)
        collector = BatchCollector(q, batch_size=4, max_latency_ms=5)
        sizes = []
        while not collector.finished:
            batch = collector.next_batch()
            if batch:
                sizes.append(len(batch))
        self.assertEqual(sum(sizes), 10)
        self.assertLessEqual(max(sizes), 4)


# ---------------------------------------------------------------------------
class TestConfig(unittest.TestCase):
    def test_yaml_round_trip(self) -> None:
        cfg = AppConfig()
        cfg.sampling.frame_stride = 3
        cfg.recognition.similarity_threshold = 0.42
        cfg.gpu.fp16 = False
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.yaml"
            cfg.save(path)
            restored = AppConfig.load(path)
        self.assertEqual(restored.sampling.frame_stride, 3)
        self.assertAlmostEqual(restored.recognition.similarity_threshold, 0.42)
        self.assertFalse(restored.gpu.fp16)

    def test_dotted_override(self) -> None:
        cfg = AppConfig().override(
            {"sampling.frame_stride": 8, "emotion.window": 11, "gpu.fp16": False}
        )
        self.assertEqual(cfg.sampling.frame_stride, 8)
        self.assertEqual(cfg.emotion.window, 11)
        self.assertFalse(cfg.gpu.fp16)

    def test_unknown_key_raises(self) -> None:
        with self.assertRaises(KeyError):
            AppConfig().override({"sampling.does_not_exist": 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
