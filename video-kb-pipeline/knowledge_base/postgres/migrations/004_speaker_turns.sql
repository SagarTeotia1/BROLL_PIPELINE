-- Level-2 Stage 0: speaker diarization turns, fused with ArcFace identity.
--
-- One row per diarized speech turn. cluster_label is the raw pyannote
-- cluster id (e.g. "SPEAKER_00") before identity fusion; person_id is the
-- resolved cast identity (fused via face co-presence in updater.py), and may
-- be NULL if fusion could not confidently resolve one from the closed cast
-- set. resolution_method + confidence record how the person_id was decided
-- so L5 reasoning can weight trust per-turn instead of per-video.
CREATE TABLE IF NOT EXISTS speaker_turns (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id          UUID REFERENCES videos(id) ON DELETE CASCADE,
  cluster_label      TEXT NOT NULL,
  person_id         UUID REFERENCES persons(id),
  start_time        FLOAT NOT NULL,
  end_time          FLOAT NOT NULL,
  confidence        FLOAT,
  resolution_method TEXT NOT NULL DEFAULT 'unresolved'
    CHECK (resolution_method IN ('face_majority', 'single_candidate', 'llm_tiebreak', 'unresolved'))
);

CREATE INDEX IF NOT EXISTS idx_speaker_turns_video_time ON speaker_turns(video_id, start_time);
CREATE INDEX IF NOT EXISTS idx_speaker_turns_video_person ON speaker_turns(video_id, person_id);
