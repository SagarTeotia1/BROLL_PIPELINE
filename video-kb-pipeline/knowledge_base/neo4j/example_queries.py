"""
Example Cypher queries for video knowledge graph.
Run via: await query_graph(cypher, params, database)
"""
from __future__ import annotations

# Find all scenes where P1 appears
SCENES_FOR_PERSON = """
MATCH (p:Person {pid: $pid, video_id: $video_id})-[:APPEARS_IN]->(f:Frame)-[:BELONGS_TO_SCENE]->(s:Scene)
RETURN DISTINCT s.scene_id AS scene_id, s.beat_type AS beat_type, s.scene_mood AS scene_mood
ORDER BY s.ref_id
"""

# Trace an object across scenes (e.g. a document passed between people)
OBJECT_CHAIN = """
MATCH path = (o:Object {name: $object_name, video_id: $video_id})<-[r]-(f:Frame)-[:BELONGS_TO_SCENE]->(s:Scene)
RETURN s.scene_id AS scene_id, type(r) AS relation, f.timestamp_s AS timestamp
ORDER BY f.timestamp_s
"""

# Multi-hop: who interacted with the same object as P1?
SHARED_OBJECT_PERSONS = """
MATCH (p1:Person {pid: $pid, video_id: $video_id})-[r1]->(obj:Object)<-[r2]-(p2:Person)
WHERE p1 <> p2
RETURN p2.pid AS other_person, obj.name AS shared_object, type(r1) AS p1_relation, type(r2) AS p2_relation
"""

# Scene continuity chain
SCENE_CHAIN = """
MATCH path = (s1:Scene {video_id: $video_id})-[:FOLLOWS*1..10]->(sN:Scene)
WHERE s1.beat_type = $start_beat
RETURN [n IN nodes(path) | n.scene_id] AS scene_chain,
       [n IN nodes(path) | n.beat_type] AS beat_chain
LIMIT 5
"""

# Person emotion arc across video
PERSON_EMOTION_ARC = """
MATCH (p:Person {pid: $pid, video_id: $video_id})-[r:APPEARS_IN]->(f:Frame)
WHERE r.emotion IS NOT NULL
RETURN f.timestamp_s AS t, r.emotion AS emotion, r.story_role AS role
ORDER BY f.timestamp_s
"""
