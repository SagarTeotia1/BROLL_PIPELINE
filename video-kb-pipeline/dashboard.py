"""Video Knowledge Base Dashboard — shows ALL data from every table/store.

Run:
    pip install streamlit plotly pandas pillow python-dotenv psycopg2-binary boto3 neo4j pinecone sentence-transformers
    streamlit run dashboard.py
"""
from __future__ import annotations

import json
import sys
import uuid as _uuid_mod
from pathlib import Path

_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))

import streamlit as st

st.set_page_config(page_title="Video KB", page_icon="🎬", layout="wide")

st.markdown("""<style>
/* Tighten sidebar */
[data-testid="stSidebar"] { min-width: 220px; max-width: 260px; }
/* Reduce tab bar font size so all tabs fit */
[data-testid="stTabs"] button { font-size: 11px; padding: 4px 8px; }
/* Tighter metrics */
[data-testid="metric-container"] { padding: 6px 10px; }
/* Reduce top padding */
.block-container { padding-top: 1rem; }
</style>""", unsafe_allow_html=True)

from dotenv import load_dotenv
load_dotenv(_HERE / ".env")

from shared.config import settings

# ─────────────────────────────────────────────────────────────────────────────
# DB / storage helpers (sync psycopg2 — no event loop conflict with Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _conn():
    import psycopg2
    import psycopg2.extras
    url = settings.asyncpg_url  # already plain postgresql:// form
    conn = psycopg2.connect(url)
    conn.autocommit = True
    return conn


import re as _re

def _pg2(sql: str) -> str:
    """Convert asyncpg $N / $N::type placeholders to psycopg2 %s."""
    return _re.sub(r'\$\d+(?:::\w+(?:\[\])?)?', '%s', sql)


def q(sql, *args):
    import psycopg2.extras
    try:
        conn = _conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_pg2(sql), args or None)
            return [dict(r) for r in cur.fetchall()]
    except Exception as exc:
        # reconnect on stale connection then retry once
        try:
            _conn.clear()
            import psycopg2.extras
            conn2 = _conn()
            with conn2.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(_pg2(sql), args or None)
                return [dict(r) for r in cur.fetchall()]
        except Exception as exc2:
            st.error(f"Query failed: {exc2}")
            return []


def q1(sql, *args):
    rows = q(sql, *args)
    return rows[0] if rows else {}


_TTL_FAST = 60      # status / jobs — refresh often
_TTL_DATA = 300     # immutable pipeline output — stable
_TTL_HEAVY = 600    # large tables — rarely change


@st.cache_data(ttl=_TTL_FAST)
def videos():
    return q("SELECT * FROM videos ORDER BY created_at DESC LIMIT 100")


@st.cache_data(ttl=_TTL_FAST)
def jobs(vid):
    return q("SELECT * FROM processing_jobs WHERE video_id=%s::uuid ORDER BY level", vid)


# ── Lightweight counts — used by Overview (no row data) ──────────────────────
@st.cache_data(ttl=_TTL_DATA)
def counts(vid):
    return q1("""
        SELECT
            (SELECT COUNT(*) FROM chunks              WHERE video_id=%s::uuid) AS chunks,
            (SELECT COUNT(*) FROM shots               WHERE video_id=%s::uuid) AS shots,
            (SELECT COUNT(*) FROM keyframes           WHERE video_id=%s::uuid) AS keyframes,
            (SELECT COUNT(*) FROM transcript_segments WHERE video_id=%s::uuid) AS transcript_segs,
            (SELECT COUNT(*) FROM persons             WHERE video_id=%s::uuid) AS persons,
            (SELECT COUNT(*) FROM face_appearances    WHERE video_id=%s::uuid) AS face_appearances,
            (SELECT COUNT(*) FROM face_timeline_events WHERE video_id=%s::uuid) AS timeline_events,
            (SELECT COUNT(*) FROM speaker_turns       WHERE video_id=%s::uuid) AS speaker_turns,
            (SELECT COUNT(*) FROM color_grades        WHERE video_id=%s::uuid) AS color_grades,
            (SELECT COUNT(*) FROM frame_analyses      WHERE video_id=%s::uuid) AS frame_analyses,
            (SELECT COUNT(*) FROM searchable_facts    WHERE video_id=%s::uuid) AS facts,
            (SELECT COUNT(*) FROM kg_nodes            WHERE video_id=%s::uuid) AS kg_nodes,
            (SELECT COUNT(*) FROM kg_edges            WHERE video_id=%s::uuid) AS kg_edges
    """, vid, vid, vid, vid, vid, vid, vid, vid, vid, vid, vid, vid, vid)


# ── Paginated fetchers ────────────────────────────────────────────────────────
@st.cache_data(ttl=_TTL_DATA)
def chunks(vid):
    return q("SELECT * FROM chunks WHERE video_id=%s::uuid ORDER BY chunk_index", vid)


@st.cache_data(ttl=_TTL_DATA)
def shots(vid):
    return q("SELECT * FROM shots WHERE video_id=%s::uuid ORDER BY shot_index", vid)


@st.cache_data(ttl=_TTL_DATA)
def keyframes(vid, limit=500, offset=0):
    return q("SELECT * FROM keyframes WHERE video_id=%s::uuid ORDER BY timestamp_s LIMIT %s OFFSET %s",
             vid, limit, offset)


@st.cache_data(ttl=_TTL_DATA)
def transcript(vid, limit=2000, offset=0):
    return q("SELECT * FROM transcript_segments WHERE video_id=%s::uuid ORDER BY start_time LIMIT %s OFFSET %s",
             vid, limit, offset)


@st.cache_data(ttl=_TTL_DATA)
def persons(vid):
    return q("SELECT id, pid, display_name FROM persons WHERE video_id=%s::uuid ORDER BY pid", vid)


@st.cache_data(ttl=_TTL_HEAVY)
def face_appearances(vid, limit=2000, offset=0):
    return q("""
        SELECT fa.frame_index, fa.timestamp_s, fa.track_id,
               fa.bbox, fa.emotion, fa.emotion_conf,
               p.pid, p.display_name
        FROM face_appearances fa
        LEFT JOIN persons p ON p.id = fa.person_id
        WHERE fa.video_id=%s::uuid ORDER BY fa.timestamp_s
        LIMIT %s OFFSET %s
    """, vid, limit, offset)


@st.cache_data(ttl=_TTL_DATA)
def face_timeline(vid):
    return q("""
        SELECT fte.emotion, fte.start_time, fte.end_time, fte.confidence,
               p.pid, p.display_name
        FROM face_timeline_events fte
        LEFT JOIN persons p ON p.id = fte.person_id
        WHERE fte.video_id=%s::uuid ORDER BY fte.start_time
    """, vid)


@st.cache_data(ttl=_TTL_DATA)
def speaker_turns(vid):
    return q("""
        SELECT st.cluster_label, st.start_time, st.end_time, st.confidence,
               st.resolution_method, p.pid, p.display_name
        FROM speaker_turns st
        LEFT JOIN persons p ON p.id = st.person_id
        WHERE st.video_id=%s::uuid ORDER BY st.start_time
    """, vid)


@st.cache_data(ttl=_TTL_DATA)
def color_grades(vid):
    rows = q("SELECT * FROM color_grades WHERE video_id=%s::uuid ORDER BY timestamp_s", vid)
    for r in rows:
        if isinstance(r.get("parameters"), str):
            r["parameters"] = json.loads(r["parameters"])
        r["style_tags"] = list(r.get("style_tags") or [])
    return rows


# frame_analyses: list view strips qwen_output (heavy JSON); detail loaded on demand
@st.cache_data(ttl=_TTL_HEAVY)
def frame_analyses_list(vid, limit=100, offset=0):
    rows = q("""
        SELECT fa.id, fa.keyframe_id, fa.scene_id, fa.scene_change,
               fa.caption, fa.beat_type, fa.scene_mood, fa.tension_level, fa.tags,
               kf.timestamp_s, kf.r2_key, kf.frame_index
        FROM frame_analyses fa
        JOIN keyframes kf ON kf.id = fa.keyframe_id
        WHERE fa.video_id=%s::uuid ORDER BY kf.timestamp_s
        LIMIT %s OFFSET %s
    """, vid, limit, offset)
    for r in rows:
        r["tags"] = list(r.get("tags") or [])
    return rows


@st.cache_data(ttl=_TTL_HEAVY)
def frame_analysis_detail(fa_id):
    """Load full qwen_output for one frame — called only when user expands it."""
    row = q1("SELECT qwen_output FROM frame_analyses WHERE id=%s::uuid", fa_id)
    raw = row.get("qwen_output")
    if raw is None:
        return {}
    return json.loads(raw) if isinstance(raw, str) else raw


# Keep old name for backward compat with Player tab (needs r2_key + timestamp for strip)
@st.cache_data(ttl=_TTL_HEAVY)
def frame_analyses(vid):
    """Full fetch — only used by Player keyframe strip (needs all for timeline JS)."""
    rows = q("""
        SELECT fa.id, fa.keyframe_id, fa.scene_id, fa.scene_change,
               fa.caption, fa.beat_type, fa.scene_mood, fa.tension_level, fa.tags,
               kf.timestamp_s, kf.r2_key, kf.frame_index
        FROM frame_analyses fa
        JOIN keyframes kf ON kf.id = fa.keyframe_id
        WHERE fa.video_id=%s::uuid ORDER BY kf.timestamp_s
    """, vid)
    for r in rows:
        r["tags"] = list(r.get("tags") or [])
    return rows


@st.cache_data(ttl=_TTL_HEAVY)
def facts(vid, limit=500, offset=0):
    return q("SELECT id, fact_text, timestamp_s, pinecone_id FROM searchable_facts "
             "WHERE video_id=%s::uuid ORDER BY timestamp_s LIMIT %s OFFSET %s",
             vid, limit, offset)


@st.cache_data(ttl=_TTL_HEAVY)
def kg_nodes(vid):
    rows = q("SELECT * FROM kg_nodes WHERE video_id=%s::uuid ORDER BY node_type, label", vid)
    for r in rows:
        if isinstance(r.get("properties"), str):
            r["properties"] = json.loads(r["properties"])
    return rows


@st.cache_data(ttl=_TTL_HEAVY)
def kg_edges(vid, limit=2000):
    rows = q("""
        SELECT ke.relation, ke.weight,
               src.node_type AS src_type, src.label AS src_label,
               tgt.node_type AS tgt_type, tgt.label AS tgt_label,
               ke.properties
        FROM kg_edges ke
        JOIN kg_nodes src ON src.id = ke.source_id
        JOIN kg_nodes tgt ON tgt.id = ke.target_id
        WHERE ke.video_id=%s::uuid ORDER BY ke.relation
        LIMIT %s
    """, vid, limit)
    for r in rows:
        if isinstance(r.get("properties"), str):
            r["properties"] = json.loads(r["properties"])
    return rows


@st.cache_resource
def pinecone_idx():
    from pinecone import Pinecone
    return Pinecone(api_key=settings.PINECONE_API_KEY).Index(settings.PINECONE_INDEX_NAME)


@st.cache_resource
def neo4j_driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))


def neo4j_q(cypher, params=None):
    with neo4j_driver().session(database=settings.NEO4J_DATABASE) as s:
        return [dict(r) for r in s.run(cypher, params or {})]


@st.cache_data(ttl=300)
def r2_url(key, exp=3600):
    if not key:
        return None
    try:
        import boto3
        s3 = boto3.client("s3",
            endpoint_url=settings.R2_ENDPOINT_URL,
            aws_access_key_id=settings.R2_ACCESS_KEY_ID,
            aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
            region_name="auto")
        return s3.generate_presigned_url("get_object",
            Params={"Bucket": settings.R2_BUCKET_NAME, "Key": key}, ExpiresIn=exp)
    except Exception:
        return None


def ft(s):
    if s is None: return "—"
    m, sec = divmod(int(s), 60)
    return f"{m}:{sec:02d}"


def jshow(d):
    if d:
        st.json(d)


# ─────────────────────────────────────────────────────────────────────────────
# Graph viz helpers (pyvis)
# ─────────────────────────────────────────────────────────────────────────────
_NODE_COLORS = {
    "Video":   "#f59e0b",
    "Chunk":   "#3b82f6",
    "Shot":    "#8b5cf6",
    "Frame":   "#10b981",
    "Person":  "#f97316",
    "Scene":   "#06b6d4",
    "Object":  "#ef4444",
    "Theme":   "#ec4899",
    "Emotion": "#facc15",
}
_EDGE_COLORS = {
    "CONTAINS":    "#6b7280",
    "APPEARS_IN":  "#f97316",
    "FOLLOWS":     "#3b82f6",
    "CAUSES":      "#ef4444",
    "HAS_EMOTION": "#facc15",
    "RELATES_TO":  "#10b981",
    "HAS_ANALYSIS":"#8b5cf6",
    "PRESENT_IN":  "#ec4899",
}


def build_pyvis(nodes, edges, height=580, directed=True, physics=True):
    """Return pyvis HTML string for embedding."""
    try:
        from pyvis.network import Network
    except ImportError:
        return "<p style='color:#9ca3af'>Install pyvis: <code>pip install pyvis</code></p>"

    net = Network(
        height=f"{height}px", width="100%",
        bgcolor="#0e1117", font_color="#e5e7eb",
        directed=directed,
    )
    net.set_options("""
    {
      "physics": {
        "enabled": """ + str(physics).lower() + """,
        "barnesHut": {"gravitationalConstant": -8000, "springLength": 120, "springConstant": 0.04},
        "minVelocity": 0.75
      },
      "edges": {"smooth": {"type": "dynamic"}, "width": 1.5, "arrows": {"to": {"enabled": true, "scaleFactor": 0.6}}},
      "nodes": {"font": {"size": 11}},
      "interaction": {"hover": true, "tooltipDelay": 100, "zoomView": true, "dragView": true}
    }""")

    for n in nodes:
        color = _NODE_COLORS.get(n.get("node_type") or n.get("type") or "", "#6b7280")
        label = (n.get("label") or n.get("id") or "")[:28]
        tooltip = f"{n.get('node_type','')}\n{n.get('label','')}\n{str(n.get('properties',''))[:120]}"
        net.add_node(
            str(n["id"]), label=label, title=tooltip,
            color={"background": color, "border": "#fff3", "highlight": {"background": color, "border": "#fff"}},
            size=16 if n.get("node_type") in ("Video", "Person", "Scene") else 10,
        )

    for e in edges:
        color = _EDGE_COLORS.get(e.get("relation") or "", "#6b7280")
        net.add_edge(
            str(e["source_id"]), str(e["target_id"]),
            label=e.get("relation") or "",
            title=e.get("relation") or "",
            color=color, width=1.5,
        )

    return net.generate_html(notebook=False)


def _legend_html(mapping: dict) -> str:
    items = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:4px;margin:2px 6px 2px 0">'
        f'<span style="width:12px;height:12px;border-radius:50%;background:{c};display:inline-block"></span>'
        f'<span style="font-size:11px;color:#d1d5db">{k}</span></span>'
        for k, c in mapping.items()
    )
    return f'<div style="margin:6px 0 10px">{items}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🎬 Video KB")

try:
    vids = videos()
except Exception as e:
    st.error(f"DB error: {e}")
    st.stop()

if not vids:
    st.warning("No videos in DB.")
    st.stop()

labels = {f"{v['id']}  ({ft(v['duration_s'])})": str(v["id"]) for v in vids}
sel = st.sidebar.selectbox("Video", list(labels.keys()))
VID = labels[sel]
VM = next((v for v in vids if str(v["id"]) == VID), {})

st.sidebar.markdown(f"""
**Duration:** {ft(VM.get('duration_s'))}
**FPS:** {VM.get('fps', '?')}
**Resolution:** {VM.get('width')}×{VM.get('height')}
**Path:** `{str(VM.get('path',''))[:60]}`
**ID:** `{VID}`
""")

if st.sidebar.button("🔄 Clear cache"):
    st.cache_data.clear()
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Status bar
# ─────────────────────────────────────────────────────────────────────────────
_jobs = jobs(VID)
jmap = {j["level"]: j for j in _jobs}
icons = {"done": "🟢", "running": "🟡", "failed": "🔴", "queued": "⚪"}
c1,c2,c3 = st.columns(3)
for col, lvl, name in [
    (c1, 1, "L1 ASHFS+Whisper"),
    (c2, 2, "L2 Face+Color"),
    (c3, 3, "L3 Qwen+Graph"),
]:
    j = jmap.get(lvl)
    col.metric(name, f"{icons.get(j['status'],'⚪')} {j['status'].upper()}" if j else "⚪ NOT RUN")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
T = st.tabs(["▶ Player", "📊 Overview", "🗂 Chunks", "🎬 Shots", "🖼 Keyframes",
             "📝 Transcript", "👤 Persons", "😊 Face Appearances", "😐 Emotions",
             "🗣 Diarization", "🎨 Color Grades", "🧠 Frame Analysis", "💡 Facts", "🕸 KG Nodes",
             "🔗 KG Edges", "🌐 Neo4j", "🔍 Search"])

(tab_player, tab_overview, tab_chunks, tab_shots, tab_kf,
 tab_trans, tab_persons, tab_faces, tab_emotions,
 tab_diarization, tab_color, tab_analysis, tab_facts, tab_kgnodes,
 tab_kgedges, tab_neo4j, tab_search) = T


# ══════════════════════════════════════════════════════════════════════════════
# ▶ PLAYER
# ══════════════════════════════════════════════════════════════════════════════
with tab_player:
    st.header("Video Player")
    path = VM.get("path", "")
    r2k  = VM.get("r2_key")
    vsrc = path if path.startswith("http") else (r2_url(r2k, 7200) if r2k else None)

    if not vsrc:
        st.warning("No public URL or R2 key for this video.")
    else:
        _shots  = shots(VID)
        _kfs    = keyframes(VID)
        _fas    = frame_analyses(VID)
        fa_map  = {str(a["keyframe_id"]): a for a in _fas}
        dur     = VM.get("duration_s") or 1

        shots_js = json.dumps([{"idx": s["shot_index"], "start": s["start_time"] or 0,
            "end": s["end_time"] or 0, "type": s["shot_type"] or "unknown"} for s in _shots])
        kf_js = json.dumps([{"t": kf["timestamp_s"],
            "label": (fa_map.get(str(kf["id"]),{}).get("caption") or f"Frame {kf['frame_index']}")[:60],
            "scene": fa_map.get(str(kf["id"]),{}).get("scene_id") or "",
            "mood":  fa_map.get(str(kf["id"]),{}).get("scene_mood") or ""} for kf in _kfs])

        import streamlit.components.v1 as components
        components.html(f"""
<!DOCTYPE html><html><head><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,sans-serif;background:#0e1117;color:#fafafa;padding:10px}}
video{{width:100%;max-height:400px;background:#000;border-radius:8px}}
#tl{{position:relative;height:44px;background:#1a1d27;border-radius:6px;margin-top:8px;overflow:hidden;cursor:pointer}}
#pb{{position:absolute;top:0;left:0;height:100%;background:rgba(99,102,241,.2);pointer-events:none}}
.sb{{position:absolute;top:0;height:100%;border-right:1px solid #0e1117;cursor:pointer}}
#kr{{position:relative;height:24px;background:#111;border-radius:4px;margin-top:3px;overflow:hidden}}
.km{{position:absolute;top:2px;width:3px;height:20px;background:#f59e0b;border-radius:2px;cursor:pointer}}
.km:hover{{background:#fcd34d}}
#info{{display:flex;gap:12px;margin-top:6px;font-size:12px;color:#9ca3af;align-items:center}}
#ct{{font-size:15px;font-weight:bold;color:#e5e7eb;font-variant-numeric:tabular-nums}}
#tip{{position:fixed;background:#1e2130;border:1px solid #374151;border-radius:6px;padding:7px 10px;
      font-size:11px;max-width:260px;pointer-events:none;display:none;z-index:999;white-space:pre-line}}
#btns{{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px;max-height:140px;overflow-y:auto}}
.jb{{background:#1f2937;border:1px solid #374151;color:#d1d5db;font-size:10px;padding:3px 7px;
     border-radius:4px;cursor:pointer}}
.jb:hover{{background:#374151;color:#fff}}
.jb.on{{border-color:#6366f1;background:#312e81;color:#fff}}
#tabs2{{display:flex;gap:4px;margin-top:10px}}
.tb{{padding:4px 10px;font-size:11px;border-radius:4px;border:1px solid #374151;
     background:#1f2937;color:#9ca3af;cursor:pointer}}
.tb.on{{background:#6366f1;color:#fff;border-color:#6366f1}}
.panel{{display:none}}.panel.on{{display:block}}
</style></head><body>
<video id="vid" controls crossorigin="anonymous"><source src="{vsrc}" type="video/mp4"></video>
<div id="tl"><div id="pb"></div></div>
<div id="kr"></div>
<div id="info"><div id="ct">0:00</div><div id="si">—</div><div id="mi"></div></div>
<div id="tip"></div>
<div id="tabs2">
  <button class="tb on" onclick="sp('shots')">Shots ({len(_shots)})</button>
  <button class="tb" onclick="sp('kf')">Keyframes ({len(_kfs)})</button>
</div>
<div id="p-shots" class="panel on"><div id="sb" class="btns" style="display:flex;flex-wrap:wrap;gap:5px;margin-top:6px;max-height:130px;overflow-y:auto"></div></div>
<div id="p-kf" class="panel"><div id="kb" class="btns" style="display:flex;flex-wrap:wrap;gap:5px;margin-top:6px;max-height:130px;overflow-y:auto"></div></div>
<script>
const vid=document.getElementById('vid'),pb=document.getElementById('pb'),
tl=document.getElementById('tl'),kr=document.getElementById('kr'),
tip=document.getElementById('tip'),ct=document.getElementById('ct'),
si=document.getElementById('si'),mi=document.getElementById('mi');
const DUR={dur},SHOTS={shots_js},KFS={kf_js};
const SC={{hard_cut:'#6366f1',soft:'#10b981',motion:'#f59e0b',unknown:'#6b7280'}};
function ft(s){{const m=Math.floor(s/60),x=Math.floor(s%60);return m+':'+String(x).padStart(2,'0')}}
function tip_show(e,t){{tip.innerHTML=t;tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY-8)+'px'}}
function tip_hide(){{tip.style.display='none'}}
document.addEventListener('mousemove',e=>{{if(tip.style.display==='block'){{tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY-8)+'px'}}}});
SHOTS.forEach(s=>{{
  const d=document.createElement('div');d.className='sb';
  d.style.left=((s.start/DUR)*100)+'%';d.style.width=(((s.end-s.start)/DUR)*100)+'%';
  d.style.background=SC[s.type]||'#6b7280';
  d.addEventListener('mouseenter',e=>tip_show(e,'Shot '+s.idx+'\\n'+s.type+'\\n'+ft(s.start)+'→'+ft(s.end)));
  d.addEventListener('mouseleave',tip_hide);
  d.addEventListener('click',e=>{{vid.currentTime=s.start;e.stopPropagation()}});
  tl.appendChild(d);
  const btn=document.createElement('button');btn.className='jb';btn.dataset.s=s.start;
  btn.textContent='S'+String(s.idx).padStart(3,'0')+' '+ft(s.start)+' ['+s.type+']';
  btn.onclick=()=>vid.currentTime=s.start;
  document.getElementById('sb').appendChild(btn);
}});
KFS.forEach(kf=>{{
  const m=document.createElement('div');m.className='km';
  m.style.left=((kf.t/DUR)*100)+'%';
  m.addEventListener('mouseenter',e=>tip_show(e,ft(kf.t)+'\\n'+(kf.label||'')+(kf.mood?'\\n🎭 '+kf.mood:'')+(kf.scene?'\\n🎬 '+kf.scene:'')));
  m.addEventListener('mouseleave',tip_hide);m.addEventListener('click',()=>vid.currentTime=kf.t);kr.appendChild(m);
  const btn=document.createElement('button');btn.className='jb';btn.dataset.t=kf.t;
  btn.textContent=ft(kf.t)+' — '+(kf.label||'').substring(0,38);btn.title=kf.label;
  btn.onclick=()=>vid.currentTime=kf.t;
  document.getElementById('kb').appendChild(btn);
}});
tl.addEventListener('click',e=>{{const r=tl.getBoundingClientRect();vid.currentTime=((e.clientX-r.left)/r.width)*DUR}});
vid.addEventListener('timeupdate',()=>{{
  pb.style.width=((vid.currentTime/DUR)*100)+'%';
  ct.textContent=ft(vid.currentTime);
  const cur=SHOTS.find(s=>vid.currentTime>=s.start&&vid.currentTime<s.end);
  if(cur){{si.textContent='S'+String(cur.idx).padStart(3,'0')+' ['+cur.type+'] '+ft(cur.start)+'→'+ft(cur.end);
    document.querySelectorAll('#sb .jb').forEach(b=>b.classList.toggle('on',parseFloat(b.dataset.s)===cur.start));}}
  const nkf=KFS.reduce((b,k)=>Math.abs(k.t-vid.currentTime)<Math.abs(b.t-vid.currentTime)?k:b,KFS[0]||{{t:0}});
  if(nkf){{mi.textContent=(nkf.mood?'🎭 '+nkf.mood:'')+(nkf.scene?' | 🎬 '+nkf.scene:'');
    document.querySelectorAll('#kb .jb').forEach(b=>b.classList.toggle('on',parseFloat(b.dataset.t)===nkf.t));}}
}});
function sp(n){{
  ['shots','kf'].forEach(x=>{{document.getElementById('p-'+x).classList.toggle('on',x===n)}});
  document.querySelectorAll('.tb').forEach((b,i)=>b.classList.toggle('on',(i===0&&n==='shots')||(i===1&&n==='kf')));
}}
</script></body></html>""", height=640)

        st.subheader("Keyframe Strip")
        _all_kf = keyframes(VID)
        scene_opts = ["— All —"] + sorted({fa_map.get(str(k["id"]),{}).get("scene_id") or "?" for k in _all_kf})
        sc_sel = st.selectbox("Scene filter", scene_opts)
        disp_kf = [k for k in _all_kf if sc_sel == "— All —" or (fa_map.get(str(k["id"]),{}).get("scene_id") or "?") == sc_sel]
        limit_kf = st.slider("Max frames", 6, min(120, len(disp_kf)), 24, key="kf_strip")
        cols_n = 6
        for ri in range(0, limit_kf, cols_n):
            row = disp_kf[ri:ri+cols_n]
            rcols = st.columns(cols_n)
            for col, kf in zip(rcols, row):
                url = r2_url(kf.get("r2_key"))
                a   = fa_map.get(str(kf["id"]), {})
                cap = f"{ft(kf['timestamp_s'])} · {a.get('scene_mood') or ''}"
                if url:
                    col.image(url, caption=cap, use_column_width=True)
                else:
                    col.caption(f"🖼 {cap}")
                if a.get("caption"):
                    col.caption(a["caption"][:60])


# ══════════════════════════════════════════════════════════════════════════════
# 📊 OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.header("Knowledge Base Overview")
    import pandas as pd, plotly.express as px

    _cnt = counts(VID)  # single fast query — 12 COUNT(*) subqueries

    metrics = [
        ("Chunks",           _cnt.get("chunks", 0)),
        ("Shots",            _cnt.get("shots", 0)),
        ("Keyframes",        _cnt.get("keyframes", 0)),
        ("Transcript segs",  _cnt.get("transcript_segs", 0)),
        ("Persons",          _cnt.get("persons", 0)),
        ("Face appearances", _cnt.get("face_appearances", 0)),
        ("Speaker turns",    _cnt.get("speaker_turns", 0)),
        ("Color grades",     _cnt.get("color_grades", 0)),
        ("Frame analyses",   _cnt.get("frame_analyses", 0)),
        ("Searchable facts", _cnt.get("facts", 0)),
        ("KG nodes",         _cnt.get("kg_nodes", 0)),
        ("KG edges",         _cnt.get("kg_edges", 0)),
    ]
    cols = st.columns(6)
    for i, (name, val) in enumerate(metrics):
        cols[i % 6].metric(name, val)

    # ── Pipeline timing ──────────────────────────────────────────────────────
    st.subheader("Pipeline Timing")
    _jobs_data = jobs(VID)
    if _jobs_data:
        level_colors = {1: "#3b82f6", 2: "#10b981", 3: "#8b5cf6"}
        level_names  = {1: "L1 ASHFS+Whisper", 2: "L2 Face+Color", 3: "L3 Qwen+Graph"}

        # Total duration row
        dur_rows = []
        for j in _jobs_data:
            if j.get("started_at") and j.get("completed_at"):
                try:
                    s = j["started_at"]
                    e = j["completed_at"]
                    # both are datetime objects from psycopg2
                    total_s = (e - s).total_seconds()
                    dur_rows.append({
                        "Level": level_names.get(j["level"], f"L{j['level']}"),
                        "Total (s)": round(total_s, 1),
                        "Status": j["status"],
                    })
                except Exception:
                    pass

        if dur_rows:
            dur_df = pd.DataFrame(dur_rows)
            st.plotly_chart(px.bar(dur_df, x="Level", y="Total (s)", color="Level",
                color_discrete_sequence=list(level_colors.values()),
                text="Total (s)", title="Total time per level (wall clock)", height=220),
                use_container_width=True)

        # Per-step breakdown from meta
        step_rows = []
        for j in _jobs_data:
            meta = j.get("meta") or {}
            if isinstance(meta, str):
                try: meta = json.loads(meta)
                except Exception: meta = {}
            steps = meta.get("steps") or {}
            lname = level_names.get(j["level"], f"L{j['level']}")
            for step, secs in steps.items():
                if isinstance(secs, (int, float)) and secs > 0:
                    step_rows.append({"Level": lname, "Step": step, "Seconds": round(float(secs), 1)})

        if step_rows:
            sdf = pd.DataFrame(step_rows)
            st.plotly_chart(px.bar(sdf, x="Step", y="Seconds", color="Level",
                barmode="group", text="Seconds",
                title="Step-level timing breakdown", height=280),
                use_container_width=True)

            # Also show as table with human-readable time
            def _hms(s):
                s = int(s)
                if s < 60: return f"{s}s"
                m, sec = divmod(s, 60)
                return f"{m}m {sec}s"
            sdf["Time"] = sdf["Seconds"].apply(_hms)
            st.dataframe(sdf[["Level","Step","Seconds","Time"]], use_container_width=True, hide_index=True)
        else:
            st.info("No step timing yet — available after next pipeline run.")

    # Charts load lazily inside expanders so they don't block Overview render
    with st.expander("📈 Shot timeline", expanded=True):
        _s_ov = shots(VID)
        if _s_ov:
            df = pd.DataFrame([{"Shot": f"S{s['shot_index']:03d}", "Start": s["start_time"] or 0,
                "End": s["end_time"] or 0, "Type": s["shot_type"] or "unknown",
                "Complexity": round(s["complexity"] or 0, 3)} for s in _s_ov])
            fig = px.bar(df, x="Start", y="Complexity", color="Type", hover_data=["Shot"],
                labels={"Start": "Time (s)"}, height=160)
            fig.update_layout(margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("🎭 Scene mood / beat / tension", expanded=False):
        _fa_ov = frame_analyses_list(VID, limit=500, offset=0)
        if _fa_ov:
            col1, col2, col3 = st.columns(3)
            moods = [a["scene_mood"] for a in _fa_ov if a.get("scene_mood")]
            beats = [a["beat_type"] for a in _fa_ov if a.get("beat_type")]
            tensions = [a["tension_level"] for a in _fa_ov if a.get("tension_level")]
            for col, data, title in [(col1, moods, "Moods"), (col2, beats, "Story beats"), (col3, tensions, "Tension")]:
                if data:
                    df2 = pd.Series(data).value_counts().reset_index()
                    df2.columns = ["Value", "Count"]
                    col.plotly_chart(px.pie(df2, names="Value", values="Count", title=title, height=260), use_container_width=True)

    with st.expander("🏷 Top tags", expanded=False):
        _fa_tags = frame_analyses_list(VID, limit=500, offset=0)
        all_tags = [t for a in _fa_tags for t in (a.get("tags") or [])]
        if all_tags:
            tag_df = pd.Series(all_tags).value_counts().head(30).reset_index()
            tag_df.columns = ["Tag", "Count"]
            st.plotly_chart(px.bar(tag_df, x="Tag", y="Count", height=220), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 🗂 CHUNKS
# ══════════════════════════════════════════════════════════════════════════════
with tab_chunks:
    st.header("Chunks (L1)")
    _c = chunks(VID); _t = transcript(VID)
    if not _c:
        st.info("No chunks.")
    else:
        import pandas as pd
        df = pd.DataFrame([{
            "Index": c["chunk_index"], "Start": ft(c["start_time"]), "End": ft(c["end_time"]),
            "Start (s)": c["start_time"], "End (s)": c["end_time"],
            "Start frame": c["start_frame"], "End frame": c["end_frame"],
        } for c in _c])
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("Chunk detail with transcript")
        sel_chunk = st.selectbox("Chunk", [f"Chunk {c['chunk_index']} ({ft(c['start_time'])}–{ft(c['end_time'])})" for c in _c])
        ci = int(sel_chunk.split()[1])
        chunk = next((c for c in _c if c["chunk_index"] == ci), None)
        if chunk:
            segs = [t for t in _t if t.get("chunk_id") and str(t["chunk_id"]) == str(chunk["id"])]
            st.markdown(f"**{ft(chunk['start_time'])} → {ft(chunk['end_time'])}** · frames {chunk['start_frame']}–{chunk['end_frame']}")
            if segs:
                for s in segs:
                    conf = f" ({s['confidence']:.0%})" if s.get("confidence") else ""
                    st.markdown(f"> `{ft(s['start_time'])}` {s['text']}{conf}")
            else:
                st.info("No transcript segments aligned to this chunk.")


# ══════════════════════════════════════════════════════════════════════════════
# 🎬 SHOTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_shots:
    st.header("Shots (L1)")
    _s = shots(VID)
    if not _s:
        st.info("No shots.")
    else:
        import pandas as pd, plotly.express as px
        df = pd.DataFrame([{
            "Shot": s["shot_index"], "Start": ft(s["start_time"]), "End": ft(s["end_time"]),
            "Duration (s)": round((s["end_time"] or 0) - (s["start_time"] or 0), 2),
            "Type": s["shot_type"] or "—", "Complexity": round(s["complexity"] or 0, 4),
            "Start frame": s["start_frame"], "End frame": s["end_frame"],
        } for s in _s])
        st.dataframe(df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        col1.plotly_chart(px.histogram(df, x="Complexity", nbins=25, title="Complexity distribution", height=240), use_container_width=True)
        col2.plotly_chart(px.pie(df, names="Type", title="Shot types", height=240), use_container_width=True)

        st.plotly_chart(px.bar(df, x="Shot", y="Duration (s)", color="Type",
            title="Shot durations", height=200), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 🖼 KEYFRAMES
# ══════════════════════════════════════════════════════════════════════════════
with tab_kf:
    st.header("Keyframes (L1)")
    _k = keyframes(VID)
    if not _k:
        st.info("No keyframes.")
    else:
        import pandas as pd
        show_imgs = st.checkbox("Show images from R2", value=True)
        reason_filter = st.multiselect("Selection reason", list({k.get("selection_reason") or "?" for k in _k}))
        filtered = [k for k in _k if not reason_filter or (k.get("selection_reason") or "?") in reason_filter]
        st.markdown(f"**{len(filtered)} keyframes**")

        if show_imgs:
            limit = st.slider("Max to show", 6, min(120, len(filtered)), 30)
            cols_n = 5
            for ri in range(0, limit, cols_n):
                row = filtered[ri:ri+cols_n]
                rcols = st.columns(cols_n)
                for col, kf in zip(rcols, row):
                    url = r2_url(kf.get("r2_key"))
                    if url:
                        col.image(url, caption=f"{ft(kf['timestamp_s'])} · {kf.get('selection_reason') or ''}", use_column_width=True)
                    else:
                        col.caption(f"🖼 {ft(kf['timestamp_s'])}")
        else:
            df = pd.DataFrame([{
                "Frame": kf["frame_index"], "Time": ft(kf["timestamp_s"]),
                "Timestamp (s)": kf["timestamp_s"],
                "Reason": kf.get("selection_reason") or "—",
                "R2 key": kf.get("r2_key") or "—",
                "Has DINO emb": kf.get("dino_embedding") is not None,
            } for kf in filtered])
            st.dataframe(df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# 📝 TRANSCRIPT
# ══════════════════════════════════════════════════════════════════════════════
with tab_trans:
    st.header("Transcript (L1 — Whisper)")
    _t = transcript(VID)
    if not _t:
        st.info("No transcript.")
    else:
        total_words = sum(len(t["text"].split()) for t in _t)
        st.metric("Segments", len(_t))
        col1, col2 = st.columns(2)
        col1.metric("Total words", total_words)
        col2.metric("Duration covered", ft((_t[-1]["end_time"] or 0) - (_t[0]["start_time"] or 0)))

        search = st.text_input("🔎 Search transcript", "")
        filtered = [t for t in _t if not search or search.lower() in t["text"].lower()]
        st.caption(f"{len(filtered)} segments")

        view = st.radio("View", ["Readable", "Table"], horizontal=True)
        if view == "Readable":
            for seg in filtered:
                conf = f" ({seg['confidence']:.0%})" if seg.get("confidence") else ""
                words_info = ""
                if seg.get("words"):
                    try:
                        wds = seg["words"] if isinstance(seg["words"], list) else json.loads(seg["words"])
                        words_info = f" · {len(wds)} words"
                    except Exception:
                        pass
                st.markdown(f"`{ft(seg['start_time'])}–{ft(seg['end_time'])}{conf}{words_info}` **{seg['text']}**")
        else:
            import pandas as pd
            df = pd.DataFrame([{
                "Start": ft(t["start_time"]), "End": ft(t["end_time"]),
                "Text": t["text"], "Confidence": round(t["confidence"] or 0, 3) if t.get("confidence") else "—",
            } for t in filtered])
            st.dataframe(df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# 👤 PERSONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_persons:
    st.header("Persons (L2 — ArcFace)")
    _p = persons(VID)
    _fa2 = face_appearances(VID)
    if not _p:
        st.info("No persons detected.")
    else:
        import pandas as pd

        # Real counts from DB — not filtered by JOIN success
        _real_counts = q("""
            SELECT p.pid, p.display_name,
                   COUNT(fa.id) AS total,
                   COUNT(fa.person_id) AS linked,
                   COUNT(*) FILTER (WHERE fa.person_id IS NULL) AS null_person_id
            FROM persons p
            LEFT JOIN face_appearances fa ON fa.person_id = p.id
            WHERE p.video_id = %s::uuid
            GROUP BY p.pid, p.display_name ORDER BY p.pid
        """, VID)

        # Also count orphan appearances (person_id IS NULL)
        _orphan = q1("SELECT COUNT(*) AS n FROM face_appearances WHERE video_id=%s::uuid AND person_id IS NULL", VID)
        _orphan_n = _orphan.get("n", 0)
        if _orphan_n:
            st.warning(f"⚠️ **{_orphan_n} face appearances have person_id = NULL** — these are detected faces not linked to any person. "
                       f"Likely cause: face pipeline wrote appearances before persons were committed, or track→PID mapping failed.")

        rc_map = {r["pid"]: r for r in _real_counts}

        for p in _p:
            name = p["display_name"] or p["pid"]
            rc = rc_map.get(p["pid"], {})
            total_real = rc.get("total", 0)
            linked = rc.get("linked", 0)

            with st.expander(f"**{name}** (`{p['pid']}`) — {total_real} appearances"):
                st.markdown(f"- **DB ID:** `{p['id']}`")
                st.markdown(f"- **PID:** `{p['pid']}`")
                st.markdown(f"- **Display name:** {p['display_name'] or '—'}")
                st.markdown(f"- **Total appearances (DB):** {total_real}")
                st.markdown(f"- **Linked (person_id set):** {linked}")

                if total_real == 0:
                    st.error("0 appearances. Possible causes:\n"
                             "1. **person_id mismatch** — appearances written with old UUID before this person row existed\n"
                             "2. **face pipeline wrote NULL person_id** for this PID's track\n"
                             "3. **Cast DB lookup failed** — person detected but not matched → no PID assigned\n\n"
                             "Run diagnostic query below ↓")

                # Load actual appearances for this person via direct DB query (bypasses the JOIN issue)
                p_apps = q("""
                    SELECT fa.frame_index, fa.timestamp_s, fa.track_id,
                           fa.emotion, fa.emotion_conf, fa.person_id
                    FROM face_appearances fa
                    WHERE fa.person_id = %s::uuid
                    ORDER BY fa.timestamp_s LIMIT 100
                """, str(p["id"]))

                if p_apps:
                    df = pd.DataFrame([{
                        "Frame": a["frame_index"],
                        "Time": ft(a["timestamp_s"]),
                        "Track ID": a.get("track_id"),
                        "Emotion": a.get("emotion") or "—",
                        "Conf": round(a["emotion_conf"] or 0, 3) if a.get("emotion_conf") else "—",
                    } for a in p_apps])
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("No face_appearances rows with this person_id UUID.")


# ══════════════════════════════════════════════════════════════════════════════
# 😊 FACE APPEARANCES
# ══════════════════════════════════════════════════════════════════════════════
with tab_faces:
    st.header("Face Appearances (L2)")
    _fa2_total = _cnt.get("face_appearances", 0)
    if not _fa2_total:
        st.info("No face appearances.")
    else:
        import pandas as pd, plotly.express as px
        c1, c2 = st.columns([3,1])
        _pids = [p["pid"] for p in persons(VID)]
        pid_filter = c1.multiselect("Filter by person", _pids)
        fa2_limit = c2.number_input("Limit", 100, 5000, 2000, step=100, key="fa2_lim")
        _fa2 = face_appearances(VID, limit=fa2_limit)
        filtered = [a for a in _fa2 if not pid_filter or a.get("pid") in pid_filter]
        st.metric("Shown", len(filtered), delta=f"of {_fa2_total} total")

        df = pd.DataFrame([{
            "Time": ft(a["timestamp_s"]),
            "Timestamp (s)": a["timestamp_s"],
            "Frame": a["frame_index"],
            "PID": a.get("pid") or "—",
            "Name": a.get("display_name") or "—",
            "Track ID": a.get("track_id"),
            "Emotion": a.get("emotion") or "—",
            "Conf": round(a["emotion_conf"] or 0, 3) if a.get("emotion_conf") else "—",
            "BBox": str(a.get("bbox") or ""),
        } for a in filtered])
        st.dataframe(df, use_container_width=True, hide_index=True)

        if not df.empty and "Emotion" in df.columns:
            emo_df = df["Emotion"].value_counts().reset_index()
            emo_df.columns = ["Emotion", "Count"]
            st.plotly_chart(px.bar(emo_df, x="Emotion", y="Count", color="Emotion",
                title="Emotion counts", height=220), use_container_width=True)

        if not df.empty:
            by_pid = df.groupby("PID").size().reset_index(name="Appearances")
            st.plotly_chart(px.pie(by_pid, names="PID", values="Appearances",
                title="Appearances by person", height=240), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 😐 EMOTION TIMELINE
# ══════════════════════════════════════════════════════════════════════════════
with tab_emotions:
    st.header("Emotion Timeline (L2)")
    _tl = face_timeline(VID)
    if not _tl:
        st.info("No emotion timeline events.")
    else:
        import pandas as pd, plotly.express as px
        df = pd.DataFrame([{
            "Person": t.get("display_name") or t.get("pid") or "Unknown",
            "Emotion": t["emotion"],
            "Start": t["start_time"] or 0,
            "End": t["end_time"] or 0,
            "Duration (s)": round((t["end_time"] or 0) - (t["start_time"] or 0), 2),
            "Confidence": round(t["confidence"] or 0, 3) if t.get("confidence") else "—",
        } for t in _tl])
        st.dataframe(df, use_container_width=True, hide_index=True)

        fig = px.timeline(df, x_start="Start", x_end="End", y="Person", color="Emotion",
            title="Emotion spans", hover_data=["Duration (s)", "Confidence"])
        fig.update_xaxes(title="Time (s)")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        emo_dur = df.groupby("Emotion")["Duration (s)"].sum().reset_index()
        col1.plotly_chart(px.pie(emo_dur, names="Emotion", values="Duration (s)",
            title="Time by emotion", height=260), use_container_width=True)
        emo_cnt = df["Emotion"].value_counts().reset_index()
        emo_cnt.columns = ["Emotion", "Count"]
        col2.plotly_chart(px.bar(emo_cnt, x="Emotion", y="Count", color="Emotion",
            title="Event count by emotion", height=260), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 🗣 DIARIZATION (Level-2 Stage 0: pyannote turns fused w/ ArcFace identity)
# ══════════════════════════════════════════════════════════════════════════════
with tab_diarization:
    st.header("Speaker Diarization (L2 Stage 0)")
    _st_turns = speaker_turns(VID)
    if not _st_turns:
        st.info("No speaker turns. Diarization may not have run for this video "
                "(non-fatal step — check HF_TOKEN + pyannote model license acceptance).")
    else:
        import pandas as pd, plotly.express as px

        df = pd.DataFrame([{
            "Speaker": t.get("display_name") or t.get("pid") or "unresolved",
            "Cluster": t["cluster_label"],
            "Start": t["start_time"] or 0,
            "End": t["end_time"] or 0,
            "Duration (s)": round((t["end_time"] or 0) - (t["start_time"] or 0), 2),
            "Confidence": round(t["confidence"], 3) if t.get("confidence") is not None else "—",
            "Resolution": t["resolution_method"],
        } for t in _st_turns])

        n_unresolved = (df["Resolution"] == "unresolved").sum()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total turns", len(df))
        col2.metric("Resolved to a person", len(df) - n_unresolved)
        col3.metric("Unresolved (needs L5 grounding)", n_unresolved)

        st.dataframe(df, use_container_width=True, hide_index=True)

        fig = px.timeline(df, x_start="Start", x_end="End", y="Speaker", color="Resolution",
            title="Speaker turns — colored by fusion resolution method",
            hover_data=["Cluster", "Duration (s)", "Confidence"])
        fig.update_xaxes(title="Time (s)")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        dur_by_speaker = df.groupby("Speaker")["Duration (s)"].sum().reset_index()
        col1.plotly_chart(px.pie(dur_by_speaker, names="Speaker", values="Duration (s)",
            title="Talk time by speaker", height=260), use_container_width=True)
        res_cnt = df["Resolution"].value_counts().reset_index()
        res_cnt.columns = ["Resolution", "Count"]
        col2.plotly_chart(px.bar(res_cnt, x="Resolution", y="Count", color="Resolution",
            title="Turns by resolution method (trust cascade)", height=260), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 🎨 COLOR GRADES
# ══════════════════════════════════════════════════════════════════════════════
with tab_color:
    st.header("Color Grades (L2)")
    _cg = color_grades(VID)
    if not _cg:
        st.info("No color grades.")
    else:
        import pandas as pd, plotly.express as px
        st.metric("Graded shots", len(_cg))

        # Style tags overview
        all_tags = [t for g in _cg for t in g["style_tags"]]
        if all_tags:
            tag_df = pd.Series(all_tags).value_counts().reset_index()
            tag_df.columns = ["Tag", "Count"]
            st.plotly_chart(px.bar(tag_df, x="Tag", y="Count", title="Style tags", height=200), use_container_width=True)

        for g in _cg:
            tags_str = ", ".join(g["style_tags"]) if g["style_tags"] else "—"
            with st.expander(f"t={ft(g['timestamp_s'])} · {tags_str}"):
                params = g.get("parameters", {})
                if params:
                    rows = []
                    for name, vals in params.items():
                        if isinstance(vals, dict):
                            rows.append({
                                "Parameter": name,
                                "Current": str(vals.get("current") if vals.get("current") is not None else "—"),
                                "Recommended": str(vals.get("recommended") if vals.get("recommended") is not None else "—"),
                                "Delta": str(vals.get("delta") if vals.get("delta") is not None else "—"),
                            })
                        else:
                            rows.append({"Parameter": name, "Current": str(vals), "Recommended": "—", "Delta": "—"})
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# 🧠 FRAME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    st.header("Frame Analysis (L3 — Qwen VL)")
    import pandas as pd
    _fa_total = _cnt.get("frame_analyses", 0)
    if not _fa_total:
        st.info("No frame analyses.")
    else:
        # ── Controls (pagination + filter) ──
        PAGE_SIZE = 25
        c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
        sc_filter = c1.selectbox("Scene", ["— All —"] + sorted({
            r["scene_id"] or "?" for r in
            q("SELECT DISTINCT scene_id FROM frame_analyses WHERE video_id=%s::uuid", VID)
        }), key="fa_sc")
        show_img = c3.checkbox("Images", True, key="fa_img")

        # Page selector
        total_for_page = int(q1(
            "SELECT COUNT(*) AS n FROM frame_analyses fa "
            "JOIN keyframes kf ON kf.id = fa.keyframe_id "
            "WHERE fa.video_id=%s::uuid" + (" AND fa.scene_id=%s" if sc_filter != "— All —" else ""),
            *([VID, sc_filter] if sc_filter != "— All —" else [VID])
        ).get("n", 0))

        max_pages = max(1, (total_for_page + PAGE_SIZE - 1) // PAGE_SIZE)
        page = c4.number_input("Page", 1, max_pages, 1, key="fa_pg") - 1
        offset = page * PAGE_SIZE

        st.caption(f"{total_for_page} frames · page {page+1}/{max_pages}")

        # Fetch only this page (no qwen_output JSON)
        scene_clause = " AND fa.scene_id=%s" if sc_filter != "— All —" else ""
        page_args = [VID] + ([sc_filter] if sc_filter != "— All —" else []) + [PAGE_SIZE, offset]
        _fa_page = q(f"""
            SELECT fa.id, fa.keyframe_id, fa.scene_id, fa.scene_change,
                   fa.caption, fa.beat_type, fa.scene_mood, fa.tension_level, fa.tags,
                   kf.timestamp_s, kf.r2_key, kf.frame_index
            FROM frame_analyses fa
            JOIN keyframes kf ON kf.id = fa.keyframe_id
            WHERE fa.video_id=%s::uuid{scene_clause}
            ORDER BY kf.timestamp_s
            LIMIT %s OFFSET %s
        """, *page_args)

        for a in _fa_page:
            a["tags"] = list(a.get("tags") or [])
            ts = ft(a["timestamp_s"])
            with st.expander(
                f"🕐 {ts} · {a.get('scene_id') or '?'} · {a.get('beat_type') or '?'} · {a.get('scene_mood') or '?'} · tension={a.get('tension_level') or '?'}"
            ):
                if show_img and a.get("r2_key"):
                    url = r2_url(a["r2_key"])
                    if url:
                        st.image(url, width=500)

                st.markdown(f"### {a.get('caption') or '—'}")
                st.markdown(f"**Scene change:** {'✅' if a.get('scene_change') else '—'}")
                tags = a.get("tags") or []
                if tags:
                    st.markdown("**Tags:** " + " ".join(f"`{t}`" for t in tags))

                # Load full qwen_output only when user opens this expander
                qwen = frame_analysis_detail(str(a["id"]))

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**⚙️ Setting**");    jshow(qwen.get("setting"))
                    st.markdown("**🎥 Composition**"); jshow(qwen.get("composition"))
                with col2:
                    st.markdown("**🌈 Lighting/Color**"); jshow(qwen.get("lighting_color"))
                    st.markdown("**🎭 Emotional tone**"); jshow(qwen.get("emotional_tone"))
                with col3:
                    st.markdown("**📖 Story beat**");   jshow(qwen.get("story_beat"))
                    st.markdown("**🔗 Causality**");    jshow(qwen.get("causality"))
                    st.markdown("**↔️ Continuity**");   jshow(qwen.get("continuity"))

                people = qwen.get("people") or []
                if people:
                    st.markdown("**👤 People**")
                    pdf = pd.DataFrame([{
                        "PID": p.get("pid"), "Action": p.get("action"), "Pose": p.get("pose"),
                        "Emotion": p.get("emotion_inferred"), "Source": p.get("emotion_source"),
                        "Role": p.get("story_role"),
                    } for p in people])
                    st.dataframe(pdf, use_container_width=True, hide_index=True)

                objects = qwen.get("objects") or []
                if objects:
                    st.markdown("**📦 Objects**")
                    st.dataframe(pd.DataFrame([{"Name": o.get("name"), "Position": o.get("position"),
                        "Relevance": o.get("story_relevance")} for o in objects]),
                        use_container_width=True, hide_index=True)

                dlg = qwen.get("dialogue_subtitle")
                if dlg: st.markdown(f"**💬** {dlg}")

                rels = qwen.get("relations") or []
                if rels:
                    def _fmt_rel(r):
                        if isinstance(r, dict):
                            return f"`{r.get('subject','?')}→{r.get('predicate','?')}→{r.get('object','?')}`"
                        if isinstance(r, (list, tuple)) and len(r) >= 3:
                            return f"`{r[0]}→{r[1]}→{r[2]}`"
                        return str(r)
                    st.markdown("**🔀 Relations:** " + " · ".join(_fmt_rel(r) for r in rels))

                themes = qwen.get("themes_symbols") or []
                if themes:
                    st.markdown("**🎯 Themes:** " + ", ".join(f"`{x}`" for x in themes))

                facts_q = qwen.get("searchable_facts") or []
                if facts_q:
                    st.markdown("**💡 Facts:**")
                    for f in facts_q: st.markdown(f"  - {f}")


# ══════════════════════════════════════════════════════════════════════════════
# 💡 FACTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_facts:
    st.header("Searchable Facts (L3)")
    import pandas as pd
    _f_total = _cnt.get("facts", 0)
    if not _f_total:
        st.info("No facts.")
    else:
        c1, c2 = st.columns([4, 1])
        search = c1.text_input("🔎 Search facts (DB-side)", "", key="facts_search")
        facts_limit = c2.number_input("Limit", 50, 2000, 500, step=50, key="facts_lim")

        if search:
            _f = q("SELECT id, fact_text, timestamp_s, pinecone_id FROM searchable_facts "
                   "WHERE video_id=%s::uuid AND fact_text ILIKE %s ORDER BY timestamp_s LIMIT %s",
                   VID, f"%{search}%", facts_limit)
        else:
            _f = facts(VID, limit=facts_limit)

        st.caption(f"{len(_f)} shown of {_f_total} total")
        df = pd.DataFrame([{
            "Time": ft(f["timestamp_s"]),
            "Fact": f["fact_text"],
            "Pinecone ID": f.get("pinecone_id") or "—",
        } for f in _f])
        st.dataframe(df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# 🕸 KG NODES  (Postgres — interactive graph)
# ══════════════════════════════════════════════════════════════════════════════
with tab_kgnodes:
    st.header("Knowledge Graph — Nodes")
    _kn = kg_nodes(VID)
    if not _kn:
        st.info("No KG nodes.")
    else:
        import pandas as pd, plotly.express as px
        import streamlit.components.v1 as components

        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Total nodes", len(_kn))
        type_counts = pd.Series([n["node_type"] for n in _kn]).value_counts()
        col_m2.metric("Node types", len(type_counts))

        st.markdown(_legend_html(_NODE_COLORS), unsafe_allow_html=True)

        # ── Controls ──
        c1, c2, c3 = st.columns([3, 2, 1])
        ntype_filter = c1.multiselect("Show types", list(type_counts.index), default=list(type_counts.index), key="kn_type")
        max_nodes = c2.slider("Max nodes in graph", 20, min(500, len(_kn)), min(150, len(_kn)), key="kn_max")
        physics_on = c3.checkbox("Physics", True, key="kn_phys")

        filtered_nodes = [n for n in _kn if n["node_type"] in ntype_filter][:max_nodes]
        node_ids = {str(n["id"]) for n in filtered_nodes}

        # Build pyvis graph (nodes only, no edges here — edges tab has full graph)
        if filtered_nodes:
            pyvis_nodes = [{"id": str(n["id"]), "node_type": n["node_type"],
                            "label": n.get("label") or n["node_type"],
                            "properties": n.get("properties") or {}} for n in filtered_nodes]
            html_graph = build_pyvis(pyvis_nodes, [], height=540, physics=physics_on)
            components.html(html_graph, height=560, scrolling=False)

        # Stats row
        tc_df = type_counts.reset_index()
        tc_df.columns = ["Type", "Count"]
        col_p1, col_p2 = st.columns([1, 2])
        col_p1.plotly_chart(px.pie(tc_df, names="Type", values="Count", height=260,
            color="Type", color_discrete_map=_NODE_COLORS), use_container_width=True)

        # Table below graph
        with st.expander("📋 Node table", expanded=False):
            df = pd.DataFrame([{
                "Type": n["node_type"], "Label": n.get("label") or "—",
                "Ref ID": str(n.get("ref_id") or "")[:36],
                "Properties": json.dumps(n.get("properties") or {}),
            } for n in filtered_nodes])
            col_p2.dataframe(df, use_container_width=True, hide_index=True, height=260)


# ══════════════════════════════════════════════════════════════════════════════
# 🔗 KG EDGES  (Postgres — full interactive graph)
# ══════════════════════════════════════════════════════════════════════════════
with tab_kgedges:
    st.header("Knowledge Graph — Full Graph (Postgres)")
    _ke = kg_edges(VID)
    _kn2 = kg_nodes(VID)
    if not _ke:
        st.info("No KG edges. Reset L3 and re-run after node-ID sync fix.")
    else:
        import pandas as pd, plotly.express as px
        import streamlit.components.v1 as components

        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Nodes", len(_kn2))
        col_m2.metric("Edges", len(_ke))

        st.markdown(_legend_html(_NODE_COLORS), unsafe_allow_html=True)

        # ── Controls ──
        c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
        all_rels = sorted({e["relation"] for e in _ke})
        rel_filter = c1.multiselect("Relations", all_rels, default=all_rels, key="ke_rel")
        all_types = sorted({n["node_type"] for n in _kn2})
        type_filter = c2.multiselect("Node types", all_types, default=all_types, key="ke_nt")
        max_nodes_g = c3.slider("Max nodes", 30, min(400, len(_kn2)), min(200, len(_kn2)), key="ke_max")
        phys2 = c4.checkbox("Physics", True, key="ke_phys")

        # Build filtered node/edge sets
        filtered_edges = [e for e in _ke if e["relation"] in rel_filter
                          and e.get("src_type") in type_filter
                          and e.get("tgt_type") in type_filter]

        # Need to load full kg_nodes with id for edge resolution
        full_nodes_map = {str(n["id"]): n for n in _kn2}
        # kg_edges returns src_label/tgt_label but not IDs — re-fetch with IDs
        edge_rows = q("""
            SELECT ke.relation, ke.weight,
                   ke.source_id, ke.target_id,
                   src.node_type AS src_type, src.label AS src_label,
                   tgt.node_type AS tgt_type, tgt.label AS tgt_label
            FROM kg_edges ke
            JOIN kg_nodes src ON src.id = ke.source_id
            JOIN kg_nodes tgt ON tgt.id = ke.target_id
            WHERE ke.video_id=%s
        """, VID)

        edge_rows_f = [e for e in edge_rows
                       if e["relation"] in rel_filter
                       and e.get("src_type") in type_filter
                       and e.get("tgt_type") in type_filter]

        # Collect node IDs referenced by filtered edges
        used_ids = set()
        for e in edge_rows_f:
            used_ids.add(str(e["source_id"]))
            used_ids.add(str(e["target_id"]))

        # Cap to max_nodes_g by keeping top-connected nodes
        from collections import Counter
        id_freq = Counter()
        for e in edge_rows_f:
            id_freq[str(e["source_id"])] += 1
            id_freq[str(e["target_id"])] += 1
        top_ids = {nid for nid, _ in id_freq.most_common(max_nodes_g)}

        graph_nodes = [n for n in _kn2 if str(n["id"]) in top_ids]
        graph_edges = [{"source_id": str(e["source_id"]), "target_id": str(e["target_id"]),
                        "relation": e["relation"]}
                       for e in edge_rows_f
                       if str(e["source_id"]) in top_ids and str(e["target_id"]) in top_ids]

        if graph_nodes:
            pyvis_nodes2 = [{"id": str(n["id"]), "node_type": n["node_type"],
                             "label": n.get("label") or n["node_type"],
                             "properties": n.get("properties") or {}} for n in graph_nodes]
            html_g2 = build_pyvis(pyvis_nodes2, graph_edges, height=620, physics=phys2)
            components.html(html_g2, height=640, scrolling=False)
        else:
            st.warning("No nodes match current filters.")

        # Edge stats
        rel_counts = pd.Series([e["relation"] for e in edge_rows_f]).value_counts().reset_index()
        rel_counts.columns = ["Relation", "Count"]
        st.plotly_chart(px.bar(rel_counts, x="Relation", y="Count", color="Relation",
            color_discrete_map=_EDGE_COLORS, title="Edges by relation", height=200),
            use_container_width=True)

        with st.expander("📋 Edge table", expanded=False):
            df = pd.DataFrame([{
                "Relation": e["relation"],
                "Source": f"[{e['src_type']}] {str(e.get('src_label') or '')[:35]}",
                "Target": f"[{e['tgt_type']}] {str(e.get('tgt_label') or '')[:35]}",
                "Weight": e["weight"],
            } for e in edge_rows_f[:1000]])
            st.dataframe(df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# 🌐 NEO4J  (live interactive graph)
# ══════════════════════════════════════════════════════════════════════════════
with tab_neo4j:
    st.header("Neo4j Graph — Live Explorer")
    import streamlit.components.v1 as components

    try:
        import pandas as pd, plotly.express as px

        nc = neo4j_q("MATCH (n) RETURN labels(n)[0] AS label, COUNT(n) AS cnt ORDER BY cnt DESC")
        rc = neo4j_q("MATCH ()-[r]->() RETURN type(r) AS rel, COUNT(r) AS cnt ORDER BY cnt DESC")

        total_n = sum(r["cnt"] for r in nc)
        total_r = sum(r["cnt"] for r in rc)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total nodes", total_n)
        m2.metric("Total relationships", total_r)
        m3.metric("Node labels", len(nc))
        m4.metric("Relation types", len(rc))

        st.markdown(_legend_html(_NODE_COLORS), unsafe_allow_html=True)

        if nc:
            col3, col4 = st.columns(2)
            nc_df = pd.DataFrame(nc)
            nc_df.columns = ["label", "cnt"]
            col3.plotly_chart(px.bar(nc_df, x="label", y="cnt", color="label",
                color_discrete_map=_NODE_COLORS, title="Nodes by label", height=200),
                use_container_width=True)
            if rc:
                rc_df = pd.DataFrame(rc)
                rc_df.columns = ["rel", "cnt"]
                col4.plotly_chart(px.bar(rc_df, x="rel", y="cnt", color="rel",
                    color_discrete_map=_EDGE_COLORS, title="Relationships", height=200),
                    use_container_width=True)

        # ── Interactive Graph Explorer ──
        st.subheader("Interactive Graph")
        ntypes = [r["label"] for r in nc if r.get("label")]

        preset_queries = {
            f"Full video graph (this video, 80 nodes)":
                f"MATCH (a)-[r]->(b) WHERE (a.video_id='{VID}' OR b.video_id='{VID}') RETURN a,r,b LIMIT 80",
            f"Scenes → Frames":
                f"MATCH (s:Scene)-[r]->(f:Frame) WHERE s.video_id='{VID}' RETURN s,r,f LIMIT 60",
            f"Persons in shots":
                f"MATCH (p:Person)-[r]->(sh:Shot) WHERE p.video_id='{VID}' RETURN p,r,sh LIMIT 50",
            f"Scene causality chain":
                f"MATCH path=(s1:Scene)-[:FOLLOWS|CAUSES*1..3]->(s2:Scene) WHERE s1.video_id='{VID}' RETURN path LIMIT 40",
            f"All persons + emotions":
                f"MATCH (p:Person)-[r:HAS_EMOTION]->(e) WHERE p.video_id='{VID}' RETURN p,r,e LIMIT 50",
        }

        col_q1, col_q2 = st.columns([3, 1])
        preset = col_q1.selectbox("Preset query", ["— Custom —"] + list(preset_queries.keys()))
        max_g = col_q2.slider("Max nodes", 20, 200, 80, key="neo_max")
        phys_neo = col_q2.checkbox("Physics", True, key="neo_phys")

        default_cypher = preset_queries.get(preset, f"MATCH (a)-[r]->(b) WHERE a.video_id='{VID}' RETURN a,r,b LIMIT {max_g}")
        cypher_in = st.text_area("Cypher", value=default_cypher, height=70, key="neo_cypher")

        col_run1, col_run2, col_run3 = st.columns([1, 1, 5])
        run_graph = col_run1.button("🕸 Graph", key="neo_run_g")
        run_table = col_run2.button("📋 Table", key="neo_run_t")

        if run_graph or run_table:
            try:
                with st.spinner("Running Cypher…"):
                    raw = neo4j_q(cypher_in)

                if run_graph:
                    # Extract nodes and edges from result
                    g_nodes, g_edges = [], []
                    seen_nodes, seen_edges = set(), set()

                    for row in raw:
                        for val in row.values():
                            # Neo4j node
                            if hasattr(val, "labels") and hasattr(val, "items"):
                                nid = str(val.id)
                                if nid not in seen_nodes:
                                    seen_nodes.add(nid)
                                    lbl = list(val.labels)[0] if val.labels else "Node"
                                    props = dict(val)
                                    label_text = (props.get("label") or props.get("name")
                                                  or props.get("pid") or lbl)[:30]
                                    g_nodes.append({"id": nid, "node_type": lbl,
                                                    "label": label_text, "properties": props})
                            # Neo4j relationship
                            elif hasattr(val, "type") and hasattr(val, "start_node"):
                                eid = str(val.id)
                                if eid not in seen_edges:
                                    seen_edges.add(eid)
                                    g_edges.append({
                                        "source_id": str(val.start_node.id),
                                        "target_id": str(val.end_node.id),
                                        "relation": val.type,
                                    })
                            # Path object
                            elif hasattr(val, "nodes") and hasattr(val, "relationships"):
                                for node in val.nodes:
                                    nid = str(node.id)
                                    if nid not in seen_nodes:
                                        seen_nodes.add(nid)
                                        lbl = list(node.labels)[0] if node.labels else "Node"
                                        props = dict(node)
                                        label_text = (props.get("label") or props.get("name")
                                                      or props.get("pid") or lbl)[:30]
                                        g_nodes.append({"id": nid, "node_type": lbl,
                                                        "label": label_text, "properties": props})
                                for rel in val.relationships:
                                    eid = str(rel.id)
                                    if eid not in seen_edges:
                                        seen_edges.add(eid)
                                        g_edges.append({
                                            "source_id": str(rel.start_node.id),
                                            "target_id": str(rel.end_node.id),
                                            "relation": rel.type,
                                        })

                    if g_nodes:
                        st.caption(f"Graph: {len(g_nodes)} nodes · {len(g_edges)} edges")
                        g_html = build_pyvis(g_nodes[:max_g], g_edges, height=600, physics=phys_neo)
                        components.html(g_html, height=620, scrolling=False)
                    else:
                        st.info("No graph objects in result — try Table view or adjust query.")
                        if raw:
                            st.dataframe(pd.DataFrame(raw), use_container_width=True)

                if run_table:
                    if raw:
                        # Flatten node/path objects to dicts
                        flat = []
                        for row in raw:
                            flat_row = {}
                            for k, v in row.items():
                                if hasattr(v, "items"):
                                    flat_row.update({f"{k}.{pk}": pv for pk, pv in v.items()})
                                elif hasattr(v, "nodes"):
                                    flat_row[k] = str(v)
                                else:
                                    flat_row[k] = v
                            flat.append(flat_row)
                        st.dataframe(pd.DataFrame(flat), use_container_width=True)
                    else:
                        st.info("No results.")

            except Exception as e:
                st.error(str(e))

        # ── Sample nodes browser ──
        with st.expander("Browse nodes by type"):
            if ntypes:
                ntype_sel = st.selectbox("Node type", ntypes, key="neo_nt")
                sample = neo4j_q(f"MATCH (n:{ntype_sel}) RETURN n LIMIT 30")
                if sample:
                    rows = [{k: v for k, v in rec["n"].items()} for rec in sample]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

        with st.expander("Browse relationships"):
            if rc:
                rel_sel = st.selectbox("Relation type", [r["rel"] for r in rc], key="neo_rel")
                samp_r = neo4j_q(
                    f"MATCH (a)-[r:{rel_sel}]->(b) "
                    f"RETURN labels(a)[0] AS src_type, a.label AS src, "
                    f"type(r) AS rel, labels(b)[0] AS tgt_type, b.label AS tgt LIMIT 30"
                )
                if samp_r:
                    st.dataframe(pd.DataFrame(samp_r), use_container_width=True)

    except Exception as e:
        st.error(f"Neo4j error: {e}")
        st.caption("Check NEO4J_URI / NEO4J_DATABASE / NEO4J_USER / NEO4J_PASSWORD in .env")


# ══════════════════════════════════════════════════════════════════════════════
# 🔍 SEARCH
# ══════════════════════════════════════════════════════════════════════════════
with tab_search:
    st.header("Semantic Search — Pinecone")

    col1, col2, col3 = st.columns([4, 1, 1])
    query = col1.text_input("Query", placeholder="red curtain, three men, nervous expression...")
    top_k = col2.number_input("Top K", 1, 50, 10)
    ns = col3.radio("Namespace", ["facts", "scenes"])

    if query:
        try:
            with st.spinner("Embedding..."):
                from models.embed_model import LocalEmbedder
                vec = LocalEmbedder.get().embed([query], batch_size=1)[0]

            with st.spinner("Querying Pinecone..."):
                resp = pinecone_idx().query(vector=vec, top_k=top_k, namespace=ns, include_metadata=True)

            matches = resp.get("matches", [])
            st.success(f"{len(matches)} results from namespace `{ns}`")

            for i, m in enumerate(matches):
                meta = m.get("metadata", {})
                score = m.get("score", 0)
                ts = ft(meta.get("timestamp_s"))
                text = meta.get("fact_text") or meta.get("caption") or "—"

                col_s, col_t, col_i = st.columns([1, 6, 2])
                col_s.metric("Score", f"{score:.4f}")
                col_t.markdown(f"**{text}**")
                col_i.caption(f"t={ts}")
                if meta.get("video_id"):
                    st.caption(f"video: `{meta['video_id'][:8]}…` · frame: `{meta.get('frame_id','')[:8]}…`")
                st.divider()

        except ImportError:
            st.warning("LocalEmbedder not available locally. Download the model first:\n```\npython -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')\"\n```")
        except Exception as e:
            st.error(str(e))
