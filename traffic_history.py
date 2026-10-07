import sqlite3
import os
import zlib
import json
import threading
from typing import Optional, List, Dict, Any, Tuple
import datetime
import numpy as np

DB_PATH = os.environ.get("IDS_TRAFFIC_HISTORY_DB", "runtime_state/traffic_history.sqlite3")

_local = threading.local()

def get_db():
    if not hasattr(_local, "conn"):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _local.conn = sqlite3.connect(DB_PATH, timeout=10.0, isolation_level=None)
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA synchronous=NORMAL")
        _local.conn.execute("PRAGMA busy_timeout=5000")
        _setup_schema(_local.conn)
    return _local.conn

def _setup_schema(conn):
    try:
        conn.execute('ALTER TABLE flow_history ADD COLUMN controller_mode TEXT')
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    conn.execute('''
        CREATE TABLE IF NOT EXISTS flow_history (
            event_id TEXT PRIMARY KEY,
            timestamp_utc TEXT,
            source_ip TEXT,
            destination_ip TEXT,
            data_profile TEXT,
            role TEXT,
            sample_id INTEGER,
            binary_prediction INTEGER,
            action TEXT,
            confidence_meaning TEXT,
            defense TEXT,
            used_intensity REAL,
            ground_truth_status TEXT,
            controller_triggered INTEGER,
            controller_state_json TEXT,
            controller_mode TEXT,
            session_id TEXT,
            review_status TEXT,
            analyst_notes TEXT,
            received_vector BLOB,
            classifier_input_vector BLOB,
            rs_member_vectors BLOB,
            rs_member_preds BLOB
        )
    ''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON flow_history(timestamp_utc DESC)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_session ON flow_history(session_id)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_defense ON flow_history(defense)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_action ON flow_history(action)')

def serialize_vector(vec: np.ndarray) -> Optional[bytes]:
    if vec is None:
        return None
    return zlib.compress(vec.astype(np.float32).tobytes())

def deserialize_vector(blob: bytes, shape=None) -> Optional[np.ndarray]:
    if blob is None:
        return None
    vec = np.frombuffer(zlib.decompress(blob), dtype=np.float32)
    if shape:
        vec = vec.reshape(shape)
    return vec

def save_flow_event(event: dict):
    try:
        conn = get_db()
        
        received_blob = serialize_vector(event.get("received_vector"))
        classifier_input_blob = serialize_vector(event.get("classifier_input_vector"))
        rs_members_blob = serialize_vector(event.get("rs_member_vectors"))
        rs_preds_blob = serialize_vector(event.get("rs_member_preds"))

        conn.execute('''
            INSERT INTO flow_history (
                event_id, timestamp_utc, source_ip, destination_ip, data_profile, role,
                sample_id, binary_prediction, action, confidence_meaning, defense, used_intensity,
                ground_truth_status, controller_triggered, controller_state_json, controller_mode, session_id,
                review_status, analyst_notes, received_vector, classifier_input_vector, rs_member_vectors, rs_member_preds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event["event_id"],
            event["timestamp_utc"],
            event["source_ip"],
            event["destination_ip"],
            event["data_profile"],
            event["role"],
            event.get("sample_id"),
            event["binary_prediction"],
            event["action"],
            event["confidence_meaning"],
            event["defense"],
            event["used_intensity"],
            event["ground_truth_status"],
            event.get("controller_triggered", 0),
            json.dumps(event.get("controller_state", {})),
            event.get("controller_mode"),
            event.get("session_id"),
            "",
            "",
            received_blob,
            classifier_input_blob,
            rs_members_blob,
            rs_preds_blob
        ))
    except Exception as e:
        import logging
        logging.error(f"Failed to save flow history: {e}")

def get_history(limit=50, offset=0, session_id=None, role=None, action=None, defense=None, is_attack=None, query_only=False):
    conn = get_db()
    query = "SELECT event_id, timestamp_utc, source_ip, destination_ip, data_profile, role, sample_id, binary_prediction, action, confidence_meaning, defense, used_intensity, ground_truth_status, session_id, review_status FROM flow_history WHERE 1=1"
    params = []
    
    if session_id:
        query += " AND session_id = ?"
        params.append(session_id)
    if role:
        query += " AND role = ?"
        params.append(role)
    if action:
        query += " AND action = ?"
        params.append(action)
    if defense:
        query += " AND defense = ?"
        params.append(defense)
    if is_attack is not None:
        if is_attack:
            query += " AND (ground_truth_status = 'Simulator: 1' OR ground_truth_status = 'Dataset: 1')"
        else:
            query += " AND (ground_truth_status = 'Simulator: 0' OR ground_truth_status = 'Dataset: 0')"
    if query_only:
        query += " AND role = 'Query'"
    else:
        # Default to NOT query (Measured) unless explicit
        if role is None:
            query += " AND role != 'Query'"
            
    count_query = query.replace("SELECT event_id, timestamp_utc, source_ip, destination_ip, data_profile, role, sample_id, binary_prediction, action, confidence_meaning, defense, used_intensity, ground_truth_status, session_id, review_status", "SELECT COUNT(*)")
    
    query += " ORDER BY timestamp_utc DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    cur = conn.execute(query, params)
    cols = [column[0] for column in cur.description]
    results = [dict(zip(cols, row)) for row in cur.fetchall()]
    
    total = conn.execute(count_query, params[:-2]).fetchone()[0]
    
    return results, total

def get_event(event_id):
    conn = get_db()
    cur = conn.execute("SELECT * FROM flow_history WHERE event_id = ?", (event_id,))
    row = cur.fetchone()
    if not row:
        return None
    cols = [column[0] for column in cur.description]
    event = dict(zip(cols, row))
    
    event["received_vector"] = deserialize_vector(event["received_vector"])
    event["classifier_input_vector"] = deserialize_vector(event["classifier_input_vector"])
    
    if event["rs_member_vectors"]:
        rs_mem = deserialize_vector(event["rs_member_vectors"])
        event["rs_member_vectors"] = rs_mem.reshape(-1, 78)
        
    if event["rs_member_preds"]:
        event["rs_member_preds"] = deserialize_vector(event["rs_member_preds"])
        
    event["controller_state"] = json.loads(event["controller_state_json"])
    
    return event

def update_review(event_id, status, notes):
    conn = get_db()
    conn.execute("UPDATE flow_history SET review_status = ?, analyst_notes = ? WHERE event_id = ?", (status, notes, event_id))

def get_sessions():
    conn = get_db()
    cur = conn.execute('''
        SELECT session_id, 
               MAX(controller_mode) as controller_mode,
               MIN(timestamp_utc) as start_time, 
               MAX(timestamp_utc) as end_time, 
               COUNT(CASE WHEN role = 'Query' THEN 1 END) as query_count,
               COUNT(CASE WHEN role != 'Query' THEN 1 END) as target_count,
               COUNT(*) as total_flows,
               SUM(CASE WHEN role != 'Query' AND (ground_truth_status = 'Simulator: 1' OR ground_truth_status = 'Dataset: 1') AND binary_prediction = 1 THEN 1 ELSE 0 END) as tp,
               SUM(CASE WHEN role != 'Query' AND (ground_truth_status = 'Simulator: 0' OR ground_truth_status = 'Dataset: 0') AND binary_prediction = 1 THEN 1 ELSE 0 END) as fp,
               SUM(CASE WHEN role != 'Query' AND (ground_truth_status = 'Simulator: 0' OR ground_truth_status = 'Dataset: 0') AND binary_prediction = 0 THEN 1 ELSE 0 END) as tn,
               SUM(CASE WHEN role != 'Query' AND (ground_truth_status = 'Simulator: 1' OR ground_truth_status = 'Dataset: 1') AND binary_prediction = 0 THEN 1 ELSE 0 END) as fn
        FROM flow_history 
        GROUP BY session_id 
        ORDER BY start_time DESC
    ''')
    cols = [column[0] for column in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]

def close_db():
    if hasattr(_local, "conn"):
        _local.conn.close()
        delattr(_local, "conn")
