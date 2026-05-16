import json
import threading
import traceback
import os
from pathlib import Path
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from datetime import datetime

from rec_v2 import record_audio
from asr_v3 import transcribe_file
from cde_v5 import ClinicalDataExtractor



app = Flask(__name__)
CORS(app)

WAV_PATH = None
ASR_MODEL = 'large-v3'
LLM_MODEL = 'llama3.1:8b'
TEMPLATE_PATH = 'report_template_1.json'

state = {
    "phase": "idle",
    "audio_path": None,
    "transcript": "",
    "transcript_words": [],
    "cde_result": None,
    "error": None,
}
_lock = threading.RLock()
_record_stop_event = threading.Event()



def set_state(**kwargs):

    '''
    thread-safe update of the shared pipeline state dict.

    Parameters
    ----------
        **kwargs : any
            key-value pairs to update in state.
    '''

    with _lock:
        state.update(kwargs)


@app.get("/status")
def status():

    '''
    returns the current pipeline state as JSON.
    '''

    with _lock:
        return jsonify(dict(state))


@app.get("/transcript")
def transcript():

    '''
    returns the current transcript text as JSON.
    '''

    with _lock:
        return jsonify({"transcript": state["transcript"]})


@app.get("/cde")
def cde():

    '''
    returns the current CDE result as JSON.
    '''

    with _lock:
        return jsonify({"cde_result": state["cde_result"]})


@app.get("/logo")
def logo():

    '''
    serves the ORCA logo image.
    '''

    #return 404 if the logo file is missing rather than crashing...
    if not os.path.exists("orai_logo.png"):
        return jsonify({"error": "Logo not found"}), 404
    return send_file("orai_logo.png", mimetype="image/png")


@app.post("/stop")
def stop():

    '''
    signals the recording to stop.
    '''

    with _lock:
        if state["phase"] != "recording":
            return jsonify({"error": "Not currently recording"}), 400
    _record_stop_event.set()
    return jsonify({"status": "stop signal sent"})


@app.post("/reset")
def reset():

    '''
    resets the pipeline state back to idle.
    '''

    _record_stop_event.set()
    set_state(phase = "idle", audio_path = None, transcript = "",
              transcript_words = [], cde_result = None, error = None)
    return jsonify({"status": "reset"})


@app.post("/start")
def start():

    '''
    starts the pipeline — records audio, transcribes, and extracts clinical data.
    '''

    with _lock:
        if state["phase"] not in ("idle", "done", "error"):
            return jsonify({"error": "Pipeline already running"}), 400
        state.update(phase = "recording", audio_path = None, transcript = "",
                     transcript_words = [], cde_result = None, error = None)
    _record_stop_event.clear()

    def _run():
        try:
            #step 1 — record. WAV_PATH (replay) > local mic...
            if WAV_PATH:
                audio_path = WAV_PATH
                set_state(audio_path = str(audio_path))
            else:
                audio_path = record_audio(stop_event = _record_stop_event)
                set_state(audio_path = str(audio_path))

            #step 2 — transcribe (stream segments into state)...
            set_state(phase = "transcribing")
            def _on_words(words):
                plain = ''.join(w['word'] for w in words).strip()
                set_state(transcript = plain, transcript_words = words)
            tx_text = transcribe_file(
                audio_path,
                asr_model = ASR_MODEL,
                on_segment = _on_words
            )
            set_state(transcript = tx_text)

            #step 3 — clinical data extraction...
            set_state(phase = "extracting")
            cde = ClinicalDataExtractor(cde_model = LLM_MODEL, cde_folder_path = 'reports')
            _extract_done = threading.Event()

            def _do_extract():
                cde.extract(tx_text, TEMPLATE_PATH, show_output = True)
                _extract_done.set()

            threading.Thread(target = _do_extract, daemon = True).start()

            while not _extract_done.wait(timeout = 1.0):
                try:
                    p = Path(cde.report_path)
                    if p.exists() and p.stat().st_size > 0:
                        with open(p, 'r', encoding = 'utf-8') as f:
                            text = f.read().strip()
                            partial = json.loads(text)
                        if partial:
                            set_state(cde_result = partial)
                except Exception:
                    pass

            #final read...
            with open(cde.report_path, 'r', encoding='utf-8') as f:
                extracted = json.load(f)

            set_state(phase = "done", cde_result = extracted)

        except Exception as exc:
            traceback.print_exc()
            set_state(phase = "error", error = str(exc))

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "recording started"})


@app.post("/save_session")
def save_session():

    '''
    saves the current session to disk as a Markdown and JSON file.
    '''

    data = request.json or {}
    #strip path separators to prevent traversal outside the sessions/ folder;
    #append a timestamp to the "unknown" fallback so unidentified saves don't collide...
    raw_id      = data.get("session_id") or f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_id  = os.path.basename(raw_id)
    duration    = data.get("duration", "00:00")
    timestamp   = data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    transcript  = data.get("transcript", "")
    cde_result  = data.get("cde_result", {})

    os.makedirs("sessions", exist_ok=True)
    filepath = f"sessions/{session_id}.md"

    section_labels = {
        "patient_info":          "Patient Information",
        "surgical_team":         "Surgical Team",
        "procedure":             "Procedure",
        "pre_op":                "Pre-Operative",
        "intra_op":              "Intra-Operative",
        "medications":           "Medications",
        "supplies_and_equipment":"Supplies & Equipment",
        "specimens":             "Specimens",
        "closure":               "Closure",
        "post_op":               "Post-Operative",
    }

  
    def _sanitize(val):
        if isinstance(val, list):
            text = ", ".join(str(v) for v in val) if val else "<none>"
        else:
            text = str(val) if val else "<none>"
        return " ".join(text.split())

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# ORCA Session: {session_id}\n\n")
        f.write("| Field | Value |\n")
        f.write("|-------|-------|\n")
        f.write(f"| Date / Time | {timestamp} |\n")
        f.write(f"| Session Duration | {duration} |\n")
        f.write(f"| Session ID | {session_id} |\n\n")
        f.write("---\n\n")

        f.write("## Clinical Data Extraction\n\n")
        for sec_key, sec_label in section_labels.items():
            fields = cde_result.get(sec_key, {})
            if not isinstance(fields, dict):
                continue
            f.write(f"### {sec_label}\n\n")
            f.write("| Field | Value |\n")
            f.write("|-------|-------|\n")
            for field_key, val in fields.items():
                label = field_key.replace("_", " ").title()
                f.write(f"| {label} | {_sanitize(val)} |\n")
            f.write("\n")

        f.write("---\n\n")
        f.write("## Transcript\n\n")
        f.write("```\n")
        f.write(transcript.strip() + "\n")
        f.write("```\n")

    #also save / update the JSON report so user edits are persisted...
    json_path = f"sessions/{session_id}.json"
    json_payload = {
        "session_id":  session_id,
        "timestamp":   timestamp,
        "duration":    duration,
        "transcript":  transcript,
        "cde_result":  cde_result,
    }
    with open(json_path, "w", encoding = "utf-8") as jf:
        json.dump(json_payload, jf, indent = 4)
    return jsonify({"status": "saved", "path": filepath, "json_path": json_path})


