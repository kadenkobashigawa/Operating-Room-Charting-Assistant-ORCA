import json
import os
import time
from pathlib import Path
from cde_v5 import ClinicalDataExtractor
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


#models to benchmark...
MODELS = [
    'llama3.1:8b',
    'phi3.5:3.8b',
    'qwen2.5:7b',
    'mistral:7b-instruct',
]

#paths and settings — anchored to this script so CWD doesn't matter...
BASE_DIR        = Path(__file__).resolve().parent
TRANSCRIPTS_DIR = BASE_DIR / 'cde_val' / 'scripts'
REPORTS_DIR     = BASE_DIR / 'cde_val' / 'val_reports'
ANSWER_KEYS_DIR = BASE_DIR / 'cde_val' / 'answer_keys'
TEMPLATE_PATH   = BASE_DIR / 'report_template_1.json'
NUM_TRANSCRIPTS = 5

#values treated as equivalent to <none>...
NONE_EQUIVALENTS = {'<none>', 'none', 'null', 'n/a', 'na', 'not applicable', ''}


def sanitize_model_name(model: str) -> str:

    '''
    converts a model tag into a safe folder name.

    Parameters
    ----------
        model : str
            raw model tag string, e.g. "mistral:7b-instruct".

    Returns
    -------
        sanitized : str
            model tag with colons replaced by hyphens and dots by underscores,
            e.g. "mistral-7b-instruct".
    '''

    sanitized = model.replace(':', '-').replace('.', '_')
    return sanitized


def normalize(value) -> str:

    '''
    normalizes a value to a lowercase, stripped string for comparison.

    Parameters
    ----------
        value : any
            any value — string, list, None, or other.

    Returns
    -------
        normalized : str
            lowercase stripped string. Lists are joined with ", " before normalization.
            None returns an empty string.
    '''

    if isinstance(value, list):
        value = ', '.join(str(v) for v in value)
    if value is None:
        normalized = ''
        return normalized
    normalized = str(value).lower().strip()
    normalized = normalized.rstrip('.,;:')
    normalized = normalized.strip()
    return normalized


def is_none_value(value) -> bool:

    '''
    checks whether a value is semantically empty or equivalent to <none>.

    Parameters
    ----------
        value : any
            any value to check against the NONE_EQUIVALENTS set.

    Returns
    -------
        is_none : bool
            True if the normalized value is in NONE_EQUIVALENTS, False otherwise.
    '''

    is_none = normalize(value) in NONE_EQUIVALENTS
    return is_none


def flatten_dict(d: dict, parent_key: str = '') -> dict:

    '''
    recursively flattens a nested dictionary into dot-notated keys.

    Parameters
    ----------
        d : dict
            dictionary to flatten, may be arbitrarily nested.
        parent_key : str
            key prefix accumulated during recursion, empty at top level.

    Returns
    -------
        items : dict
            flat dictionary mapping dot-notated key strings to their leaf values,
            e.g. {"patient_info.patient_name": "John Doe"}.
    '''

    items = {}
    for k, v in d.items():
        full_key = f'{parent_key}.{k}' if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, full_key))
        else:
            items[full_key] = v
    return items


def compare_field(ak_value, extracted_value) -> dict:

    '''
    compares a single extracted field against its answer key value.

    Parameters
    ----------
        ak_value : any
            the expected value from the answer key.
        extracted_value : any
            the value returned by the model extraction.

    Returns
    -------
        result : dict
            dict with keys "verdict", "answer_key_raw", and "extracted_raw".
            Verdict is one of "correct", "incorrect", or "review_needed".
    '''

    ak_norm = normalize(ak_value)
    ex_norm = normalize(extracted_value)

    ak_is_none = is_none_value(ak_value)
    ex_is_none = is_none_value(extracted_value)

    #both are none equivalents -> correct...
    if ak_is_none and ex_is_none:
        result = {
            'verdict': 'correct',
            'answer_key_raw': ak_value,
            'extracted_raw':  extracted_value,
        }
        return result

    #answer key is none but model returned something -> incorrect...
    if ak_is_none and not ex_is_none:
        result = {
            'verdict': 'incorrect',
            'answer_key_raw': ak_value,
            'extracted_raw':  extracted_value,
        }
        return result

    #normalized exact match -> correct...
    if ak_norm == ex_norm:
        result = {
            'verdict': 'correct',
            'answer_key_raw': ak_value,
            'extracted_raw':  extracted_value,
        }
        return result

    #no match -> flag for manual review with unedited values side by side...
    result = {
        'verdict': 'review_needed',
        'answer_key_raw': ak_value,
        'extracted_raw':  extracted_value,
    }
    return result


def load_transcript(transcript_id: int) -> str:

    '''
    loads a raw transcript text file by its numeric ID.

    Parameters
    ----------
        transcript_id : int
            integer ID corresponding to a transcript file,
            e.g. 1 maps to "transcript1.txt".

    Returns
    -------
        transcript : str
            full transcript text as a string.

    Raises
    ------
        FileNotFoundError
            if the transcript file does not exist.
    '''

    path = TRANSCRIPTS_DIR / f'transcript{transcript_id}.txt'
    if not path.exists():
        raise FileNotFoundError(f'Transcript not found: {path}')
    transcript = path.read_text(encoding = 'utf-8')
    return transcript


def load_answer_key(transcript_id: int) -> dict:

    '''
    loads the answer key JSON for a given transcript ID.

    Parameters
    ----------
        transcript_id : int
            integer ID corresponding to an answer key file,
            e.g. 1 maps to "ak1.json".

    Returns
    -------
        answer_key : dict
            parsed answer key as a dictionary.

    Raises
    ------
        FileNotFoundError
            if the answer key file does not exist.
    '''

    path = ANSWER_KEYS_DIR / f'ak{transcript_id}.json'
    if not path.exists():
        raise FileNotFoundError(f'Answer key not found: {path}')
    with open(path, 'r', encoding = 'utf-8') as f:
        answer_key = json.load(f)
    return answer_key


def load_latest_report(model_dir: Path, transcript_id: int) -> dict:

    '''
    loads the extracted report corresponding to a given transcript ID from a model's output directory.

    Parameters
    ----------
        model_dir : Path
            path to the directory containing the model's extracted report JSON files.
        transcript_id : int
            1-based index used to select the correct report after sorting by timestamp token.

    Returns
    -------
        report : dict
            parsed extracted report as a dictionary.
        chosen : Path
            filesystem path of the report that was loaded.

    Raises
    ------
        FileNotFoundError
            if no reports exist in the directory or fewer reports exist than transcript_id.
    '''

    #limit to the most-recent NUM_TRANSCRIPTS reports so stale reports from
    #previous test_models() runs don't shadow the latest results...
    candidates = sorted(model_dir.glob('extracted_report_*.json'), key = lambda p: p.stat().st_mtime)
    candidates = candidates[-NUM_TRANSCRIPTS:]
    if not candidates:
        raise FileNotFoundError(f'No reports found in {model_dir}')
    if len(candidates) < transcript_id:
        raise FileNotFoundError(f'Expected at least {transcript_id} reports in {model_dir}, found {len(candidates)}')
    chosen = candidates[transcript_id - 1]
    with open(chosen, 'r', encoding = 'utf-8') as f:
        report = json.load(f)
    return report, chosen


def grade_transcript(ak_flat: dict, extracted_flat: dict) -> dict:

    '''
    grades all fields for a single transcript by comparing flattened extraction output to the answer key.

    Parameters
    ----------
        ak_flat : dict
            flattened answer key dict with dot-notated keys.
        extracted_flat : dict
            flattened model extraction dict with dot-notated keys.

    Returns
    -------
        grade : dict
            dict with keys "summary" and "fields". Summary contains total_fields, correct,
            incorrect, review_needed, and accuracy_pct. Fields maps each key to its
            compare_field result dict.
    '''

    #grade all fields for one transcript, return per-field results and summary...
    all_keys = set(ak_flat.keys()) | set(extracted_flat.keys())
    field_results = {}
    counts = {'correct': 0, 'incorrect': 0, 'review_needed': 0}

    for key in sorted(all_keys):
        ak_val = ak_flat.get(key, ['<none>'])
        ex_val = extracted_flat.get(key)

        #field absent from extraction -> treat as <none> and compare normally...
        if ex_val is None:
            ex_val = ['<none>']

        result = compare_field(ak_val, ex_val)
        field_results[key] = result
        counts[result['verdict']] += 1

    total = sum(counts.values())
    auto_acc = round(counts['correct'] / total * 100, 1) if total else 0

    summary = {
        'total_fields':          total,
        'correct':               counts['correct'],
        'incorrect':             counts['incorrect'],
        'review_needed':         counts['review_needed'],
        'accuracy_pct':     auto_acc,
    }
    grade = {'summary': summary, 'fields': field_results}
    return grade


def review_fields(grade: dict, model: str, transcript_id: int) -> dict:

    '''
    runs an interactive terminal review session for all fields flagged as review_needed.

    Parameters
    ----------
        grade : dict
            grade dict returned by grade_transcript, modified in place.
        model : str
            model name string, used for display only.
        transcript_id : int
            transcript ID integer, used for display only.

    Returns
    -------
        grade : dict
            updated grade dict with review_needed fields resolved to correct or incorrect,
            summary counts adjusted, and accuracy_pct recalculated.
    '''

    #interactively review flagged fields in terminal, all at once with live progress bar...
    flagged = {k: v for k, v in grade['fields'].items() if v['verdict'] == 'review_needed'}
    if not flagged:
        return grade

    fields_list = list(flagged.items())
    total_flagged = len(fields_list)
    decisions = {}

    for i, (field, _) in enumerate(fields_list):

        #clear terminal and reprint all fields on every correction...
        os.system('clear')

        print(f'  Manual Review — {model} | Transcript {transcript_id}')
        print(f'  ENTER = correct   x = incorrect')
        print(f'  {"─" * 56}\n')

        for j, (f, r) in enumerate(fields_list):
            if j < i:
                marker = '✔' if decisions[f] == 'correct' else '🅇'
                print(f'  [{marker}] {f}')
            elif j == i:
                print(f'  [?] {f}')
                print(f'      AK:        {r["answer_key_raw"]}')
                print(f'      Extracted: {r["extracted_raw"]}')
            else:
                print(f'  [ ] {f}')

        #progress bar showing review completion...
        done = i
        bar_len = 40
        filled = int(bar_len * done / total_flagged)
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f'\n  [{bar}] {done}/{total_flagged}')

        try:
            response = input(f'\n  > ').strip().lower()
        except KeyboardInterrupt:

            #abort mid-review cleanly — leave remaining fields as review_needed...
            print('\n\n  Review interrupted — remaining fields left as review_needed.')
            break

        #log grade...
        decisions[field] = 'incorrect' if response == 'x' else 'correct'

    #final screen showing all results...
    os.system('clear')
    print(f'  Manual Review — {model} | Transcript {transcript_id} — Complete')
    print(f'  {"─" * 56}\n')
    for field, verdict in decisions.items():
        marker = '✔' if verdict == 'correct' else '🅇'
        print(f'  [{marker}] {field}')
    print(f'\n  [{"█" * 40}] {total_flagged}/{total_flagged}\n')

    #apply decisions to grade...
    for field, verdict in decisions.items():
        grade['fields'][field]['verdict'] = verdict
        if verdict == 'incorrect':
            grade['summary']['incorrect'] += 1
        else:
            grade['summary']['correct'] += 1
        grade['summary']['review_needed'] -= 1

    #recalculate accuracy after manual review...
    total = grade['summary']['total_fields']
    grade['summary']['accuracy_pct'] = round(grade['summary']['correct'] / total * 100, 1) if total else 0
    grade['summary']['note'] = 'accuracy_pct includes manual review results'
    return grade


def test_models():

    '''
    runs the full extraction pipeline for every model across all transcripts and saves reports to disk.
    '''

    #ensure top-level output directory exists...
    REPORTS_DIR.mkdir(parents = True, exist_ok = True)

    total_runs = len(MODELS) * NUM_TRANSCRIPTS
    run = 0

    for model in MODELS:
        print(f'\n{"=" * 60}')
        print(f'Model: {model}')
        print(f'{"=" * 60}')

        #instantiate extractor once per model to avoid reloading weights...
        model_dir = REPORTS_DIR / sanitize_model_name(model)
        model_dir.mkdir(parents = True, exist_ok = True)

        for t_id in range(1, NUM_TRANSCRIPTS + 1):
            run += 1
            print(f'\n[{run}/{total_runs}] {model} ┈┈→ transcript{t_id}.txt ... ')

            try:
                #load raw transcript text...
                transcript_chunk = load_transcript(t_id)

                #run extraction...
                cde = ClinicalDataExtractor(
                    cde_folder_path = model_dir,
                    cde_model = model,
                )
                cde.extract(
                    transcript_chunk = transcript_chunk,
                    template_path = TEMPLATE_PATH,
                )

            except FileNotFoundError as e:

                #skip missing transcripts without crashing the run...
                print(f'🅇  SKIP - {e}')

            except Exception as e:

                #catch ollama errors, model load failures, etc. without aborting the run...
                print(f'🅇  ERROR - {e}')

    print(f'\n{"=" * 60}')
    print(f'Done. Reports saved under: {REPORTS_DIR.resolve()}')
    print(f'{"=" * 60}\n')


def grade_models():

    '''
    grades all models across all transcripts, runs manual review for flagged fields,
        and saves per-transcript grade reports and a model-level rollup to disk.
    '''

    #grade all models across all transcripts and save per-transcript grade reports...
    for model in MODELS:
        print(f'\n{"=" * 60}')
        print(f'Grading: {model}')
        print(f'{"=" * 60}')

        model_dir = REPORTS_DIR / sanitize_model_name(model)
        grades_dir = model_dir / 'grades'
        grades_dir.mkdir(parents = True, exist_ok = True)

        model_totals = {'correct': 0, 'incorrect': 0, 'review_needed': 0, 'total': 0}

        for t_id in tqdm(range(1, NUM_TRANSCRIPTS + 1), desc = f'  {model}', unit = 'transcript'):

            try:
                #load answer key and latest extracted report...
                ak_data = load_answer_key(t_id)
                report, report_path = load_latest_report(model_dir, t_id)
                extracted_data = report

                #flatten both to dot-notated dicts for field-level comparison...
                ak_flat = flatten_dict(ak_data)
                extracted_flat = flatten_dict(extracted_data) if extracted_data else {}

                #grade this transcript...
                grade = grade_transcript(ak_flat, extracted_flat)

                #run interactive manual review for flagged fields...
                grade = review_fields(grade, model, t_id)

                #attach metadata...
                grade['meta'] = {
                    'model':         model,
                    'transcript_id': t_id,
                    'graded_at':     time.strftime('%Y-%m-%d %H:%M:%S'),
                    'source_report': str(report_path),
                }

                #save grade report...
                out_path = grades_dir / f'transcript{t_id}_grade.json'
                with open(out_path, 'w', encoding = 'utf-8') as f:
                    json.dump(grade, f, indent = 4)

                #accumulate model-level totals (explicit per-key so a future
                #rename of any summary field surfaces as a KeyError, not a silent zero)...
                s = grade['summary']
                model_totals['correct']       += s['correct']
                model_totals['incorrect']     += s['incorrect']
                model_totals['review_needed'] += s['review_needed']
                model_totals['total']         += s['total_fields']

                print(f'✔  {s["correct"]}/{s["total_fields"]} correct  |  {s["review_needed"]} to review  →  {out_path}')

            except FileNotFoundError as e:
                print(f'🅇  SKIP - {e}')

            except Exception as e:
                print(f'🅇  ERROR - {e}')

        #save model-level rollup across all 5 transcripts...
        total = model_totals['total']
        rollup = {
            'model':                 model,
            'total_fields_graded':   total,
            'correct':               model_totals['correct'],
            'incorrect':             model_totals['incorrect'],
            'review_needed':         model_totals['review_needed'],
            'accuracy_pct':     round(model_totals['correct'] / total * 100, 1) if total else 0,
        }
        rollup_path = grades_dir / 'model_rollup.json'
        with open(rollup_path, 'w', encoding = 'utf-8') as f:
            json.dump(rollup, f, indent = 4)

        print(f'\n  Rollup → {rollup_path}')
        print(f'  Auto accuracy: {rollup["accuracy_pct"]}%  |  {rollup["review_needed"]} fields pending manual review')

    print(f'\n{"=" * 60}')
    print(f'Grading complete.')
    print(f'{"=" * 60}\n')


#plotting...
def plot_benchmark():

    '''
    loads each model's rollup JSON from disk and renders a horizontal
    bar chart of extraction accuracy, sorted descending by accuracy.

    Parameters
    ----------
        None

    Returns
    -------
        None
            saves the chart as a PNG to REPORTS_DIR and calls plt.show().
    '''

    #colour palette — dark to light, one bar per model...
    palette = [
        "#1F4E79", 
        "#2E75B6",
        "#5BA3D9",
        "#89BFE8",
    ]

    #load rollup for each model...
    rows = []
    for model in MODELS:
        rollup_path = REPORTS_DIR / sanitize_model_name(model) / 'grades' / 'model_rollup.json'
        if not rollup_path.exists():
            print(f'  WARNING: missing rollup for {model} — skipping.')
            continue
        with open(rollup_path, 'r', encoding = 'utf-8') as f:
            data = json.load(f)
        rows.append({
            'model':        model,
            'accuracy_pct': data.get('accuracy_pct', 0),
            'correct':      data.get('correct', 0),
            'total':        data.get('total_fields_graded', 0),
        })

    if not rows:
        print('No rollup data found — run grade_models() first.')
        return

    #sort descending by accuracy so highest-performing model is on top...
    rows.sort(key = lambda r: r['accuracy_pct'], reverse = True)

    models   = [r['model']        for r in rows]
    accuracy = [r['accuracy_pct'] for r in rows]
    colors   = [palette[i % len(palette)] for i in range(len(rows))]

    y = np.arange(len(models))

    fig, ax = plt.subplots(figsize = (10, max(4, len(models) * 1.2)))

    #draw bars...
    bars = ax.barh(y, accuracy, color = colors, edgecolor = 'none', zorder = 3)

    #annotate each bar with its accuracy value...
    for bar, row in zip(bars, rows):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f'{row["accuracy_pct"]}%  ({row["correct"]}/{row["total"]})',
            va = 'center', ha = 'left', fontsize = 11, fontweight = 'bold',
        )

    #formatting...
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize = 12, fontweight = 'bold')
    ax.set_xlabel('Extraction Accuracy (%)', fontsize = 14, fontweight = 'bold', labelpad = 8)
    ax.set_title('CDE Benchmark: Extraction Accuracy by Model', fontsize = 16, fontweight = 'bold', pad = 14)
    ax.set_xlim(0, 115)
    ax.tick_params(axis = 'both', labelsize = 11)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    ax.axvline(x = 100, linestyle = '-', linewidth = 1.5, color = 'k', alpha = 0.2)
    ax.grid(axis = 'x', linestyle = '--', alpha = 0.4, zorder = 0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout()

    out_png = REPORTS_DIR / 'cde_benchmark_chart.png'
    plt.savefig(out_png, dpi = 150)
    print(f'\nChart saved to: {out_png}')
    plt.show()