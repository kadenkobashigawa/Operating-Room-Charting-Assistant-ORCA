import re
import os
import io
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from num2words import num2words
from tqdm import tqdm
from asr_v3 import transcribe_file



#dataset / HuggingFace config...
HF_DATASET = "aline-gassenn/MedDialog-Audio"
HF_BASE_URL = "https://huggingface.co/datasets/{}/resolve/main".format(HF_DATASET)
HF_API_URL = "https://huggingface.co/api/datasets/{}/tree/main".format(HF_DATASET)
METADATA_URL = "{}/metadata.csv".format(HF_BASE_URL)

#batch to sample from (one batch per run)...
BATCH = "batch_1"

#number of audio samples to evaluate per (noise_folder x model) combo...
NUM_SAMPLES  = 50

#local directory where audio files & output are temporarily saved...
REC_DIR = "recordings"
OUT_DIR = "asr_val_output"
os.makedirs(REC_DIR, exist_ok = True)
os.makedirs(OUT_DIR, exist_ok = True)

#whisper model sizes to evaluate...
MODELS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v1",
    "large-v3",
]

#noise-level folders to evaluate (relative paths inside the HF repo)...
NOISE_FOLDERS = [
    "noise-free audio",
    "white_noise/noise_2%",
    "white_noise/noise_6%",
    "white_noise/noise_10%",
    # "background_noise/noise_20%",
    # "background_noise/noise_40%",
    # "background_noise/noise_60%"
]


#accuracy helpers...
def normalize_text(text):

    '''
    lowercases text, converts digit sequences to their English word
    equivalents, and strips all punctuation.

    Parameters
    ----------
        text : str
            raw input string to normalize.

    Returns
    -------
        text : str
            cleaned string with digits as words and punctuation removed.
    '''

    #lowercase everything first...
    text = text.lower()

    #swap every run of digits for its spoken-word form...
    text = re.sub(r'\d+', lambda m: num2words(int(m.group())), text)

    #drop all punctuation characters...
    text = re.sub(r'[^\w\s]', '', text)

    return text


def word_error_rate(reference, hypothesis):

    '''
    computes the Word Error Rate (WER) between a reference and a hypothesis
    transcription using dynamic-programming edit distance.

    Parameters
    ----------
        reference : str
            the ground-truth transcription string.
        hypothesis : str
            the ASR model output string.

    Returns
    -------
        wer : float
            WER in [0, inf). 0.0 = perfect match.
    '''

    #normalise both strings before comparison...
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()

    #single-row DP — saves memory vs. full matrix when we only need WER...
    dp = list(range(len(hyp) + 1))

    for i, r in enumerate(ref, 1):
        new_dp = [i]
        for j, h in enumerate(hyp, 1):
            if r == h:
                new_dp.append(dp[j - 1])           #match — no cost...
            else:
                new_dp.append(1 + min(dp[j],       #deletion...
                                      new_dp[-1],  #insertion...
                                      dp[j - 1]))  #substitution...
        dp = new_dp

    wer = dp[-1] / max(len(ref), 1)
    return wer


def accuracy_score(wer):

    '''
    returns word-level accuracy as a percentage: (1 - WER) x 100.

    Parameters
    ----------
        wer : float
            word error rate from word_error_rate().

    Returns
    -------
        accuracy : float
            accuracy as a percentage, clamped to [0, 100].
    '''

    accuracy = max(0.0, min(100.0, round((1 - wer) * 100, 2)))
    return accuracy


def get_word_errors(reference, hypothesis):

    '''
    traces back through a full DP edit-distance matrix to identify every
    substitution, insertion, and deletion between reference and hypothesis.

    Parameters
    ----------
        reference : str
            ground-truth transcription string.
        hypothesis : str
            ASR model output string.

    Returns
    -------
        word_errors : list
            list of (error_type, ref_word, hyp_word) tuples in left-to-right
            order. error_type is "SUBSTITUTION", "INSERTION", or "DELETION".
    '''

    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()

    #build the full (len_ref+1) x (len_hyp+1) DP matrix...
    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]

    for i in range(len(ref) + 1):
        dp[i][0] = i #cost of deleting all ref words up to i...
    for j in range(len(hyp) + 1):
        dp[0][j] = j #cost of inserting all hyp words up to j...

    for i, r in enumerate(ref, 1):
        for j, h in enumerate(hyp, 1):
            if r == h:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],       #deletion...
                                    dp[i][j - 1],      #insertion...
                                    dp[i - 1][j - 1])  #substitution...

    #backtrace from bottom-right to top-left to collect errors...
    errors = []
    i, j = len(ref), len(hyp)

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1; j -= 1  #correct — skip...
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            errors.append(("SUBSTITUTION", ref[i - 1], hyp[j - 1]))
            i -= 1; j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            errors.append(("INSERTION", "-", hyp[j - 1]))
            j -= 1
        else:
            errors.append(("DELETION", ref[i - 1], "-"))
            i -= 1

    word_errors = list(reversed(errors))
    return word_errors


#data-fetching helpers...
def load_metadata():

    '''
    downloads metadata.csv from the HuggingFace dataset and returns a
    dict mapping filename to reference transcription.

    Returns
    -------
        meta_lookup : dict
            mapping of filename strings to ground-truth transcription strings.
    '''

    print("Fetching metadata.csv...")
    resp = requests.get(METADATA_URL)
    resp.raise_for_status()

    meta = pd.read_csv(io.StringIO(resp.text))
    meta.columns = meta.columns.str.strip()

    #build a quick-lookup dict: filename -> ground-truth transcription...
    meta_lookup = dict(zip(
        meta["filename"].str.strip(),
        meta["transcription"].str.strip()
    ))

    print("Loaded {} metadata rows".format(len(meta_lookup)))
    return meta_lookup


def fetch_sample_filenames(noise_folder):

    '''
    queries the HF dataset API to get the list of audio files in the
    specified noise_folder/BATCH directory and returns up to NUM_SAMPLES
    filenames that exist in the metadata lookup.

    Parameters
    ----------
        noise_folder : str
            relative path inside the repo, e.g.
            "noise-free audio" or "white_noise/noise_2%".

    Returns
    -------
        filenames : list
            list of bare filenames (no directory prefix). not pre-filtered
            against metadata — matching happens in evaluate_combination.
    '''

    #build the sub-path that combines noise folder and batch name...
    sub_path = "{}/{}".format(noise_folder, BATCH)
    encoded  = sub_path.replace("%", "%25").replace(" ", "%20")  #encode % first to avoid double-encoding spaces...

    print("Fetching file listing from {}...".format(sub_path))
    api_resp = requests.get("{}/{}?limit={}".format(HF_API_URL, encoded, NUM_SAMPLES * 2))
    api_resp.raise_for_status()

    files     = json.loads(api_resp.text)
    filenames = [os.path.basename(f["path"]) for f in files if f["type"] == "file"]

    print("Found {} files in {}.".format(len(filenames), sub_path))
    return filenames


def download_audio(noise_folder, filename):

    '''
    downloads a single audio file from HuggingFace and saves it to REC_DIR.

    Parameters
    ----------
        noise_folder : str
            repo-relative noise-level path.
        filename : str
            bare filename, e.g. "sample_001.wav".

    Returns
    -------
        local_path : str
            local filesystem path where the file was saved.
    '''

    sub_path = "{}/{}".format(noise_folder, BATCH)
    encoded  = sub_path.replace("%", "%25").replace(" ", "%20")  #encode % first to avoid double-encoding spaces...
    audio_url = "{}/{}/{}".format(HF_BASE_URL, encoded, filename)
    local_path = os.path.join(REC_DIR, filename)

    #skip download if file already exists locally...
    if not os.path.exists(local_path):
        tqdm.write("    Downloading: {}".format(filename))
        audio_resp = requests.get(audio_url)
        audio_resp.raise_for_status()

        with open(local_path, "wb") as f:
            f.write(audio_resp.content)

    return local_path


#evaluation loop...
def evaluate_combination(noise_folder, model_name, meta_lookup):

    '''
    runs the full download -> transcribe -> score pipeline for every sample
    in one (noise_folder, model_name) combination.

    Parameters
    ----------
        noise_folder : str
            repo-relative noise-level path.
        model_name : str
            whisper model size string, e.g. "small".
        meta_lookup : dict
            dict mapping filename to reference transcription.

    Returns
    -------
        results : list
            per-sample result dicts.
        avg_wer : float
            average WER across all samples (0-100).
        avg_acc : float
            average accuracy across all samples (0-100).
    '''

    #get filenames for this noise level...
    filenames = fetch_sample_filenames(noise_folder)

    #build a lookup from base ID (e.g. "100096_1") to metadata filename...
    base_to_meta = {re.sub(r'_[a-z]\d+\.wav$', '', k): k for k in meta_lookup}

    #match each noisy filename to its metadata entry via shared base ID...
    matched = []
    for fn in filenames:
        base = re.sub(r'_[a-z]\d+\.wav$', '', fn)
        if base in base_to_meta:
            matched.append((fn, base_to_meta[base]))

    if len(matched) < NUM_SAMPLES:
        print("  Warning: only {}/{} files matched metadata.".format(
            len(matched), NUM_SAMPLES))

    samples = matched[:NUM_SAMPLES]
    results = []

    for filename, meta_filename in tqdm(samples,
                                        desc = "  [{} | {}]".format(model_name, os.path.basename(noise_folder)),
                                        unit = "sample",
                                        total = len(samples),
                                        colour = "#2E75B6"):

        reference  = meta_lookup[meta_filename]
        local_path = download_audio(noise_folder, filename)

        try:
            hypothesis = transcribe_file(local_path, asr_model = model_name)
        except Exception as e:
            tqdm.write("    WARNING: transcription failed for {} — skipping. ({})".format(filename, e))
            continue

        wer = word_error_rate(reference, hypothesis)
        acc = accuracy_score(wer)

        results.append({
            "noise_folder": noise_folder,
            "model":        model_name,
            "file":         filename,
            "reference":    reference,
            "hypothesis":   hypothesis,
            "WER (%)":      round(wer * 100, 2),
            "accuracy (%)": acc,
        })
        tqdm.write("    WER: {:6.1f}%  |  Accuracy: {:6.2f}%".format(wer * 100, acc))

    avg_acc = sum(r["accuracy (%)"] for r in results) / len(results) if results else 0
    avg_wer = sum(r["WER (%)"]      for r in results) / len(results) if results else 0

    return results, avg_wer, avg_acc


#plotting...
def plot_results(summary_rows):

    '''
    generates a grouped bar chart of average accuracy for every
    (noise_folder, model) combination and saves it as a PNG.

    Parameters
    ----------
        summary_rows : list
            list of dicts with keys "noise_folder", "model",
            "avg_WER (%)", and "avg_accuracy (%)".
    '''

    df = pd.DataFrame(summary_rows)
    df.columns = df.columns.str.strip()

    #pivot so rows = noise levels, columns = models...
    pivot = df.pivot(index = "noise_folder", columns = "model", values = "avg_accuracy (%)")

    #keep the noise folders in the same order as NOISE_FOLDERS...
    pivot = pivot.reindex([nf for nf in NOISE_FOLDERS if nf in pivot.index])
    
    #reorder columns to match MODELS list order...
    ordered_models = [m for m in MODELS if m in pivot.columns]
    pivot = pivot[ordered_models]

    #colour palette — light to dark, correlates small to large model size...
    palette = [
        "#B8D9F5",  
        "#89BFE8",  
        "#5BA3D9",  
        "#2E75B6",  
        "#1F4E79", 
        "#0D2B45", 
    ]

    n_groups = len(pivot)
    n_bars = len(pivot.columns)
    x = np.arange(n_groups)
    width = 0.8 / n_bars   

    fig, ax = plt.subplots(figsize = (max(10, n_groups * 1.6), 7))

    for idx, model in enumerate(pivot.columns):
        offsets = x + (idx - n_bars / 2 + 0.5) * width
        ax.bar(offsets, pivot[model], width, label = model,
                color = palette[idx % len(palette)], edgecolor = "none",
                zorder = 3, rasterized = True)
    
    #x-tick labels: strip the directory prefix for readability...
    tick_labels = [
        nf.replace("background_noise/", "").replace("white_noise/", "").replace("noise-free audio", "Noise Free").replace('_', ' ').replace('n', 'N')
        for nf in pivot.index
    ]

    #formatting...
    ax.set_xlabel("Background Noise Condition", fontsize = 16, fontweight = "bold", labelpad = 8)
    ax.set_ylabel("Average Accuracy (%)", fontsize = 16, fontweight = "bold", labelpad = 8)
    ax.set_title("ASR Accuracy by Noise Level and Model", fontsize = 20, fontweight = "bold", pad = 16)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontweight = "bold")
    ax.tick_params(axis = "both", labelsize = 12)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_ylim(50, 140)
    ax.axhline(y = 100, linestyle = "-", linewidth = 2, color = 'k', alpha = 0.2)
    ax.legend(prop = {"size": 14, "weight": "bold"},
              title_fontsize = 14)
    ax.grid(axis = "y", linestyle = "--", alpha = 0.4, zorder = 0)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    out_png = os.path.join(OUT_DIR, "asr_accuracy_chart.png")
    plt.savefig(out_png, dpi = 150)
    print("\nChart saved to: {}".format(out_png))
    plt.show()


#main...
def asr_val():

    '''
    entry point. Iterates over every combination of NOISE_FOLDERS x MODELS,
    runs the ASR evaluation pipeline, prints a per-combination summary,
    saves all raw results to CSV, and plots a grouped bar chart.
    '''

    #load metadata once — shared across all combinations...
    meta_lookup = load_metadata()

    all_results  = []   #flat list of per-sample result dicts...
    summary_rows = []   #one row per (noise_folder, model) combo...

    #outer loop: noise conditions...
    for noise_folder in NOISE_FOLDERS:
        print("\n\n\n" + "=" * 60)
        print("NOISE FOLDER: {}".format(noise_folder))
        print("=" * 60)

        #inner loop: model sizes...
        for model_name in MODELS:
            print(f'\n\n\n{"-" * 60}')
            print("MODEL: {}".format(model_name))
            print("-" * 60)

            results, avg_wer, avg_acc = evaluate_combination(
                noise_folder, model_name, meta_lookup
            )
            all_results.extend(results)

            summary_rows.append({
                "noise_folder":     noise_folder,
                "model":            model_name,
                "avg_WER (%)":      round(avg_wer, 2),
                "avg_accuracy (%)": round(avg_acc, 2),
            })

            #print word-level error report for this combo...
            print(f"\n{'-' * 60}\nERROR REPORT\n{'-' * 60}\n")
            for r in results:
                errors = get_word_errors(r["reference"], r["hypothesis"])
                if errors:
                    print("  [{}]".format(r["file"]))
                    for err_type, ref_word, hyp_word in errors:
                        if err_type == "SUBSTITUTION":
                            print("    '{}' -> '{}'".format(ref_word, hyp_word))
                        elif err_type == "INSERTION":
                            print("    extra: '{}'".format(hyp_word))
                        elif err_type == "DELETION":
                            print("    missing: '{}'".format(ref_word))

    #print overall summary table...
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for row in summary_rows:
        print("  {:30s}  {:10s}  WER: {:6.2f}%  Acc: {:6.2f}%".format(
            row["noise_folder"], row["model"],
            row["avg_WER (%)"], row["avg_accuracy (%)"]
        ))

    #save raw per-sample results to CSV...
    out_csv = os.path.join(OUT_DIR, "asr_accuracy_results.csv")
    pd.DataFrame(all_results).to_csv(out_csv, index = False)
    print("\nRaw results saved to: {}".format(out_csv))

    #save per-combo summary to a second CSV...
    out_summary_csv = os.path.join(OUT_DIR, "asr_accuracy_summary.csv")
    pd.DataFrame(summary_rows).to_csv(out_summary_csv, index = False)
    print("Summary saved to: {}".format(out_summary_csv))

    #render and save the results chart...
    plot_results(summary_rows)