"""
Colab all-in-one runner: produce required transcripts and auto-download results.
Run in Google Colab (GPU): python colab_run_all.py
"""
import os
import sys
import subprocess
import time
import glob
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

LOG = RESULTS / "colab_run_all.log"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    sys.stdout.write(line)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line)


def run(cmd, env=None):
    if isinstance(cmd, str):
        cmd_list = cmd.split()
    else:
        cmd_list = list(cmd)
    log(f"RUN: {' '.join(cmd_list)}")
    e = os.environ.copy()
    if env:
        e.update(env)
    p = subprocess.run(
        cmd_list,
        cwd=str(ROOT),
        env=e,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(p.stdout)
    if p.returncode != 0:
        log(f"FAIL rc={p.returncode}")
        raise SystemExit(p.returncode)
    log(f"OK rc=0")
    return p.stdout


def main():
    log("=== START colab_run_all ===")
    try:
        run([sys.executable, "step7_sweep.py"], env={**os.environ, "MESE_DRY": "1"})
    except Exception as ex:
        log(f"DRY failed (continuing? {ex})")

    run([sys.executable, "step8_dose.py"], env={**os.environ, "MESE_SEED": "1"})

    run(
        [sys.executable, "step7_sweep.py"],
        env={
            **os.environ,
            "MESE_SHAM": "20",
            "MESE_LAYERS": "14,20,26",
            "MESE_POOL": "mean,last",
        },
    )

    for t in [
        "tests/test_steering_collapse.py",
        "tests/test_frame_leak.py",
        "tests/test_grouped_cv.py",
    ]:
        try:
            run([sys.executable, t])
        except Exception as ex:
            log(f"TEST {t} failed: {ex}")

    log("=== DONE ===")
    needed = []
    for pat in [
        "results/step8_*.txt",
        "results/step7.txt",
        "results/step7_dry.txt",
        "results/colab_run_all.log",
    ]:
        needed.extend(glob.glob(str(ROOT / pat.replace("results/", "")), recursive=False))
        needed.extend(glob.glob(str(ROOT / pat)))
    needed = sorted(set(needed))
    log("NEEDED: " + " ".join(os.path.relpath(x, ROOT) for x in needed))

    zip_path = ROOT / "results_to_return.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in needed:
            try:
                z.write(f, arcname=os.path.relpath(f, ROOT))
            except Exception:
                pass
    log(f"ZIP: {zip_path.name}")

    try:
        from google.colab import files

        files.download(str(zip_path))
        log("DOWNLOADED results_to_return.zip")
    except Exception as e:
        log(f"NO_COLAB_DOWNLOAD ({e})")
        log("Download results_to_return.zip manually from Colab Files panel")


if __name__ == "__main__":
    main()
