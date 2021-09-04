import argparse
import collections
import json
import logging
import os
import pathlib
import stat
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import performance as perf

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Load files")
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path.cwd(),
                        help="Working directory with DOT Iowa Accident videos")
    args = parser.parse_args()

    REPO_PATH = pathlib.Path(__file__).resolve().parent / "NVIDIA_AICITY"
    sys.path.append(os.fspath(REPO_PATH))
    DARKNET_PATH = pathlib.Path.cwd() / "darknet"
    cmd = "git clone -b concurrent --depth 1 https://github.com/Mandroide/NVIDIA_AICITY.git"
    subprocess.run(cmd.split())
    cmd = f"pip install -r '{os.fspath(REPO_PATH / 'requirements.txt')}'"
    subprocess.run(cmd, shell=True)
    cmd = f"python '{os.fspath(REPO_PATH / 'install_decord.py')}'"
    decord = subprocess.Popen(cmd, shell=True)
    cwd = pathlib.Path(__file__).parent
    cmd = f"pip install -r '{os.fspath(cwd / 'requirements.txt')}'"
    popen = subprocess.Popen(cmd, shell=True)
    cmd = "change.npy bounds1.npy bounds2.npy centers1.npy centers2.npy result1.npy result2.npy"
    cmd = collections.deque([f"'{os.fspath(REPO_PATH / c)}'" for c in cmd.split()])
    cmd.appendleft("rm")
    subprocess.run(cmd)
    # Install Darknet
    cmd = "git clone --depth 1 https://github.com/AlexeyAB/darknet.git"
    darknet_run = subprocess.Popen(cmd.split())
    darknet_run.wait()
    cmd = (f"wget -P {os.fspath(DARKNET_PATH)}"
           f" https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights")
    wget = subprocess.Popen(cmd.split())
    cmd = (f"sed -in 's/OPENCV=0/OPENCV=1/' '{os.fspath(DARKNET_PATH / 'Makefile')}'"
           f" && sed -in 's/GPU=0/GPU=1/' '{os.fspath(DARKNET_PATH / 'Makefile')}'"
           f" && sed -in 's/CUDNN=0/CUDNN=1/' '{os.fspath(DARKNET_PATH / 'Makefile')}'"
           f" && sed -in 's/CUDNN_HALF=0/CUDNN_HALF=1/' '{os.fspath(DARKNET_PATH / 'Makefile')}'"
           f" && make -C '{os.fspath(DARKNET_PATH)}'")
    make = subprocess.Popen(cmd, shell=True)
    popen.wait()
    logger = logging.getLogger(__name__)
    fmt = logging.Formatter("%(levelname)s - %(module)s - %(funcName)s - %(message)s")
    hdlr = logging.StreamHandler()
    hdlr.setLevel(logging.DEBUG)
    hdlr.setFormatter(fmt)
    logger.addHandler(hdlr)
    logger.setLevel(logging.DEBUG)
    decord.wait()
    import extract_frames

    dfs = pd.read_excel(os.fspath(REPO_PATH / "DOT Iowa Accident Labels.ods"), sheet_name=None)
    extract_frames.drop_columns(dfs)
    video_extensions = [ext for ext in frozenset.union(*frozenset(
        map(lambda df: frozenset(
            map(lambda tupl: pathlib.PurePath(tupl[1]["Video Name"]).suffix[1:],
                df.iterrows())), dfs.values())))]
    video_extensions.remove("")
    video_extensions = " ".join(video_extensions)
    # Background Modelling
    cmd = (
        f"python '{os.fspath(REPO_PATH / 'extract_frames.py')}' --freq 100 --root '{os.fspath(args.root)}'"
        f" --ext '{video_extensions}'")
    popen = subprocess.Popen(cmd, shell=True)
    popen.wait()
    cmd = f"python '{os.fspath(REPO_PATH / 'extract_processed.py')}'"
    popen = subprocess.Popen(cmd, shell=True)
    # Run darknet on the processed images.
    wget.wait()
    make.wait()
    popen.wait()
    exe = DARKNET_PATH / "darknet"
    exe.chmod(stat.S_IRUSR | stat.S_IEXEC)
    os.chdir(DARKNET_PATH)
    cmd = (
        f"./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -ext_output -dont_show"
        f" -out '{os.fspath(REPO_PATH / 'result.json')}' < '{os.fspath(REPO_PATH / 'processed_images2.txt')}'")
    darknet_result = subprocess.Popen(cmd, shell=True)

    # Segmentation Maps
    cmd = (
        f"python '{os.fspath(REPO_PATH / 'extract_frames.py')}' --freq 10 --root '{os.fspath(args.root)}'"
        f" --ext '{video_extensions}'")
    popen = subprocess.Popen(cmd, shell=True)
    popen.wait()
    cmd = (
        f"./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -ext_output -dont_show"
        f" -out '{os.fspath(REPO_PATH / 'Masks/part1.json')}' < '{os.fspath(REPO_PATH / 'ori_images.txt')}'")
    popen = subprocess.Popen(cmd, shell=True)
    popen.wait()
    cmd = f"python '{os.fspath(REPO_PATH / 'Seg_masks.py')}'"
    popen = subprocess.Popen(cmd, shell=True)
    popen.wait()
    cmd = f"python '{os.fspath(REPO_PATH / 'Masks/get_ignore_area.py')}'"
    popen = subprocess.Popen(cmd, shell=True)
    darknet_result.wait()
    popen.wait()
    cmd = f"python '{os.fspath(REPO_PATH / 'Detector.py')}'"
    popen = subprocess.Popen(cmd, shell=True)
    popen.wait()

    # ----------------------------------------------------------------------------------------------------------------------
    nrmse = {}
    y_true_all = []
    y_pred_all = []
    detected_anomalies = {}
    was_detected = []

    with open(REPO_PATH / "dataset.json") as f:
        dataset = json.load(f)
        df_result = pd.read_csv(REPO_PATH / "Result.csv")
        dataset = iter(map(lambda p: pathlib.Path(p), dataset))

        # Iterate every dataset
        for idx, data in enumerate(dataset, start=1):
            # Iterate every dataframe year
            for df in dfs.values():
                row_true = df.loc[df["Video Name"] == data.name]
                if not row_true.empty:
                    # Find prediction of video
                    row_pred = df_result.loc[df_result["video_id"] == idx]
                    detected_anomalies[idx] = len(row_pred.index)
                    y_pred = row_pred.iloc[:, 1].to_numpy()
                    y_true = row_true.iloc[:, 2].to_numpy()
                    # Check if an anomaly wasn't detected.
                    was_detected.append(not row_pred.empty)
                    if was_detected[-1]:
                        y_true_all.append(y_true)
                        y_pred_all.append(y_pred)
                        nrmse[idx] = perf.measure_detection_time_error(y_true, y_pred)
                        print("NRMSE[", idx, "]", nrmse[idx])

    was_detected = np.array(was_detected)
    print()
    print("Number of detected anomalies: ", detected_anomalies)
    print("NRMSE: ", nrmse)
    print("Length of NRMSE: ", len(nrmse))
    f1 = f1_score(np.ones(was_detected.shape), was_detected)
    print("F1 score: ", f1)
    s4 = f1 * (1 - np.array(list(nrmse.values())))
    print("S4: ", s4)
