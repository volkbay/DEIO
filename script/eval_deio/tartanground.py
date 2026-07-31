"""Run DEIO on devogam's TartanGround split, reading our reps convention directly.

Added for the devogam baseline table: DEIO is the tightly-coupled event-inertial
competitor to the scale-only study in ``doc/imu_scale/``. It shares our
front-end (it runs on ``DEVO.pth``), so the comparison isolates *how* the IMU is
used rather than which network sees the events.

No data conversion: this reads the sequence layout devogam already has —

    <seq>/reps_voxel_grid_frame_lcam_front/reps.h5   (M, 5, H, W) raw voxels
    <seq>/tss_imgs_sec.txt                           frame timestamps
    <seq>/pose_lcam_front.txt                        GT, TartanAir NED quats
    <seq>/imu/{acc,gyro,imu_time}.npy                100 Hz IMU

Two conversions that are *not* optional (see ``src/util/imu.py``):

1. TartanGround stores ``acc = R_bw(a + g)``, i.e. acceleration *including*
   gravity with z up. A real accelerometer — which is what DEIO's
   preintegration expects — measures specific force ``f = R_bw(a - g)``.
   Feeding ``acc`` straight in flips gravity and the VI alignment diverges.
2. Poses are reordered NED -> camera with ``[1,2,0,4,5,3,6]``, exactly as
   ``src/util/eval.py`` does, so the GT matches DEVO's optical-frame output.
   The IMU stays in the native body frame; ``Ti1c`` in the config carries the
   relative rotation.

Usage (from dep/DEIO, DEIO env active):

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python script/eval_deio/tartanground.py \\
        --inputdir=/data/hdda/tartanground \\
        --config=config/tartanground.yaml \\
        --val_split=../../cfg/splits/tartanground/tg_val.txt \\
        --network=../../bin/DEVO.pth --enable_event --trials=1 --save_trajectory
"""

import argparse
import math
import os
import sys

import h5py
import numpy as np
import quaternion
import torch

import evo
from evo.tools.settings import SETTINGS
SETTINGS['plot_backend'] = 'Agg'

from devo.config import cfg
from utils.eval_utils import compute_median_results, log_results, run_DEIO2

# NED -> camera(optical) reorder, identical to src/util/eval.py's poseSlc
POSE_SLC = [1, 2, 0, 4, 5, 3, 6]
G_NORM = 9.81
DEFAULT_INTRINSICS = (320.0, 320.0, 320.0, 320.0)   # 640x640, 90 deg FOV


def read_split(path):
    """devogam eval-split lines: ``<seqpath>[:cam][,start,stop]``."""
    out = []
    for line in open(path).read().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line.split(",")[0].split(":")[0].strip())
    return out


def load_gt(seq):
    """GT in the optical camera frame; returns (tss_us, N x 7 xyz+quat xyzw)."""
    poses = np.loadtxt(os.path.join(seq, "pose_lcam_front.txt"))[:, POSE_SLC]
    tss = np.loadtxt(os.path.join(seq, "tss_imgs_sec.txt")) * 1e6
    n = min(len(poses), len(tss))
    return tss[:n], poses[:n]


def load_imu(seq):
    """DEIO's IMU array: [t_us, gx, gy, gz, ax, ay, az], rad/s + specific force.

    ``eval_deio`` multiplies columns 1:4 by 180/pi and ``dba.py`` divides by the
    same, so the net expectation is rad/s. Acceleration must be specific force;
    TartanGround's ``acc`` is not (see module docstring).
    """
    d = os.path.join(seq, "imu")
    t = np.load(os.path.join(d, "imu_time.npy"))
    acc = np.load(os.path.join(d, "acc.npy"))
    gyro = np.load(os.path.join(d, "gyro.npy"))
    ori = np.load(os.path.join(d, "ori_global.npy"))

    from scipy.spatial.transform import Rotation
    R_wb = Rotation.from_euler("XYZ", ori)
    g_vec = np.array([0.0, 0.0, -G_NORM])
    a_world = R_wb.apply(acc) - g_vec              # true world acceleration
    f_body = R_wb.inv().apply(a_world - g_vec)     # specific force, body frame

    out = np.zeros((len(t), 7))
    out[:, 0] = t * 1e6                            # seconds -> us
    out[:, 1:4] = gyro
    out[:, 4:7] = f_body

    # The IMU stream ends at 129.29 s while the last camera frame is stamped
    # 129.30 s, so DEIO's preintegration walks off the end of the array
    # (IndexError in dba.py). Hold the last measurement for a short tail so
    # every frame has IMU on both sides; 10 samples = 0.1 s of held value,
    # which is one keyframe interval and cannot affect the trajectory.
    dt = float(np.mean(np.diff(out[:, 0])))
    tail = np.repeat(out[-1:], 10, axis=0)
    tail[:, 0] = out[-1, 0] + dt * np.arange(1, 11)
    return np.vstack([out, tail])


def reps_iterator(seq, cam="lcam", stride=1, intrinsics=DEFAULT_INTRINSICS):
    """Yield ``(voxel, intrinsics, ts_us)`` from our offline reps.h5.

    Raw voxels, exactly as ``src/script/eval.py`` feeds them; the network's own
    NORM ('std' for voxel_grid) does the normalisation.
    """
    reps_dir = None
    for name in (f"reps_voxel_grid_frame_{cam}_front",
                 f"reps_voxel_grid_0_{cam}_front"):
        if os.path.isdir(os.path.join(seq, name)):
            reps_dir = os.path.join(seq, name)
            break
    if reps_dir is None:
        raise FileNotFoundError(f"no voxel_grid reps under {seq}")

    tss = np.loadtxt(os.path.join(seq, "tss_imgs_sec.txt")) * 1e6
    K = torch.tensor(list(intrinsics), dtype=torch.float32)
    with h5py.File(os.path.join(reps_dir, "reps.h5"), "r") as f:
        reps = f["reps"]
        n = min(len(reps), len(tss) - 1)
        print(f"Loaded {n} voxels from {reps_dir}")
        for i in range(0, n, stride):
            v = torch.from_numpy(np.asarray(reps[i], dtype=np.float32))
            # reps are stamped at the END of their window, as tss[i+1]
            yield v.cuda(), K.cuda(), float(tss[i + 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputdir", default="/data/hdda/tartanground")
    parser.add_argument("--network", type=str, required=True)
    parser.add_argument("--val_split", type=str, required=True)
    parser.add_argument("--config", default="config/tartanground.yaml")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--enable_event", action="store_true")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--opts", nargs="+", default=[])
    parser.add_argument("--save_trajectory", action="store_true")
    parser.add_argument("--side", type=str, default="left")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--rawdir", default="/devogam/log/results/deio_raw",
                        help="where to dump unaligned TUM trajectories for "
                             "scoring with devogam's own metric code")
    parser.add_argument("--resnet", action="store_true")
    parser.add_argument("--block_dims", type=str, default="64,128,256")
    parser.add_argument("--initial_dim", type=int, default=64)
    parser.add_argument("--pretrain", type=str, default="resnet18")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.resnet = args.resnet
    cfg.block_dims = list(map(int, args.block_dims.split(",")))
    cfg.initial_dim = args.initial_dim
    cfg.pretrain = args.pretrain
    print(cfg, "\n")

    assert not cfg.CLASSIC_LOOP_CLOSURE
    assert args.enable_event, "TartanGround DEIO runs on events only"
    assert cfg.ENALBE_IMU, "this script is the DEIO (inertial) baseline"

    scenes = read_split(args.val_split)
    print(f"{len(scenes)} scenes: {scenes}")

    cam = "rcam" if args.side in ("right", "rcam") else "lcam"
    results_dict_scene, figures, all_results = {}, {}, []
    outfolder = None

    for scene in scenes:
        seq = os.path.join(args.inputdir, scene)
        # log_results keys this dict by the name it is handed, which is the
        # slash-free one (paths are built from it), so register that key.
        flat = scene.replace("/", "_")
        results_dict_scene[flat] = []
        tss_traj_us, traj_hf = load_gt(seq)

        # DEIO uses all_gt for visualisation only -- VisualIMUAlignment solves
        # gravity from IMU + VO alone -- but it must still be populated.
        all_gt = {}
        for ts, d in zip(tss_traj_us, traj_hf):
            q = quaternion.from_float_array([d[6], d[3], d[4], d[5]])
            T = np.eye(4)
            T[:3, :3] = quaternion.as_rotation_matrix(q)
            T[:3, 3] = d[:3]
            all_gt[float(ts / 1e6)] = {"T": T}
        all_gt_keys = sorted(all_gt.keys())

        all_imu = load_imu(seq)
        all_imu[:, 1:4] *= 180 / math.pi        # undone inside devo/dba.py
        all_imu = all_imu[all_imu[:, 0].argsort()]

        for trial in range(args.trials):
            print(f"\n=== {scene} trial {trial}")
            traj_est, tstamps, flowdata, avg_fps = run_DEIO2(
                seq, cfg, args.network, viz=args.viz,
                iterator=reps_iterator(seq, cam=cam, stride=args.stride),
                _all_imu=all_imu.copy(), _all_gt=all_gt,
                _all_gt_keys=all_gt_keys,
                timing=args.timing, H=640, W=640, viz_flow=False)

            # DEIO's log_results scores with correct_scale=True even though the
            # method is metric, so its numbers are NOT comparable to an
            # uncorrected evaluation. Dump the raw estimate first -- before
            # anything that can fail -- so it can be scored both ways by
            # src/script/score_traj.py, the same code that scores every row.
            os.makedirs(args.rawdir, exist_ok=True)
            ts = np.asarray(tstamps, dtype=np.float64)
            if ts.max() > 1e4:                     # us -> s
                ts = ts / 1e6
            np.savetxt(os.path.join(args.rawdir, f"{flat}_trial{trial:02d}.txt"),
                       np.column_stack([ts, traj_est]))

            data = (traj_hf, tss_traj_us, traj_est, tstamps)
            hyperparam = (None, args.network, "tartanground/DEIO_IMU",
                          flat, trial, cfg, args)
            all_results, results_dict_scene, figures, outfolder = log_results(
                data, hyperparam, all_results, results_dict_scene, figures,
                plot=args.plot, save=args.save_trajectory, return_figure=False,
                stride=args.stride, expname=flat,
                _n_to_align=1000, avg_fps=avg_fps)

        print(scene, sorted(results_dict_scene[flat]))

    results_dict = compute_median_results(results_dict_scene, all_results,
                                          "tartanground/DEIO_IMU",
                                          outfolder=outfolder)
    for k in results_dict:
        print(k, results_dict[k])
    print("Done!")


if __name__ == "__main__":
    main()
