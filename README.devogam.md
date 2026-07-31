# DEIO (devogam fork)

Fork of [arclab-hku/DEIO](https://github.com/arclab-hku/DEIO) used by *devogam*
as the **tightly-coupled event-inertial baseline** for the scale-only study in
`doc/imu_scale/`. Added as a submodule at `dep/DEIO`.

DEIO runs on **`DEVO.pth`** — it is stock DEVO's front-end plus an IMU factor
graph — so a DEIO-vs-us comparison isolates *how the IMU is used*, not which
network sees the events.

## Changes vs upstream

- `script/eval_deio/tartanground.py` — evaluate on devogam's TartanGround split,
  reading our layout directly with no data conversion:
  `reps_voxel_grid_frame_lcam_front/reps.h5`, `tss_imgs_sec.txt`,
  `pose_lcam_front.txt`, `imu/*.npy`. Also dumps the **unaligned** trajectory so
  it can be scored by `src/script/score_traj.py` (see the warning below).
- `config/tartanground.yaml` — devogam's 96-patch operating point + the IMU
  extrinsic and noise parameters.

## Run it

```bash
conda activate DEIO
cd dep/DEIO
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python script/eval_deio/tartanground.py \
    --inputdir=/data/hdda/tartanground \
    --config=config/tartanground.yaml \
    --val_split=../../cfg/splits/tartanground/tg_val.txt \
    --network=../../bin/DEVO.pth \
    --enable_event --trials=5 --save_trajectory \
    --rawdir=../../log/results/deio_raw5
# then, from the repo root, score with devogam's own metric code:
python src/script/score_traj.py --dir log/results/deio_raw5 --trim
```

## ⚠️ DEIO's own evaluator fits the scale

`utils/eval_utils.py` calls `main_ape.ape(..., correct_scale=True)`, so its
reported MPE — and the paper's — is **Sim(3)-aligned even though DEIO is a
metric-scale method**. Its metric output is worth much less than its published
numbers suggest (on TartanGround: 0.216 % median Sim(3) vs 0.696 % median
SE(3)). Always re-score the raw dumps.

## Integration facts (from reading the code, not the docs)

1. **IMU array is `[t, gx, gy, gz, ax, ay, az]` with gyro in rad/s.** The eval
   script's rad→deg multiply is undone inside `devo/dba.py`, so it nets out.
2. **Acceleration must be real specific force.** TartanGround stores
   `acc = R(a + g)`; feed that raw and gravity inverts and the VI alignment
   diverges. With the correct conversion gravity converges to
   `[-0.0002, 0.0037, 9.8197]` — that is the check that your frames are right.
3. **`Ti1c` is `Pᵀ`, not identity.** The IMU body frame *is* `lcam_front`, but
   DEVO works in the optical convention reached by the NED permutation.
4. **`all_gt` is used for visualisation only.** `VisualIMUAlignment` solves
   gravity from IMU + VO alone, so there is no ground-truth-frame dependency in
   the estimate.
5. **IMU needs a short tail.** TartanGround's IMU ends 0.01 s before the last
   frame, which walks `dba.py` off the end of the array; the adapter holds the
   last sample for 0.1 s.
6. **Scene names containing `/`** break `log_results`' output paths — register
   the flattened name in `results_dict_scene` too.

## Build (env `DEIO`, CUDA 11.8) — the fixes upstream's README omits

```bash
conda env create -f environment.yml && conda activate DEIO
pip install --no-build-isolation .          # setup.py imports torch
conda install -y -c nvidia cuda-nvcc=11.8   # system nvcc 11.3 mismatches torch
conda install -y -c nvidia cuda-cccl=11.8   # thrust headers
pip install numpy-quaternion==2022.4.3
conda install -y -c conda-forge libboost-devel=1.82   # conda's `boost` is headers-only
cd thirdparty/gtsam && mkdir -p build && cd build
cmake .. -DGTSAM_BUILD_PYTHON=1 -DGTSAM_PYTHON_VERSION=3.10.20 \
  -DCMAKE_BUILD_TYPE=Release -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_UNSTABLE=OFF \
  -DBOOST_ROOT=$CONDA_PREFIX -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
make -j8 python-install
```

**GTSAM must be built from `thirdparty/gtsam`.** It is a *fork* adding
`CombinedImuFactor.evaluateErrorCustom`; the PyPI wheel imports fine and only
fails at the first VIO update, which is a very misleading failure mode.
