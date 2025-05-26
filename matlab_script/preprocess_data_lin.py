import os
import glob
import re
import numpy as np
from PIL import Image

def mkdirfunc(path):
    os.makedirs(path, exist_ok=True)

def _numeric_key(fp):
    """
    Sort key: extract the first integer in the filename for numeric ordering.
    E.g. raw__187.jpg → 187, raw__1703.jpg → 1703
    """
    name = os.path.basename(fp)
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else -1

def script_index_gen(data_root_path, clip_len=16, skip_step=1, overlap_rate=0):
    """
    Generate .npz files containing frame-index arrays for each video clip.
    Adapted to processed/data_lin_256/{Train,Test}/{traj_id}.
    """
    base_processed = os.path.join('/home/haoj/memae/datasets/processed', 'data_lin_256')
    frame_ext = 'jpg'

    clip_rng = clip_len * skip_step - 1
    overlap_shift = clip_len * skip_step - 1
    step = (clip_rng + 1) - overlap_shift

    for subset in ['Train', 'Test']:
        print(f"Processing {subset} indices...")
        sub_in  = os.path.join(base_processed, subset)
        sub_out = os.path.join(base_processed, f"{subset}_idx")
        mkdirfunc(sub_out)

        video_dirs = [
            d for d in os.listdir(sub_in)
            if os.path.isdir(os.path.join(sub_in, d))
        ]

        for v_name in sorted(video_dirs):
            print(f"  Video: {v_name}")
            frames = sorted(
                glob.glob(os.path.join(sub_in, v_name, f'*.{frame_ext}')),
                key=_numeric_key
            )
            n = len(frames)
            if n < clip_len:
                print(f"    ! only {n} frames, skipping")
                continue
            starts = range(1, n - clip_rng + 1, step)

            for j, start in enumerate(starts, 1):
                idx = np.arange(start, start + clip_rng + 1, skip_step, dtype=int)
                out_dir = os.path.join(sub_out, v_name)
                mkdirfunc(out_dir)
                out_fp = os.path.join(out_dir, f"{v_name}_i{j:03d}.npz")
                np.savez(out_fp, v_name=v_name, idx=idx)
                print(f"    › saved {out_fp}")

def trans_img2img(inpath, outpath, opts):
    """
    Read raw .jpg frames, convert to gray/RGB, resize to opts['outsize'], save as .jpg.
    """
    mkdirfunc(outpath)
    pattern = f"*.{opts['img_type']}"
    imgs = sorted(glob.glob(os.path.join(inpath, pattern)), key=_numeric_key)
    if not imgs:
        print(f"  ! no {pattern} in {inpath}")
        return

    for i, fp in enumerate(imgs, 1):
        img = Image.open(fp)
        img = img.convert('L') if opts.get('is_gray', False) else img.convert('RGB')
        if opts.get('outsize'):
            img = img.resize(opts['outsize'], Image.BICUBIC)
        img.save(os.path.join(outpath, f"{i:03d}.jpg"))

def script_img_prep(data_root_path):
    """
    Master prep for data_lin:
      - trajectory "1" → Test
      - all other trajectories → Train
      - convert & resize frames into processed/data_lin_256/{Train,Test}/{traj_id}
    """
    raw_base  = os.path.join(data_root_path, 'data_lin')
    proc_base = os.path.join("/home/haoj/memae/datasets", 'processed', 'data_lin_256')
    mkdirfunc(proc_base)

    # discover numeric subdirectories only
    all_items = os.listdir(raw_base)
    traj_dirs = sorted([d for d in all_items
                        if os.path.isdir(os.path.join(raw_base, d)) and d.isdigit()])

    test_traj  = ['1']
    train_traj = [d for d in traj_dirs if d not in test_traj]

    opts = {
        'is_gray': False,
        'outsize': (256, 256),
        'img_type': 'jpg'
    }

    # process Train
    for v_name in train_traj:
        src = os.path.join(raw_base, v_name)
        dst = os.path.join(proc_base, 'Train', v_name)
        print(f"[Train] {v_name}: {src} → {dst}")
        trans_img2img(src, dst, opts)

    # process Test
    for v_name in test_traj:
        src = os.path.join(raw_base, v_name)
        dst = os.path.join(proc_base, 'Test', v_name)
        print(f"[Test]  {v_name}: {src} → {dst}")
        trans_img2img(src, dst, opts)

if __name__ == '__main__':
    # point to the folder containing "data_lin"
    data_root = '/home/haoj/0/data'
    # 1) convert & resize all frames
    script_img_prep(data_root)
    # 2) generate sliding-window index files with a temporal stride of 3
    script_index_gen(data_root, skip_step=3)
