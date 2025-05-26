import os
import glob
import numpy as np
from PIL import Image

def mkdirfunc(path):
    """
    Ensure that the directory at 'path' exists.
    If it does not, create it (including any missing parent directories).
    """
    os.makedirs(path, exist_ok=True)
    # exist_ok=True avoids error if the directory already exists.

def script_index_gen(data_root_path, clip_len=16, skip_step=1, overlap_rate=0):
    """
    Generate NumPy-based index files (.npz) for sliding-window video clips.
    Each .npz contains:
      - 'v_name': the video directory name
      - 'idx'   : a 1D integer array of frame indices within that clip

    Parameters:
    -----------
    data_root_path : str
        Root of your data folder, containing 'datasets/processed/UCSD_P2_256'.
    clip_len       : int
        Number of frames per extracted clip.
    skip_step      : int
        Temporal sampling interval (e.g., skip_step=2 picks every other frame).
    overlap_rate   : float
        Fractional overlap between successive clips (currently unused; for future extension).
    """
    # Base directory where processed 256×256 frames live
    base_processed = os.path.join(data_root_path, 'datasets', 'processed', 'UCSD_P2_256')
    frame_ext = 'jpg'  # We expect processed frames in JPEG format

    # Compute the span of indices covered by one clip:
    # clip_rng = last_index - first_index within a clip
    clip_rng = clip_len * skip_step - 1
    # If you wanted an overlap of N frames, you'd shift start by (clip_len - N)
    overlap_shift = clip_len - 1
    # Effective step between clip-starts: one full clip minus the overlap
    step = (clip_rng + 1) - overlap_shift

    # Process both training and testing sets
    for subset in ['Train', 'Test']:
        print(f"=== Generating indices for {subset} set ===")
        input_dir  = os.path.join(base_processed, subset)
        output_dir = os.path.join(base_processed, f"{subset}_idx")
        mkdirfunc(output_dir)

        # Discover all video subdirectories (e.g. Train001, Train002, …)
        video_dirs = [
            d for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d))
        ]

        for v_name in sorted(video_dirs):
            print(f"  • Video: {v_name}")
            # Gather all frame filepaths and sort them lexicographically
            frame_paths = sorted(glob.glob(
                os.path.join(input_dir, v_name, f'*.{frame_ext}')
            ))
            num_frames = len(frame_paths)
            if num_frames == 0:
                print(f"    ! Warning: no frames found in {v_name}")
                continue

            # Determine valid 1-based start indices
            # e.g. range(1, num_frames - clip_rng + 1, step)
            starts = list(range(1, num_frames - clip_rng + 1, step))

            # For each possible clip start, build its index array
            for clip_idx, start in enumerate(starts, start=1):
                # Build a NumPy arange: start, start+skip_step, …, up to start+clip_rng
                idx_array = np.arange(
                    start,
                    start + clip_rng + 1,
                    skip_step,
                    dtype=int
                )
                # Create per-video subfolder in output if needed
                vid_out_dir = os.path.join(output_dir, v_name)
                mkdirfunc(vid_out_dir)

                # Save as a compressed .npz archive:
                #   - 'v_name' lets downstream code know which video
                #   - 'idx'    is the frame-index vector
                filename = f"{v_name}_i{clip_idx:03d}.npz"
                out_path = os.path.join(vid_out_dir, filename)
                np.savez(out_path, v_name=v_name, idx=idx_array)
                print(f"    › Saved clip #{clip_idx} indices to {filename}")

def trans_img2img(inpath, outpath, opts):
    """
    Convert raw frames to JPEGs, with optional grayscale + resizing.

    Parameters:
    -----------
    inpath : str
        Directory containing raw source images (e.g. TIFF frames).
    outpath: str
        Directory to save processed JPEG frames.
    opts : dict
        Must contain:
          - 'img_type': file extension of source images (e.g. 'tif')
          - 'is_gray' : bool → convert to grayscale if True, else RGB
          - 'outsize' : tuple (width, height) for bicubic resizing
    """
    mkdirfunc(outpath)
    pattern = f"*.{opts['img_type']}"
    src_files = sorted(glob.glob(os.path.join(inpath, pattern)))

    if not src_files:
        print(f"  ! No files matching {pattern} in {inpath}")
        return

    for idx, filepath in enumerate(src_files, start=1):
        img = Image.open(filepath)

        # Color‐space conversion
        if opts.get('is_gray', False):
            img = img.convert('L')   # single-channel grayscale
        else:
            img = img.convert('RGB') # three-channel color

        # Resize if target size provided
        if opts.get('outsize'):
            img = img.resize(opts['outsize'], Image.BICUBIC)

        # Save with zero-padded numbering: 001.jpg, 002.jpg, …
        out_name = f"{idx:03d}.jpg"
        img.save(os.path.join(outpath, out_name))

def trans_img2label(inpath, idx, outpath):
    """
    Read bitmap masks for Test video #idx, threshold to binary labels,
    and save the 1D label array as a .npy file.

    Parameters:
    -----------
    inpath : str
        Base directory under which mask folders like 'Test001_gt' live.
    idx    : int
        Index of the Test video (1-based), matches folder name Test{idx:03d}_gt.
    outpath: str
        Directory to save the resulting .npy label vectors.
    """
    gt_folder = os.path.join(inpath, f"Test{idx:03d}_gt")
    print(f"--- Processing ground-truth in {gt_folder}")
    mkdirfunc(outpath)

    mask_files = sorted(glob.glob(os.path.join(gt_folder, '*.bmp')))
    num_masks = len(mask_files)
    labels = np.zeros(num_masks, dtype=np.uint8)

    for i, mask_fp in enumerate(mask_files):
        # Load as grayscale and normalize to [0,1]
        arr = np.array(Image.open(mask_fp).convert('L'), dtype=np.float32) / 255.0
        # If any pixel > 0, assign label 1 (foreground present)
        labels[i] = 1 if arr.sum() > 0 else 0

    # Save the entire label sequence for this video:
    out_file = os.path.join(outpath, f"Test{idx:03d}.npy")
    np.save(out_file, labels)
    print(f"  › Saved labels array of length {num_masks} to {out_file}")

def script_img_prep(data_root_path):
    """
    High‐level orchestration of image preprocessing and label extraction.

    Steps:
      1) For each Train/Test video in UCSDped2:
         - Convert + grayscale + resize frames → JPEGs in UCSD_P2_256/{Train,Test}/
      2) For each Test video:
         - Read .bmp masks → binary labels → .npy in UCSD_P2_256/Test_gt/

    Parameters:
    -----------
    data_root_path : str
        Root directory containing 'datasets/UCSDped2' and where
        'datasets/processed/UCSD_P2_256' will be created.
    """
    raw_base   = os.path.join(data_root_path, 'datasets', 'UCSDped2')
    proc_base  = os.path.join(data_root_path, 'datasets', 'processed', 'UCSD_P2_256')
    mkdirfunc(proc_base)

    counts = {'Train': 16, 'Test': 12}
    opts = {
        'is_gray': True,         # convert to single-channel
        'outsize': (256, 256),   # resize dimensions
        'img_type': 'tif'        # raw frame extension
    }

    # Process raw frames
    for subset, n_videos in counts.items():
        for i in range(1, n_videos + 1):
            vid_name = f"{subset}{i:03d}"
            src_dir  = os.path.join(raw_base, subset, vid_name)
            dst_dir  = os.path.join(proc_base, subset, vid_name)
            mkdirfunc(dst_dir)
            print(f"[Frame prep] {src_dir} → {dst_dir}")
            trans_img2img(src_dir, dst_dir, opts)

    # Process ground-truth masks into binary labels
    gt_src_base = os.path.join(raw_base, 'Test')
    gt_dst_base = os.path.join(proc_base, 'Test_gt')
    mkdirfunc(gt_dst_base)
    for i in range(1, counts['Test'] + 1):
        trans_img2label(gt_src_base, i, gt_dst_base)

if __name__ == '__main__':
    # === Entry Point ===
    # Modify this to point at your local dataset root:
    data_root = '/home/haoj/memae/'

    # 1) Prepare frames (+resize, grayscale)
    script_img_prep(data_root)

    # 2) Generate sliding‐window index files
    script_index_gen(data_root)
