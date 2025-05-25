import os
import glob
import numpy as np
from PIL import Image
from scipy.io import savemat


def mkdirfunc(path):
    os.makedirs(path, exist_ok=True)


def script_index_gen(data_root_path, clip_len=16, skip_step=1, overlap_rate=0):
    """
    Generate .mat files containing frame index lists for each video clip.
    Mirrors MATLAB script_index_gen.m logic fileciteturn1file3.
    """
    in_path = os.path.join(data_root_path, 'datasets', 'processed', 'UCSD_P2_256')
    frame_file_type = 'jpg'
    clip_rng = clip_len * skip_step - 1
    overlap_shift = clip_len - 1
    # step between clip starts
    step = clip_rng + 1 - overlap_shift
    sub_dir_list = ['Train', 'Test']

    for sub_dir in sub_dir_list:
        print(f"Processing {sub_dir} indices...")
        sub_in_path = os.path.join(in_path, sub_dir)
        idx_out_path = os.path.join(in_path, f"{sub_dir}_idx")
        mkdirfunc(idx_out_path)

        video_dirs = [d for d in os.listdir(sub_in_path)
                      if os.path.isdir(os.path.join(sub_in_path, d))]
        for v_name in video_dirs:
            print(f"  Video: {v_name}")
            frame_list = sorted(glob.glob(os.path.join(sub_in_path, v_name, f'*.{frame_file_type}')))
            frame_num = len(frame_list)
            starts = list(range(1, frame_num - clip_rng + 1, step))

            for j, start in enumerate(starts, 1):
                idx = list(range(start, start + clip_rng + 1, skip_step))
                out_dir = os.path.join(idx_out_path, v_name)
                mkdirfunc(out_dir)
                out_file = os.path.join(out_dir, f"{v_name}_i{j:03d}.mat")
                savemat(out_file, {'v_name': v_name, 'idx': np.array(idx)})


def trans_img2img(inpath, outpath, opts):
    """
    Read frames, convert to gray/RGB, resize, and save as .jpg.
    Mirrors MATLAB trans_img2img.m logic fileciteturn1file2.
    """
    mkdirfunc(outpath)
    pattern = f"*.{opts['img_type']}"
    img_files = sorted(glob.glob(os.path.join(inpath, pattern)))
    for i, fp in enumerate(img_files, 1):
        img = Image.open(fp)
        # grayscale or RGB
        if opts.get('is_gray', False):
            img = img.convert('L')
        else:
            img = img.convert('RGB')
        # resize if requested
        if opts.get('outsize'):
            img = img.resize(opts['outsize'], Image.BICUBIC)
        out_fp = os.path.join(outpath, f"{i:03d}.jpg")
        img.save(out_fp)


def trans_img2label(inpath, idx, outpath):
    """
    Read ground-truth masks, threshold to binary labels, save vector as .mat.
    Mirrors MATLAB trans_img2label.m logic fileciteturn1file1.
    """
    gt_dir = os.path.join(inpath, f"Test{idx:03d}_gt")
    print(f"Processing GT: {gt_dir}")
    mkdirfunc(outpath)
    bmp_files = sorted(glob.glob(os.path.join(gt_dir, '*.bmp')))
    labels = []
    for fp in bmp_files:
        img = Image.open(fp).convert('L')
        arr = np.array(img, dtype=np.float64) / 255.0
        labels.append(0 if arr.sum() < 1 else 1)
    out_fp = os.path.join(outpath, f"Test{idx:03d}.mat")
    savemat(out_fp, {'l': np.array(labels)})


def script_img_prep(data_root_path):
    """
    Master preprocessing: resize frames and extract GT labels.
    Mirrors script_img_prep.m orchestration fileciteturn1file0.
    """
    in_path = os.path.join(data_root_path, 'datasets', 'UCSDped2')
    out_path = os.path.join(data_root_path, 'datasets', 'processed', 'UCSD_P2_256')
    mkdirfunc(out_path)

    sub_dirs = ['Train', 'Test']
    file_nums = {'Train': 16, 'Test': 12}
    opts = {
        'is_gray': True,
        'maxs': 320,
        'outsize': (256, 256),
        'img_type': 'tif'
    }

    for subdir in sub_dirs:
        sub_in = os.path.join(in_path, subdir)
        sub_out = os.path.join(out_path, subdir)
        for i in range(1, file_nums[subdir] + 1):
            v_name = f"{subdir}{i:03d}"
            v_in = os.path.join(sub_in, v_name)
            v_out = os.path.join(sub_out, v_name)
            mkdirfunc(v_out)
            print(f"Frame prep: {v_in} -> {v_out}")
            trans_img2img(v_in, v_out, opts)

    # generate GT labels for Test
    gt_in_base = os.path.join(in_path, 'Test')
    gt_out_base = os.path.join(out_path, 'Test_gt')
    mkdirfunc(gt_out_base)
    for i in range(1, file_nums['Test'] + 1):
        trans_img2label(gt_in_base, i, gt_out_base)


if __name__ == '__main__':
    # set your root path here
    data_root = '/home/haoj/memae-anomaly-detection/'
    script_img_prep(data_root)
    script_index_gen(data_root)
