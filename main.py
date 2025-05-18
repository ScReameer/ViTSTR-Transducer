import argparse
import os
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='path to config file')
parser.add_argument('--output-dir', type=str, required=True, help='output path directory')
parser.add_argument('--device', type=int, default=0, help='CUDA device id')
args = parser.parse_args()

if __name__ == '__main__':
    os.system(f'python train.py --config {args.config} --device {args.device} --output-dir {args.output_dir}')
    os.system(f'python test.py --config {args.config} --device {args.device} --weights {args.output_dir}/train/weights/ViTSTR-FP32.ckpt')