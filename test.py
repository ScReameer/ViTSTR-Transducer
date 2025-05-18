import yaml
import argparse
import time
from multiprocessing import cpu_count

import torch
torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from clearml import Task

from src.utils.predictor import Predictor
from src.net.model import ViTSTRTransducer
from src.data_processing.dataset import PriceDataset, Collate
from src.data_processing.vocabulary import Vocabulary


# Args
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='path to config file')
parser.add_argument('--device', type=int, default=0, help='CUDA device id')
parser.add_argument('--weights', type=str, required=True, help='path to weights')
args = parser.parse_args()

CONFIG = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
DEVICE_IDX = args.device
WEIGHTS_PATH = args.weights    

# Project hyperparameters
NUM_WORKERS = cpu_count() - 1
DATASET_PATH = CONFIG['DATASET_PATH']
PROJECT_NAME = CONFIG['CML_PROJECT']
TASK_NAME = CONFIG['CML_TASK']
OUTPUT_PATH = WEIGHTS_PATH.split("/")[0] + '/test_predictions'
# Evaluation hyperparameters
TEST_FOLDER = CONFIG['ViTSTR-T']['EVAL']['FOLDER']
BATCH_SIZE = CONFIG['ViTSTR-T']['EVAL']['BATCH_SIZE']


def test():
    # ClearML initialization
    task: Task = Task.init(
        project_name=PROJECT_NAME,
        task_name=TASK_NAME,
        task_type=Task.TaskTypes.testing
    )
    task.connect_configuration(CONFIG)
    # Load trained model
    model = ViTSTRTransducer.load_from_checkpoint(WEIGHTS_PATH, map_location=f'cuda:{DEVICE_IDX}', training=False).eval()
    model.freeze()
    img_size = model.input_size
    
    # Test dataloader
    vocab = model.vocab
    dataset_test = PriceDataset(DATASET_PATH, vocab, sample='test', img_size=img_size)
    collate = Collate(pad_idx=vocab.digit2idx['<PAD>'])
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, collate_fn=collate, shuffle=False, num_workers=NUM_WORKERS)

    # Validate test dataloader
    trainer = Trainer(devices=[DEVICE_IDX], logger=False, enable_checkpointing=False)
    start_time = time.time()
    trainer.test(model=model, dataloaders=dataloader_test)
    end_time = time.time()
    avg_inference_time = (end_time - start_time) / len(dataloader_test)
    print(f'Inference time: {avg_inference_time:.4f}s per image')

    # Save predictions
    predictions_visualizer = Predictor(img_size=img_size, device=DEVICE_IDX, output_path=f'{OUTPUT_PATH}')
    predictions_visualizer.caption_dataloader(dataloader=dataloader_test, model=model)

    print(f'\nDone! Results saved in {OUTPUT_PATH}\n')
    
if __name__ == '__main__':
    test()