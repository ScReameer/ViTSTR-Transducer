import os
import yaml
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from multiprocessing import cpu_count

import torch
torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.attention import SDPBackend
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, OnExceptionCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from clearml import Task, OutputModel

from src.net.model import ViTSTRTransducer
from src.data import Collate, LmdbDataset, JsonDataset, Database, Vocabulary
from src.utils import Predictor, History

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='path to config file')
parser.add_argument('--device', type=int, default=0, help='CUDA device id')
parser.add_argument('--output-dir', type=str, required=True, help='output path directory')
args = parser.parse_args()

CONFIG = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
DEVICE_IDX = args.device
OUTPUT_DIR = Path(args.output_dir)
TRAIN_DIR = Path(OUTPUT_DIR) / 'train'
WEIGHTS_DIR = Path(TRAIN_DIR) / 'weights'
WEIGHTS_NAME = 'ViTSTR-FP32-base'
WEIGHTS_EXTENSION = '.ckpt'
WEIGHTS_FULL_PATH = WEIGHTS_DIR / (WEIGHTS_NAME+WEIGHTS_EXTENSION)
TS_BF16_WEIGHTS = WEIGHTS_DIR / f'ViTSTR-BF16-base.torchscript'
TS_FP16_WEIGHTS = WEIGHTS_DIR / f'ViTSTR-FP16-base.torchscript'
VAL_EXAMPLES_PATH = OUTPUT_DIR / 'val_examples'
TEST_EXAMPLES_PATH = OUTPUT_DIR / 'test_examples'

# Project hyperparameters
NUM_WORKERS = CONFIG['ViTSTR-T']['NUM_WORKERS'] or cpu_count() - 1
DATASET_PATH = Path(CONFIG['DATASET']['PATH'])
DATASET_TYPE = CONFIG['DATASET']['TYPE']
LABELS = CONFIG['LABELS']
CASE_SENSITIVE = bool(CONFIG['CASE_SENSITIVE'])
CML_ENABLED = bool(CONFIG['ClearML']['ENABLED'])
PROJECT_NAME = CONFIG['ClearML']['PROJECT']
TASK_NAME = CONFIG['ClearML']['TASK']

# Model hyperparameters
IMG_SIZE: tuple[int, int] = CONFIG['ViTSTR-T']['IMG_SIZE']
INPUT_CHANNELS = CONFIG['ViTSTR-T']['INPUT_CHANNELS']
BACKBONE_TYPE = CONFIG['ViTSTR-T']['BACKBONE_TYPE']
match BACKBONE_TYPE:
    case 'vitstr_tiny':
        D_MODEL = 192 # Number of features in the text/visual embedding
        NUM_HEADS = 3 # Number of heads in the TransformerDecoderLayer. D_MODEL must be divisible by NUM_HEADS without remainder
    case 'vitstr_small':
        D_MODEL = 384
        NUM_HEADS = 6
    case 'vitstr_base':
        D_MODEL = 768
        NUM_HEADS = 12
    case 'vit':
        D_MODEL = 192
        NUM_HEADS = 3

# Training hyperparameters
PRECISION = CONFIG['ViTSTR-T']['TRAIN']['PRECISION']
TRAIN_BATCH_SIZE = CONFIG['ViTSTR-T']['TRAIN']['BATCH_SIZE']
MAX_EPOCHS = CONFIG['ViTSTR-T']['TRAIN']['MAX_EPOCHS']
LR = float(CONFIG['ViTSTR-T']['TRAIN']['LR'])
LOSS = CONFIG['ViTSTR-T']['TRAIN']['LOSS']
WEIGHT_DECAY = float(CONFIG['ViTSTR-T']['TRAIN']['WEIGHT_DECAY'])
DROPOUT_RATE = CONFIG['ViTSTR-T']['TRAIN']['DROPOUT_RATE']
GAMMA = CONFIG['ViTSTR-T']['TRAIN']['GAMMA']
TRAIN_FOLDER: str = CONFIG['ViTSTR-T']['TRAIN']['FOLDERS'][0]
VAL_FOLDER: str = CONFIG['ViTSTR-T']['TRAIN']['FOLDERS'][1]
PATIENCE = int(CONFIG['ViTSTR-T']['TRAIN']['PATIENCE'])
MIN_DELTA = float(CONFIG['ViTSTR-T']['TRAIN']['MIN_DELTA'])
try:
    sdp_backend_literal = CONFIG['ViTSTR-T']['TRAIN']['SDPBackend']
    SDP_BACKEND = eval(f"SDPBackend.{sdp_backend_literal}")
except:
    raise ValueError(f"Wrong SDPBackend {sdp_backend_literal}, should be one of: {[k for k, _ in SDPBackend.__dict__.items() if not k.startswith('_') and k.isupper()]}")

# Test
TEST_FOLDER = CONFIG['ViTSTR-T']['TEST']['FOLDER']
TEST_BATCH_SIZE = CONFIG['ViTSTR-T']['TEST']['BATCH_SIZE']


class OnExceptionCheckpointWeightsOnly(OnExceptionCheckpoint):
    def __init__(self, dirpath, filename):
        super().__init__(dirpath, filename)
    
    def on_exception(self, trainer: Trainer, *_, **__) -> None:
        trainer.save_checkpoint(self.ckpt_path, weights_only=True)
    
    def teardown(self, trainer: Trainer, *_, **__) -> None:
        pass


def get_lmdb_paths(root_path: Path):
    paths = []
    for path in root_path.rglob('*'):
        if path.is_dir():
            children = list(path.iterdir())
            if not any(child.is_dir() for child in children):
                paths.append(path)
    return paths

def create_lmdb_dataset(paths, sample_folder, vocab):
    return ConcatDataset([
        LmdbDataset(
            db=Database(root=str(p), max_readers=NUM_WORKERS),
            vocab=vocab,
            sample=sample_folder,
            img_size=IMG_SIZE,
            input_channels=INPUT_CHANNELS
        ) for p in paths
    ])

def train():
    # ClearML initialization
    if CML_ENABLED:
        task: Task = Task.init(
            project_name=PROJECT_NAME,
            task_name=TASK_NAME,
            task_type=Task.TaskTypes.training,
            output_uri=False
        )
        task.connect_configuration(CONFIG)

    # Initialize datasets
    vocab = Vocabulary(LABELS, case_sensitive=CASE_SENSITIVE)
    collate = Collate(pad_idx=vocab.pad_token_idx)
    match DATASET_TYPE:
        case 'json':
            json_dataset_args = dict(dataset_path=DATASET_PATH, vocab=vocab, img_size=IMG_SIZE, input_channels=INPUT_CHANNELS)
            dataset_train = JsonDataset(sample=TRAIN_FOLDER, **json_dataset_args)
            dataset_valid = JsonDataset(sample=VAL_FOLDER, **json_dataset_args)

        case 'lmdb':
            train_lmdb_paths = get_lmdb_paths(DATASET_PATH / TRAIN_FOLDER)
            val_lmdb_paths = get_lmdb_paths(DATASET_PATH / VAL_FOLDER)
            dataset_train = create_lmdb_dataset(train_lmdb_paths, TRAIN_FOLDER, vocab)
            dataset_valid = create_lmdb_dataset(val_lmdb_paths, VAL_FOLDER, vocab)

    dataloader_train = DataLoader(dataset_train, batch_size=TRAIN_BATCH_SIZE, collate_fn=collate, shuffle=True, num_workers=NUM_WORKERS)
    dataloader_valid = DataLoader(dataset_valid, batch_size=TRAIN_BATCH_SIZE, collate_fn=collate, shuffle=False, num_workers=NUM_WORKERS)

    # Print dataset sizes
    print(f'\nTrain size: {len(dataset_train)} images')
    print(f'Val size: {len(dataset_valid)} images\n')

    # Model
    model = ViTSTRTransducer(
        backbone_type=BACKBONE_TYPE,
        vocab=vocab,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        input_size=IMG_SIZE,
        input_channels=INPUT_CHANNELS,
        lr=LR,
        loss=LOSS,
        weight_decay=WEIGHT_DECAY,
        dropout_rate=DROPOUT_RATE,
        gamma=GAMMA,
        training=True,
        sdp_backend=SDP_BACKEND
    )
    
    # Delete previous run results in output directory
    if os.path.exists(OUTPUT_DIR) and len(os.listdir(OUTPUT_DIR)) > 0:
        for subdir in os.listdir(OUTPUT_DIR):
            os.system(f'rm -r {os.path.join(OUTPUT_DIR, subdir)}')
            
    # Callbacks and logger
    csv_logger = CSVLogger(OUTPUT_DIR, name=None, version='train', prefix=None)
    tb_logger = TensorBoardLogger(OUTPUT_DIR, name=None, version='train', prefix=None, default_hp_metric=False)
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(WEIGHTS_DIR),
        filename=WEIGHTS_NAME,
        save_weights_only=True,
        mode='max',
        monitor='val_f1',
        enable_version_counter=False
    )
    on_exception_checkpoint_callback = OnExceptionCheckpointWeightsOnly(
        dirpath=str(WEIGHTS_DIR),
        filename=WEIGHTS_NAME,
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_f1',
        mode='max',
        patience=PATIENCE, 
        verbose=True, 
        min_delta=MIN_DELTA
    )
    
    # Trainer
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[
            checkpoint_callback,
            lr_monitor_callback,
            early_stopping_callback,
            on_exception_checkpoint_callback
        ], 
        logger=[csv_logger, tb_logger], 
        devices=[DEVICE_IDX],
        log_every_n_steps=len(dataloader_train), # Every epoch
        precision=PRECISION
    )
    
    # Training
    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_valid)

    if CML_ENABLED:
        task.output_uri = True
        output_model_pt = OutputModel(task, config_dict=CONFIG, framework='PyTorch')
        output_model_pt.update_weights(str(WEIGHTS_FULL_PATH), auto_delete_file=False)

    # Validation on test set
    model = ViTSTRTransducer.load_from_checkpoint(WEIGHTS_FULL_PATH, training=False).eval()
    model.freeze()
    trainer.logger = False
    trainer.validate(model, dataloaders=dataloader_valid, ckpt_path=WEIGHTS_FULL_PATH)
    
    # Save history as plots
    history_visualizer = History(TRAIN_DIR)
    lr_fig = history_visualizer.save_lr(column_name='lr-RAdam')
    loss_fig = history_visualizer.save_metric(column_name='loss', metric_name='Loss')
    f1_fig = history_visualizer.save_metric(column_name='f1', metric_name='F1-Score')
    acc_fig = history_visualizer.save_metric(column_name='acc', metric_name='Accuracy')

    # Log images to ClearML
    if CML_ENABLED:
        task.logger.report_plotly('Learning Rate', 'Learning Rate', figure=lr_fig)
        task.logger.report_plotly('Loss', 'Loss', figure=loss_fig)
        task.logger.report_plotly('F1-Score', 'F1-Score', figure=f1_fig)
        task.logger.report_plotly('Accuracy', 'Accuracy', figure=acc_fig)

    model.to(f'cuda:{DEVICE_IDX}')

    # Predict few samples from test set
    
    predictions_visualizer = Predictor(output_path=VAL_EXAMPLES_PATH, input_channels=INPUT_CHANNELS)
    predictions_visualizer.caption_dataloader(dataloader=dataloader_valid, model=model)
    if CML_ENABLED:
        for image in os.listdir(VAL_EXAMPLES_PATH):
            task.logger.report_image(title='Predictions', series=image, local_path=os.path.join(VAL_EXAMPLES_PATH, image), iteration=0)
    
    # Save model as torchscript
    model.bfloat16().to_torchscript(TS_BF16_WEIGHTS)
    model.half().to_torchscript(TS_FP16_WEIGHTS)
    
    # Upload weights to ClearML
    if CML_ENABLED:
        for ts_model in [TS_BF16_WEIGHTS, TS_FP16_WEIGHTS]:
            output_model_ts = OutputModel(task, config_dict=CONFIG, framework='PyTorch')
            output_model_ts.update_weights(str(ts_model), auto_delete_file=False)


def test():
    # ClearML initialization
    if CML_ENABLED:
        task: Task = Task.init(
            project_name=PROJECT_NAME,
            task_name=TASK_NAME,
            task_type=Task.TaskTypes.testing
        )
        task.connect_configuration(CONFIG)

    # Load trained model
    model = ViTSTRTransducer.load_from_checkpoint(WEIGHTS_FULL_PATH, training=False).eval()
    model.freeze()
    
    # Test dataloader
    vocab = model.vocab
    collate = Collate(pad_idx=vocab.pad_token_idx)
    match DATASET_TYPE:
        case 'json':
            dataset_test = JsonDataset(sample=TEST_FOLDER, dataset_path=DATASET_PATH, vocab=vocab, img_size=IMG_SIZE, input_channels=INPUT_CHANNELS)

        case 'lmdb':
            test_lmdb_paths = get_lmdb_paths(DATASET_PATH / TEST_FOLDER)
            dataset_test = create_lmdb_dataset(test_lmdb_paths, TEST_FOLDER, vocab)
    dataloader_test = DataLoader(dataset_test, batch_size=TEST_BATCH_SIZE, collate_fn=collate, shuffle=False, num_workers=NUM_WORKERS)

    # Validate test dataloader
    trainer = Trainer(devices=[DEVICE_IDX], logger=False, enable_checkpointing=False, precision=PRECISION)
    start_time = time.time()
    trainer.test(model=model, dataloaders=dataloader_test)
    end_time = time.time()
    avg_inference_time = (end_time - start_time) / len(dataloader_test)
    print(f'Inference time: {avg_inference_time:.4f}s per image')

    model.to(f'cuda:{DEVICE_IDX}')

    # Save predictions
    predictions_visualizer = Predictor(output_path=TEST_EXAMPLES_PATH, input_channels=INPUT_CHANNELS)
    predictions_visualizer.caption_dataloader(dataloader=dataloader_test, model=model)

    
if __name__ == '__main__':
    train()
    test()
    print(f'\nDone! Results saved in {OUTPUT_DIR}\n')
