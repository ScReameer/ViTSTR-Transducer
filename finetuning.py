import os
import yaml
import argparse
from multiprocessing import cpu_count

import torch
torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from clearml import Task, OutputModel

from src.net.model import ViTSTRTransducer
from src.data_processing.dataset import Collate, PriceDataset
from src.utils.predictor import Predictor
from src.utils.history import History

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='path to config file')
parser.add_argument('--device', type=int, default=0, help='CUDA device id')
parser.add_argument('--output-dir', type=str, required=True, help='output path directory')
args = parser.parse_args()

CONFIG = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
DEVICE_IDX = args.device
OUTPUT_PATH = args.output_dir
WEIGHTS_PATH = f'{OUTPUT_PATH}/finetuning/weights/ViTSTR-FP32.ckpt'
TS_WEIGHTS_PATH = f'{OUTPUT_PATH}/finetuning/weights/ViTSTR-TS-FP16.torchscript'

# Project hyperparameters
NUM_WORKERS = cpu_count() // 2 # Use half of available CPU's
DATASET_PATH = CONFIG['DATASET_PATH']
PROJECT_NAME = CONFIG['CML_PROJECT']
TASK_NAME = CONFIG['CML_TASK']
WEIGHTS = CONFIG['ViTSTR-T']['WEIGHTS'] # ckpt file
# Training hyperparameters
BATCH_SIZE = CONFIG['ViTSTR-T']['TRAIN']['BATCH_SIZE']
MAX_EPOCHS = CONFIG['ViTSTR-T']['TRAIN']['MAX_EPOCHS']
LR = float(CONFIG['ViTSTR-T']['TRAIN']['LR'])
GAMMA = CONFIG['ViTSTR-T']['TRAIN']['GAMMA']
TRAIN_FOLDER, VAL_FOLDER = CONFIG['ViTSTR-T']['TRAIN']['FOLDERS']
PATIENCE = int(CONFIG['ViTSTR-T']['TRAIN']['PATIENCE'])


def finetune():
    # ClearML initialization
    task: Task = Task.init(
        project_name=PROJECT_NAME,
        task_name=TASK_NAME,
        task_type=Task.TaskTypes.training,
        output_uri=False
    )
    task.connect_configuration(CONFIG)
    model = ViTSTRTransducer.load_from_checkpoint(
        WEIGHTS, 
        map_location=f'cuda:{DEVICE_IDX}', 
        training=False,
        lr=LR,
        gamma=GAMMA
    ).train(True)
    # Initialize datasets
    vocab = model.vocab
    dataset_train = PriceDataset(DATASET_PATH, vocab, sample=TRAIN_FOLDER, img_size=model.input_size)
    dataset_valid = PriceDataset(DATASET_PATH, vocab, sample=VAL_FOLDER, img_size=model.input_size)
    collate = Collate(pad_idx=vocab.digit2idx['<PAD>'])
    # Initialize dataloaders
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, collate_fn=collate, shuffle=True, num_workers=NUM_WORKERS)
    dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, collate_fn=collate, shuffle=False, num_workers=NUM_WORKERS)
    # Print dataset sizes
    print(f'\nTrain size: {len(dataset_train)} images')
    print(f'Val size: {len(dataset_valid)} images\n')


    
    # Delete previous run results in output directory
    if os.path.exists(OUTPUT_PATH) and len(os.listdir(OUTPUT_PATH)) > 0:
        for version in os.listdir(OUTPUT_PATH):
            os.system(f'rm -r {os.path.join(OUTPUT_PATH, version)}')
            
    # Callbacks and logger
    csv_logger = CSVLogger(OUTPUT_PATH, name=None, version='train', prefix=None)
    tb_logger = TensorBoardLogger(OUTPUT_PATH, name=None, version='train', prefix=None, default_hp_metric=False)
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(OUTPUT_PATH, 'finetuning', 'weights'),
        filename='ViTSTR-FP32',
        save_weights_only=True,
        mode='max',
        monitor='val_f1',
        enable_version_counter=False
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_f1',
        mode='max',
        patience=PATIENCE, 
        verbose=True, 
        min_delta=1e-3
    )
    
    # Trainer
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[
            checkpoint_callback,
            lr_monitor_callback,
            early_stopping_callback
        ], 
        logger=[csv_logger, tb_logger], 
        devices=[DEVICE_IDX],
        log_every_n_steps=len(dataloader_train) # Every epoch
    )
    
    # Training
    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_valid)
    task.output_uri = True
    output_model_pt = OutputModel(task, config_dict=CONFIG, framework='PyTorch')
    output_model_pt.update_weights(WEIGHTS_PATH, auto_delete_file=False)
    # Validation on test set
    model = ViTSTRTransducer.load_from_checkpoint(WEIGHTS_PATH, training=False).eval()
    model.freeze()
    trainer.logger = False
    trainer.validate(model, dataloaders=dataloader_valid, ckpt_path=WEIGHTS_PATH)
    
    # Save history as plots
    history_visualizer = History(f'{OUTPUT_PATH}/train')
    lr_fig = history_visualizer.save_lr(column_name='lr-RAdam')
    loss_fig = history_visualizer.save_metric(column_name='loss', metric_name='Loss')
    f1_fig = history_visualizer.save_metric(column_name='f1', metric_name='F1-Score')
    acc_fig = history_visualizer.save_metric(column_name='acc', metric_name='Accuracy')
    # Log images to ClearML
    task.logger.report_plotly('Learning Rate', 'Learning Rate', figure=lr_fig)
    task.logger.report_plotly('Loss', 'Loss', figure=loss_fig)
    task.logger.report_plotly('F1-Score', 'F1-Score', figure=f1_fig)
    task.logger.report_plotly('Accuracy', 'Accuracy', figure=acc_fig)

    # Predict few samples from test set
    images_output_path = f'{OUTPUT_PATH}/train/predictions'
    predictions_visualizer = Predictor(img_size=model.input_size, device=DEVICE_IDX, output_path=images_output_path)
    predictions_visualizer.caption_dataloader(dataloader=dataloader_valid, model=model)
    for image in os.listdir(images_output_path):
        task.logger.report_image(title='Predictions', series=image, local_path=os.path.join(images_output_path, image), iteration=0)
    
    # Change forward method to inference forward method (unmasked inputs)
    model.forward = model.forward_inference
    model.to(f'cuda:{DEVICE_IDX}')
    
    # Save model as torchscript
    model.half().to_torchscript(TS_WEIGHTS_PATH)
    # Upload weights to ClearML
    output_model_ts = OutputModel(task, config_dict=CONFIG, framework='PyTorch')
    output_model_ts.update_weights(TS_WEIGHTS_PATH, auto_delete_file=False)
    
    print(f'\nDone! Results saved in {OUTPUT_PATH}\n')

if __name__ == '__main__':
    finetune()
    os.system(f'python test.py --config {args.config} --device {DEVICE_IDX} --weights {WEIGHTS_PATH}')