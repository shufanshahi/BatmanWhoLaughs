"""
Configuration module for GraphSmile
Handles all hyperparameters, GPU settings, and dataset configurations
"""

import argparse
import os
import torch


def get_args():
    """Parse command line arguments for GraphSmile training"""
    parser = argparse.ArgumentParser(description='GraphSmile - Multimodal Emotion Recognition')

    # ===================== GPU & Training Settings =====================
    parser.add_argument('--no_cuda',
                        action='store_true',
                        default=False,
                        help='Disable CUDA (use CPU only)')
    
    parser.add_argument('--gpu',
                        default='0',
                        type=str,
                        help='GPU device IDs to use (e.g., "0" or "0,1,2,3")')
    
    parser.add_argument('--port',
                        default='12355',
                        type=str,
                        help='Master port for distributed training')
    
    parser.add_argument('--seed',
                        default=2024,
                        type=int,
                        help='Random seed for reproducibility')

    # ===================== Dataset Settings =====================
    parser.add_argument('--dataset',
                        default='IEMOCAP',
                        type=str,
                        choices=['IEMOCAP', 'IEMOCAP4', 'MELD', 'CMUMOSEI7'],
                        help='Dataset to train and test')
    
    parser.add_argument('--classify',
                        default='emotion',
                        type=str,
                        choices=['emotion', 'sentiment'],
                        help='Classification task: emotion or sentiment')

    # ===================== Model Architecture =====================
    parser.add_argument('--modals',
                        default='avl',
                        type=str,
                        help='Modalities to use (a=audio, v=visual, l=language/text)')
    
    parser.add_argument('--textf_mode',
                        default='textf0',
                        type=str,
                        choices=['textf0', 'textf1', 'textf2', 'textf3', 
                                'concat2', 'concat4', 'sum2', 'sum4'],
                        help='Text feature mode')
    
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=256,
                        help='Hidden dimension size')
    
    parser.add_argument('--win',
                        nargs='+',
                        type=int,
                        default=[17, 17],
                        help='Window size [win_past, win_future], -1 = all nodes')
    
    parser.add_argument('--heter_n_layers',
                        nargs='+',
                        type=int,
                        default=[6, 6, 6],
                        help='Number of heterogeneous graph layers for each branch [TV, TA, VA]')
    
    parser.add_argument('--drop',
                        type=float,
                        default=0.3,
                        help='Dropout rate')
    
    parser.add_argument('--shift_win',
                        type=int,
                        default=12,
                        help='Window size for sentiment shift detection')

    # ===================== Training Hyperparameters =====================
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='Learning rate')
    
    parser.add_argument('--l2',
                        type=float,
                        default=0.0001,
                        help='L2 regularization weight')
    
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='Batch size')
    
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='Number of training epochs')
    
    parser.add_argument('--valid_ratio',
                        type=float,
                        default=0.1,
                        help='Validation set ratio')

    # ===================== Loss Function Settings =====================
    parser.add_argument('--loss_type',
                        default='emo_sen_sft',
                        type=str,
                        choices=['emo', 'sen', 'emo_sen', 'emo_sft', 
                                'sen_sft', 'emo_sen_sft', 'auto', 'epoch'],
                        help='Loss combination strategy')
    
    parser.add_argument('--lambd',
                        nargs='+',
                        type=float,
                        default=[1.0, 1.0, 1.0],
                        help='Loss weights [emotion, sentiment, shift]')

    # ===================== Logging & Checkpointing =====================
    parser.add_argument('--tensorboard',
                        action='store_true',
                        default=False,
                        help='Enable TensorBoard logging')
    
    parser.add_argument('--log_interval',
                        type=int,
                        default=10,
                        help='Log every N batches')
    
    parser.add_argument('--save_dir',
                        type=str,
                        default='./checkpoints',
                        help='Directory to save model checkpoints')

    # ===================== Dataset Paths =====================
    parser.add_argument('--data_dir',
                        type=str,
                        default='./data',
                        help='Root directory for datasets')

    args = parser.parse_args()
    return args


def setup_gpu(args):
    """
    Configure GPU environment variables and device settings
    
    Args:
        args: Parsed arguments containing GPU configuration
        
    Returns:
        tuple: (cuda_available, device, world_size)
    """
    # Set visible GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Setup for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available() and not args.no_cuda
    
    if cuda_available:
        world_size = torch.cuda.device_count()
        os.environ['WORLD_SIZE'] = str(world_size)
        
        print("=" * 60)
        print("GPU Configuration:")
        print(f"  CUDA Available: {cuda_available}")
        print(f"  GPU Count: {world_size}")
        print(f"  GPU IDs: {args.gpu}")
        print(f"  Master Port: {args.port}")
        
        for i in range(world_size):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print("=" * 60)
    else:
        world_size = 1
        print("=" * 60)
        print("Running on CPU (CUDA not available or disabled)")
        print("=" * 60)
    
    # Set device
    if cuda_available:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    return cuda_available, device, world_size


def get_dataset_config(args):
    """
    Get dataset-specific configurations
    
    Args:
        args: Parsed arguments
        
    Returns:
        dict: Dataset configuration with embedding dimensions and number of classes
    """
    dataset_configs = {
        'IEMOCAP': {
            'embedding_dims': [1024, 342, 1582],  # [text, visual, audio]
            'n_classes_emo': 6,
            'path_key': 'IEMOCAP_path',
            'emotions': ['angry', 'happy', 'sad', 'neutral', 'excited', 'frustrated']
        },
        'IEMOCAP4': {
            'embedding_dims': [1024, 512, 100],
            'n_classes_emo': 4,
            'path_key': 'IEMOCAP4_path',
            'emotions': ['angry', 'happy', 'sad', 'neutral']
        },
        'MELD': {
            'embedding_dims': [1024, 342, 300],
            'n_classes_emo': 7,
            'path_key': 'MELD_path',
            'emotions': ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
        },
        'CMUMOSEI7': {
            'embedding_dims': [1024, 35, 384],
            'n_classes_emo': 7,
            'path_key': 'CMUMOSEI7_path',
            'emotions': ['highly_negative', 'negative', 'slightly_negative', 
                        'neutral', 'slightly_positive', 'positive', 'highly_positive']
        }
    }
    
    if args.dataset not in dataset_configs:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    config = dataset_configs[args.dataset]
    
    # Construct dataset path
    config['path'] = os.path.join(args.data_dir, f"{args.dataset}.pkl")
    
    return config


def get_optimizer_config(args):
    """
    Get optimizer configuration
    
    Args:
        args: Parsed arguments
        
    Returns:
        dict: Optimizer configuration
    """
    return {
        'type': 'AdamW',
        'lr': args.lr,
        'weight_decay': args.l2,
        'amsgrad': True
    }


def print_config(args):
    """Print all configuration parameters"""
    print("\n" + "=" * 60)
    print("GraphSmile Configuration")
    print("=" * 60)
    
    # Group parameters
    groups = {
        'Dataset': ['dataset', 'classify', 'data_dir'],
        'Model': ['hidden_dim', 'textf_mode', 'modals', 'win', 'heter_n_layers', 'drop', 'shift_win'],
        'Training': ['batch_size', 'epochs', 'lr', 'l2', 'valid_ratio'],
        'Loss': ['loss_type', 'lambd'],
        'GPU': ['no_cuda', 'gpu', 'port'],
        'Other': ['seed', 'tensorboard', 'save_dir']
    }
    
    for group_name, param_names in groups.items():
        print(f"\n{group_name}:")
        for param in param_names:
            if hasattr(args, param):
                value = getattr(args, param)
                print(f"  {param}: {value}")
    
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    # Test configuration
    args = get_args()
    print_config(args)
    cuda_available, device, world_size = setup_gpu(args)
    dataset_config = get_dataset_config(args)
    print(f"\nDataset Config: {dataset_config}")
