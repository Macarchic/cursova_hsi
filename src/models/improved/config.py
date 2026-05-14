from src.models.baseline.config import Config

CONFIGS = {
    'IP':   Config(dataset='IP',   patch_size=15, pca_components=150, kernel_size=9,
                   num_classes=16, batch_size=200),
    'PU':   Config(dataset='PU',   patch_size=25, pca_components=20,  kernel_size=17,
                   num_classes=9,  batch_size=100),
    'WHHH': Config(dataset='WHHH', patch_size=31, pca_components=135, kernel_size=19,
                   num_classes=22, batch_size=20),
    'HC':   Config(dataset='HC',   patch_size=30, pca_components=135, kernel_size=19,
                   num_classes=16, batch_size=20),
    'LK':   Config(dataset='LK',   patch_size=30, pca_components=135, kernel_size=19,
                   num_classes=9,  batch_size=20),
}
