import os, json
from PCA import HSIDataLoader
from utils import recorder
from trainer import get_trainer1
from utils import config_path_prefix
import datetime

DEFAULT_RES_SAVE_PATH_PREFIX = "./res"

def train_by_param(param):
    recorder.reset()
    dataloader = HSIDataLoader(param)
    train_loader, unlabel_loader, test_loader, all_loader = dataloader.generate_torch_dataset()
    trainer = get_trainer1(param)
    trainer.train(train_loader, unlabel_loader, test_loader)
    eval_res = trainer.final_eval(test_loader)
    recorder.record_param(param)
    recorder.record_eval(eval_res)
    return recorder


include_path = [
'scmtformer.json',
]
def run_all():
    save_path_prefix = DEFAULT_RES_SAVE_PATH_PREFIX
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    for name in include_path:
        path_param = '%s/%s' % (config_path_prefix, name)
        with open(path_param, 'r') as fin:
            param = json.loads(fin.read())
        uniq_name = param.get('uniq_name', name)
        print(datetime.datetime.now())
        print('start to train %s...' % uniq_name)
        train_by_param(param)
        print('model eval done of %s...' % uniq_name)
        path = '%s/%s' % (save_path_prefix, uniq_name)
        recorder.to_file(path)

if __name__ == "__main__":
    run_all()
