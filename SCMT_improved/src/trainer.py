from models import SCMT as SQSFormer
import utils
from models.SCMT import Init
from utils import recorder
from evaluation import HSIEvaluation
from utils import device
import logging
import numpy as np
import torch.optim as optim
import torch
from torch import nn

logger = logging.getLogger(__name__)

class SCMTTrainer(object):
    def __init__(self, params) -> None:
        self.params = params
        self.net_params = params['net']
        self.train_params = params['train']
        self.device = device
        self.evalator = HSIEvaluation(param=params)
        self.net = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.clip = 15
        self.real_init()

    def real_init(self):
        """初始化网络、损失函数和优化器"""
        # 初始化网络
        self.net = SQSFormer.SCMT(self.params).to(self.device)
        Init(self.net)

        # 初始化损失函数 (+ label smoothing для регуляризації при 10 зразках/клас)
        label_smoothing = self.train_params.get('label_smoothing', 0.05)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # 初始化优化器
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Cosine annealing LR: плавне зниження lr -> lr*0.1 для кращого фінального збігу
        epochs = self.train_params.get('epochs', 100)
        if self.train_params.get('cosine_lr', True):
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=self.lr * 0.1)
        else:
            self.scheduler = None

    def get_loss(self, outputs, target):
        """计算损失函数。如果 outputs 是元组（例如包含多个输出），默认使用第一个输出计算损失。"""
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        return self.criterion(logits, target)

    def train(self, train_loader, unlabel_loader=None, test_loader=None):
        """训练模型"""
        epochs = self.params['train'].get('epochs')

        total_loss = 0
        epoch_avg_loss = utils.AvgrageMeter()
        for i, (data, target) in enumerate(train_loader):
            break  # 只检查第一个 batch，避免打印太多数据

        for epoch in range(epochs):
            self.net.train()
            epoch_avg_loss.reset()

            for i, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                outputs = self.net(data)

                loss = self.get_loss(outputs, target)
                self.optimizer.zero_grad()  # 优化器
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
                self.optimizer.step()
                total_loss += loss.item()
                epoch_avg_loss.update(loss.item(), data.shape[0])

            recorder.append_index_value("epoch_loss", epoch + 1, epoch_avg_loss.get_avg())
            print('[Epoch: %d]  [epoch_loss: %.5f]  [all_epoch_loss: %.5f] [current_batch_loss: %.5f] [batch_num: %s]' % (
                epoch + 1,
                epoch_avg_loss.get_avg(),
                total_loss / (epoch + 1),
                loss.item(), epoch_avg_loss.get_num()))

            if self.scheduler is not None:
                self.scheduler.step()

            # In-training eval: інтервал конфігурований (eval_interval).
            # Раніше було жорстко % 201 при epochs=100 -> ніколи не спрацьовувало.
            # 0 (default) = вимкнено, щоб не сповільнювати навчання.
            eval_interval = self.train_params.get('eval_interval', 0)
            if test_loader and eval_interval and (epoch + 1) % eval_interval == 0:
                print("开始测试")
                y_pred_test, y_test = self.test(test_loader, save_to_excel=False)
                temp_res = self.evalator.eval(y_test, y_pred_test)
                recorder.append_index_value("train_oa", epoch + 1, temp_res['oa'])
                recorder.append_index_value("train_aa", epoch + 1, temp_res['aa'])
                recorder.append_index_value("train_kappa", epoch + 1, temp_res['kappa'])
                print('[--TEST--] [Epoch: %d] [oa: %.5f] [aa: %.5f] [kappa: %.5f] [num: %s]' % (
                    epoch + 1, temp_res['oa'], temp_res['aa'], temp_res['kappa'], str(y_test.shape)))

        print('Finished Training')
        return True

    def final_eval(self, test_loader):
        """最终评估模型"""

        y_pred_test, y_test = self.test(test_loader, save_to_excel=False)
        temp_res = self.evalator.eval(y_test, y_pred_test)

        return temp_res

    def get_logits(self, output):
        """获取模型的 logits（即分类得分）"""
        if isinstance(output, tuple):
            return output[0]  # 如果输出是元组，返回第一个元素
        return output

    def test(self, test_loader, save_to_excel=False):
        """测试模型"""
        count = 0
        self.net.eval()
        y_pred_test = 0
        y_test = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(self.device)
            outputs = self.get_logits(self.net(inputs))
            # Раніше batch з одним зразком (1D-вихід) мовчки пропускався (continue),
            # що викидало зразки з фінальної оцінки. Тепер повертаємо йому вимір батча.
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            labels = labels.cpu().numpy()
            if count == 0:
                y_pred_test = outputs
                y_test = labels
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))
                y_test = np.concatenate((y_test, labels))

        return y_pred_test, y_test


def get_trainer1(params):
    """获取训练器实例"""
    trainer_type = params['net']['trainer']
    if trainer_type == "scmtformer":
        return SCMTTrainer(params)
    raise Exception("Trainer not implemented!")