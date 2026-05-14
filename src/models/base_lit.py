import numpy as np
import torch
import torch.nn as nn
import lightning as L

from src.utils import compute_metrics


class EpochLogger(L.Callback):
    """
    Prints one line per epoch after validation using metrics stored directly on pl_module.
    Avoids reading from trainer.callback_metrics (has timing/NaN issues in Lightning 2.x).

    Within-epoch progress: on_train_epoch_start prints "Ep X/100 (X%) ..." without a newline.
    After validation: on_validation_epoch_end overwrites that line via \\r with full metrics.
    """

    def __init__(self, log_every: int = 1):
        self.log_every = log_every

    def on_train_epoch_start(self, trainer, pl_module):
        ep    = trainer.current_epoch + 1
        ep_pct = ep / trainer.max_epochs
        print(f'  Ep {ep:3d}/{trainer.max_epochs} ({ep_pct:3.0%}) ...', end='', flush=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.log_every != 0 and epoch != 1:
            print()  # commit the "..." line
            return
        ep_pct = epoch / trainer.max_epochs
        t_loss = getattr(pl_module, '_log_train_loss', float('nan'))
        t_acc  = getattr(pl_module, '_log_train_acc',  float('nan'))
        v_loss = getattr(pl_module, '_log_val_loss',   float('nan'))
        v_oa   = getattr(pl_module, '_log_val_oa',     float('nan'))
        v_aa   = getattr(pl_module, '_log_val_aa',     float('nan'))
        v_k    = getattr(pl_module, '_log_val_kappa',  float('nan'))
        print(
            f'\r  Ep {epoch:3d}/{trainer.max_epochs} ({ep_pct:3.0%})'
            f'  train: loss={t_loss:.4f}  acc={t_acc:.1%}'
            f'  │  val: loss={v_loss:.4f}  OA={v_oa:.1%}  AA={v_aa:.1%}  κ={v_k:.4f}'
        )


class BaseLitModule(L.LightningModule):
    """
    Shared Lightning training/validation loop for all HSI models.
    Subclasses must implement:
      - __init__(cfg): call super().__init__(cfg), then set self.model
      - forward(x): delegate to self.model
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss()
        self._val_preds:  list = []
        self._val_trues:  list = []
        self._val_losses: list = []
        self._train_losses: list = []
        self._train_accs:   list = []
        # Attrs read by EpochLogger (set each epoch, never NaN after epoch 1)
        self._log_train_loss: float = float('nan')
        self._log_train_acc:  float = float('nan')
        self._log_val_loss:   float = float('nan')
        self._log_val_oa:     float = float('nan')
        self._log_val_aa:     float = float('nan')
        self._log_val_kappa:  float = float('nan')

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc',  acc,  prog_bar=True, on_step=False, on_epoch=True)
        self._train_losses.append(loss.item())
        self._train_accs.append(acc.item())
        return loss

    def on_train_epoch_end(self):
        if self._train_losses:
            self._log_train_loss = float(np.mean(self._train_losses))
            self._log_train_acc  = float(np.mean(self._train_accs))
            self._train_losses.clear()
            self._train_accs.clear()

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self._val_preds.append(logits.argmax(1).cpu())
        self._val_trues.append(y.cpu())
        self._val_losses.append(loss.item())

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        preds = torch.cat(self._val_preds).numpy()
        trues = torch.cat(self._val_trues).numpy()
        m = compute_metrics(preds, trues, self.cfg.num_classes)
        self.log('val_OA',    m['OA'],    prog_bar=True)
        self.log('val_AA',    m['AA'],    prog_bar=False)
        self.log('val_kappa', m['Kappa'], prog_bar=False)
        self.log('val_loss',  float(np.mean(self._val_losses)), prog_bar=False)
        self._log_val_oa    = m['OA']
        self._log_val_aa    = m['AA']
        self._log_val_kappa = m['Kappa']
        self._log_val_loss  = float(np.mean(self._val_losses))
        self._val_preds.clear()
        self._val_trues.clear()
        self._val_losses.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
