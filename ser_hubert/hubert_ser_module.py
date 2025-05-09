import torch, torch.nn as nn, torch.nn.functional as F
import pytorch_lightning as pl
from transformers import HubertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torchmetrics

class HubertSER(pl.LightningModule):
    """HuBERT‑based speech‑emotion classifier with mean/max pooling."""

    def __init__(self,
                 hubert_name: str = "facebook/hubert-base-ls960",
                 num_classes: int = 8,
                 class_weights: torch.Tensor | None = None,
                 lr: float = 2e-5,
                 warmup_steps: int = 500,
                 total_steps: int = 10_000,
                 freeze_epochs: int = 2,
                 pooling: str = "mean",
                 dropout: float = 0.1):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])  # weights tensor too big for ckpt

        # ------------ backbone ----------------
        self.hubert = HubertModel.from_pretrained(hubert_name)
        for p in self.hubert.parameters():
            p.requires_grad = False  # will unfreeze later
        self.hubert.eval()

        hid = self.hubert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid, num_classes)

        self.crit = nn.CrossEntropyLoss(weight=class_weights)

        # ------------ metrics ------------------
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc   = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc  = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_uar   = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.test_uar  = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro")

    # ---------- forward & pooling ------------
    def _pool(self, hidden, mask):
        """Pools hidden states. Handles potential mask shape mismatch."""
        # hidden shape: [Batch, SeqLen, HiddenDim]
        # mask shape (from dataloader): [Batch, InputSeqLen] - potentially WRONG length

        # --- MEAN POOLING ---
        if self.hparams.pooling == "mean":
            # For mean pooling, simply average over the sequence dimension (dim=1).
            # Ignore the potentially incorrectly shaped 'mask' argument.
            return hidden.mean(dim=1)

        # --- MAX POOLING ---
        elif self.hparams.pooling == "max":
            # Max pooling *requires* a mask with the *same sequence length* as 'hidden'.
            if mask is not None and mask.shape[1] != hidden.shape[1]:
                 # If mask shape is wrong, log a warning and perform unmasked max pooling.
                 # This might include padding tokens if not handled internally by HubertModel.
                 print(f"Warning: Pooling mask sequence length ({mask.shape[1]}) "
                       f"does not match hidden state sequence length ({hidden.shape[1]}). "
                       f"Performing unmasked max pooling.")
                 return hidden.max(dim=1).values
            elif mask is not None:
                 # Mask shape matches, proceed with masked max pooling
                 # Ensure mask is boolean for masked_fill
                 mask_bool = (mask == 1).unsqueeze(-1) # Add feature dim
                 # Expand mask to match hidden state dimensions (B, SeqLen, HiddenDim)
                 expanded_mask = mask_bool.expand_as(hidden)
                 # Fill masked positions with a very small number before max pooling
                 hidden_masked = hidden.masked_fill(~expanded_mask, -float('inf'))
                 return hidden_masked.max(dim=1).values
            else:
                 # No mask provided, perform unmasked max pooling
                 return hidden.max(dim=1).values
        else:
             raise ValueError(f"pooling mode '{self.hparams.pooling}' must be 'mean' or 'max'")

    def forward(self, input_values, attention_mask=None):
        out = self.hubert(input_values=input_values, attention_mask=attention_mask)
        pooled = self._pool(out.last_hidden_state, attention_mask)
        return self.fc(self.dropout(pooled))

    # ---------- shared step ------------------
    def _step(self, batch):
        # Ensure keys exist, otherwise raise an error or handle appropriately
        if "input_values" not in batch:
            raise KeyError("Batch missing 'input_values'")
        if "labels" not in batch:
            raise KeyError("Batch missing 'labels'")

        x      = batch["input_values"]
        mask   = batch.get("attention_mask") # .get() is fine for optional keys
        y      = batch["labels"]
        logits = self(x, mask)
        loss   = self.crit(logits, y)
        return loss, logits, y

    # ---------- PL hooks ---------------------
    def training_step(self, batch, _):
        loss, logits, y = self._step(batch)
        self.log("train_loss", loss, prog_bar=False)
        self.train_acc(logits, y)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, logits, y = self._step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(logits, y);  self.val_uar(logits, y)
        self.log("val_acc", self.val_acc, prog_bar=False)
        self.log("val_uar", self.val_uar, prog_bar=True)

    def test_step(self, batch, _):
        loss, logits, y = self._step(batch)
        self.log("test_loss", loss)
        self.test_acc(logits, y); self.test_uar(logits, y)
        self.log("test_acc", self.test_acc); self.log("test_uar", self.test_uar)

    def on_validation_epoch_end(self):
        # Reset metrics manually if needed, PL usually handles this
        # self.val_acc.reset(); self.val_uar.reset()
        pass # PL handles reset by default

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {"params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 1e-2},
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        opt = AdamW(params, lr=self.hparams.lr, eps=1e-8)
        sch = get_linear_schedule_with_warmup(opt, self.hparams.warmup_steps, self.hparams.total_steps)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}, "monitor": "val_uar"}

    def on_train_epoch_start(self):
        if self.current_epoch == self.hparams.freeze_epochs:
            if self.trainer.is_global_zero: # Use trainer property for rank check
                print(f"Epoch {self.current_epoch}: unfreezing HuBERT…")
            for p in self.hubert.parameters():
                p.requires_grad = True
            self.hubert.train()
        elif self.current_epoch < self.hparams.freeze_epochs:
            self.hubert.eval()
            for p in self.hubert.parameters():
                p.requires_grad = False
