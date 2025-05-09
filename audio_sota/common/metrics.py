import torchmetrics

class UnweightedAverageRecall(torchmetrics.Recall):
    """
    Wrapper for torchmetrics.Recall with average='macro' to compute UAR.
    """
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(task="multiclass", num_classes=num_classes, average='macro', **kwargs)

# Example usage:
# uar_metric = UnweightedAverageRecall(num_classes=8)
# uar_metric.update(preds, target)
# uar_score = uar_metric.compute()
