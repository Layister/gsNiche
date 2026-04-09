import logging
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .model import GATModel

logger = logging.getLogger(__name__)


def reconstruction_loss(decoded, x):
    """Compute the mean squared error loss."""
    return F.mse_loss(decoded, x)


def label_loss(pred_label, true_label):
    """Compute the cross-entropy loss."""
    return F.cross_entropy(pred_label, true_label.long())


def variance_regularization_loss(z: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4):
    """VICReg-style variance term to prevent representation collapse."""
    z_centered = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z_centered.var(dim=0, unbiased=False) + eps)
    return torch.mean(F.relu(target_std - std))


def covariance_regularization_loss(z: torch.Tensor):
    """VICReg-style covariance term to reduce redundant latent dimensions."""
    n, d = z.shape
    if n <= 1 or d <= 1:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)

    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / max(1, n - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).mean()


def _set_torch_seed(seed):
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_weight_decay(params):
    if hasattr(params, "weight_decay"):
        return float(params.weight_decay)
    if hasattr(params, "gcn_decay"):
        return float(params.gcn_decay)
    return 0.01


class ModelTrainer:
    def __init__(self, node_x, graph_dict, params, label=None):
        """Initialize the ModelTrainer with data and hyperparameters."""
        seed = getattr(params, "random_seed", None)
        _set_torch_seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = params
        self.epochs = params.epochs
        self.node_x = torch.as_tensor(node_x, dtype=torch.float32, device=self.device)

        self.edge_index = graph_dict["edge_index"].to(self.device)
        edge_attr = graph_dict.get("edge_attr")
        if edge_attr is None:
            edge_weight = graph_dict.get("edge_weight")
            edge_attr = edge_weight.view(-1, 1) if edge_weight is not None else None
        self.edge_attr = edge_attr.to(self.device) if edge_attr is not None else None

        self.label = label
        self.num_classes = 1

        if self.label is not None:
            self.label = torch.as_tensor(self.label, dtype=torch.long, device=self.device)
            self.num_classes = int(len(torch.unique(self.label)))

        self.model = GATModel(
            self.params.feat_cell,
            self.params,
            self.num_classes,
            edge_attr_dim=(self.edge_attr.shape[1] if self.edge_attr is not None else None),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params.gat_lr,
            weight_decay=_resolve_weight_decay(self.params),
        )

        self.anticollapse_enabled = bool(getattr(self.params, "anticollapse_enabled", True))
        self.anticollapse_var_weight = float(getattr(self.params, "anticollapse_var_weight", 1.0))
        self.anticollapse_cov_weight = float(getattr(self.params, "anticollapse_cov_weight", 0.01))
        self.anticollapse_var_target = float(getattr(self.params, "anticollapse_var_target", 1.0))
        self.anticollapse_start_epoch = int(getattr(self.params, "anticollapse_start_epoch", 20))

        self.training_summary = {
            "epochs_requested": int(self.epochs),
            "epochs_run": 0,
            "converged": False,
            "final_loss": None,
            "best_loss": None,
            "final_loss_rec": None,
            "final_loss_label": None,
            "final_loss_var": None,
            "final_loss_cov": None,
            "anti_collapse": {
                "enabled": self.anticollapse_enabled,
                "var_weight": self.anticollapse_var_weight,
                "cov_weight": self.anticollapse_cov_weight,
                "var_target": self.anticollapse_var_target,
                "start_epoch": self.anticollapse_start_epoch,
            },
        }

    def run_train(self):
        """Train the model."""
        self.model.train()
        prev_loss = float("inf")
        best_loss = float("inf")

        logger.info("Start training GAT-AE...")
        pbar = tqdm(range(self.epochs), desc="GAT-AE model train", total=self.epochs)

        for epoch in range(self.epochs):
            start_time = time.time()
            self.optimizer.zero_grad()

            pred_label, de_feat, latent_z, mu, logvar = self.model(
                self.node_x,
                self.edge_index,
                edge_attr=self.edge_attr,
            )
            del latent_z, logvar

            loss_rec = reconstruction_loss(de_feat, self.node_x)
            if self.label is not None:
                loss_pre = label_loss(pred_label, self.label)
            else:
                loss_pre = torch.tensor(0.0, device=self.device)

            if self.anticollapse_enabled and epoch >= self.anticollapse_start_epoch:
                loss_var = variance_regularization_loss(mu, target_std=self.anticollapse_var_target)
                loss_cov = covariance_regularization_loss(mu)
            else:
                loss_var = torch.tensor(0.0, device=self.device)
                loss_cov = torch.tensor(0.0, device=self.device)

            loss = (
                self.params.rec_w * loss_rec
                + self.params.label_w * loss_pre
                + self.anticollapse_var_weight * loss_var
                + self.anticollapse_cov_weight * loss_cov
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            loss_value = float(loss.item())
            best_loss = min(best_loss, loss_value)
            self.training_summary["epochs_run"] = int(epoch + 1)
            self.training_summary["final_loss"] = loss_value
            self.training_summary["best_loss"] = float(best_loss)
            self.training_summary["final_loss_rec"] = float(loss_rec.item())
            self.training_summary["final_loss_label"] = float(loss_pre.item())
            self.training_summary["final_loss_var"] = float(loss_var.item())
            self.training_summary["final_loss_cov"] = float(loss_cov.item())

            batch_time = time.time() - start_time
            left_time = batch_time * (self.epochs - epoch - 1) / 60
            pbar.set_postfix(
                {
                    "left(min)": f"{left_time:.2f}",
                    "loss": f"{loss_value:.4f}",
                    "rec": f"{float(loss_rec.item()):.4f}",
                    "var": f"{float(loss_var.item()):.4f}",
                    "cov": f"{float(loss_cov.item()):.4f}",
                }
            )
            pbar.update(1)

            if abs(loss_value - prev_loss) <= self.params.convergence_threshold and epoch >= 200:
                self.training_summary["converged"] = True
                pbar.close()
                logger.info("Convergence reached. Training stopped.")
                break

            prev_loss = loss_value
        else:
            pbar.close()
            logger.info("Max epochs reached. Training stopped.")

    def get_latent(self):
        """Retrieve the latent representation from the model."""
        self.model.eval()
        with torch.no_grad():
            _, _, latent_z, _, _ = self.model(
                self.node_x,
                self.edge_index,
                edge_attr=self.edge_attr,
            )
        return latent_z.cpu().numpy()

    def get_training_summary(self):
        return dict(self.training_summary)
