import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _checkpoint  # private: this module is star-imported
    
class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, scandots_output_dim, output_activation=None, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images.unsqueeze(1))
        latent = self.output_activation(images_compressed)
        return latent
    

class RecurrentDepthBackbone(nn.Module):
    def __init__(self, base_backbone, depth_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        num_prop = depth_cfg['num_prop']
        if num_prop == None:
            self.combination_mlp = nn.Sequential(
                                    nn.Linear(32 + 53, 128),
                                    activation,
                                    nn.Linear(128, 32)
                                )
        else:
            self.combination_mlp = nn.Sequential(
                                        nn.Linear(32 + num_prop, 128),
                                        activation,
                                        nn.Linear(128, 32)
                                    )
        self.recurrent_size = 512
            
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        self.output_mlp = nn.Sequential(
                                nn.Linear(512, 32+2),
                                last_activation
                            )
        self.hidden_states = torch.zeros(1, 0, 512)
        self.rnn.flatten_parameters()

    def forward(self, depth_image, proprioception):
        if self.hidden_states.shape[1] == 0:
            # On the first forward pass, initialize hidden states to proper batch size
            self.hidden_states = torch.zeros(1, depth_image.shape[0], self.recurrent_size).to(depth_image.device)

        depth_image = self.base_backbone(depth_image)
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        depth_latent = self.output_mlp(depth_latent.squeeze(1))
        return depth_latent

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()


class RecurrentMultiDepthBackbone(nn.Module):
    """K camera images -> one 32-d depth latent: the omni depth student's encoder (Task 4 M1b, DESIGN B4).

    Used ONLY when the runner cfg selects it (``CrabHexOmniRslRlDepthEncoderCfg``); the forward
    ``RecurrentDepthBackbone`` above is untouched. Per encode:

        (N, K, 58, 87) --shared conv (``base_backbone``, one for all K)--> (N, 32 K)
        cat proprio (N, num_prop) -> combination_mlp(32 K + num_prop -> 128 -> 32) -> GRU(32 -> 512)
        -> output_mlp(512 -> latent_dim, tanh)

    No yaw head: the omni student never writes the facing command (obs cols 6-7), so the output is the latent
    alone (``latent_dim`` = the actor's scan-latent width, 32). The GRU hidden state is explicit in
    :meth:`step` (the export graphs use it) and held in ``hidden_states`` by :meth:`forward`;
    :meth:`reset` zeroes it on the done rows only.

    ``grad_checkpoint`` (cfg, default True for the omni student): when gradients are being recorded, the shared
    conv runs under ``torch.utils.checkpoint`` -- its activations are recomputed in the backward pass instead of
    stored. ``learn_vision`` back-propagates through every encode of a 120-step rollout (24 encodes), and the
    stored conv activations are ~1.26 MB per image: ~23 GB at 192 envs x K = 4 (vs ~6 GB at the forward K = 1).
    The recompute gives the same gradient; inference (``no_grad`` / ``inference_mode``) takes the plain path.
    """

    def __init__(self, base_backbone, depth_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        self.num_cameras = int(depth_cfg["num_cameras"])
        self.num_prop = int(depth_cfg["num_prop"])
        self.latent_dim = int(depth_cfg.get("latent_dim", 32))
        self.grad_checkpoint = bool(depth_cfg.get("grad_checkpoint", True))
        feat = 32
        self.combination_mlp = nn.Sequential(
            nn.Linear(feat * self.num_cameras + self.num_prop, 128),
            activation,
            nn.Linear(128, 32),
        )
        self.recurrent_size = 512
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        self.output_mlp = nn.Sequential(
            nn.Linear(512, self.latent_dim),
            last_activation,
        )
        self.hidden_states = torch.zeros(1, 0, 512)
        self.rnn.flatten_parameters()

    def image_features(self, depth_images: torch.Tensor) -> torch.Tensor:
        """``(N, K, H, W)`` -> ``(N, 32 K)``: every camera through the ONE shared conv, camera-major per row."""
        if depth_images.dim() != 4 or depth_images.shape[1] != self.num_cameras:
            raise ValueError(f"expected (N, {self.num_cameras}, H, W) depth images, got {tuple(depth_images.shape)}")
        n, k, h, w = depth_images.shape
        flat = depth_images.reshape(n * k, h, w)
        if self.grad_checkpoint and torch.is_grad_enabled() and self.training:
            feats = _checkpoint(self.base_backbone, flat, use_reentrant=False)
        else:
            feats = self.base_backbone(flat)
        return feats.reshape(n, k * feats.shape[-1])

    def step(self, depth_images, proprioception, hidden_states):
        """One stateless encode: ``(latent (N, latent_dim), hidden_out (1, N, 512))``."""
        feats = self.image_features(depth_images)
        latent = self.combination_mlp(torch.cat((feats, proprioception), dim=-1))
        latent, hidden_out = self.rnn(latent[:, None, :], hidden_states)
        return self.output_mlp(latent.squeeze(1)), hidden_out

    def forward(self, depth_images, proprioception):
        n = depth_images.shape[0]
        if self.hidden_states.dim() != 3 or self.hidden_states.shape[1] != n:
            self.hidden_states = torch.zeros(1, n, self.recurrent_size, device=depth_images.device)
        latent, self.hidden_states = self.step(depth_images, proprioception, self.hidden_states)
        return latent

    def reset(self, dones=None):
        """Zero the GRU state of the done rows (all rows when ``dones`` is None); out of place (autograd-safe)."""
        h = self.hidden_states
        if dones is None or h.dim() != 3 or h.shape[1] == 0:
            self.hidden_states = torch.zeros(1, 0, self.recurrent_size)
            return
        done = torch.as_tensor(dones, device=h.device).reshape(-1).bool()
        if done.numel() != h.shape[1]:
            raise ValueError(f"reset: {done.numel()} done flags for a GRU state of batch {h.shape[1]}")
        self.hidden_states = torch.where(done[None, :, None], torch.zeros_like(h), h)

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()
