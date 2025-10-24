import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# --- 1. Model Definition (Must match the saved model structure) ---
class FastGridInterpolator(nn.Module):
    """
    A deployable model that performs hardware-accelerated bilinear interpolation
    and includes an optional, embedded safety mask to check for out-of-bounds queries.
    """
    def __init__(self, grid_data, grid_axes, safe_mask=None):
        super().__init__()
        
        if grid_data.ndim > 3:
            original_shape = grid_data.shape
            num_channels = np.prod(original_shape[2:])
            grid_data = grid_data.reshape(original_shape[0], original_shape[1], num_channels)

        grid_tensor = torch.from_numpy(grid_data.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('grid', grid_tensor)

        x_coords, y_coords = grid_axes
        self.register_buffer('x_min', torch.tensor(float(x_coords.min()), dtype=torch.float32))
        self.register_buffer('x_max', torch.tensor(float(x_coords.max()), dtype=torch.float32))
        self.register_buffer('y_min', torch.tensor(float(y_coords.min()), dtype=torch.float32))
        self.register_buffer('y_max', torch.tensor(float(y_coords.max()), dtype=torch.float32))

        if safe_mask is not None:
            assert safe_mask.shape == grid_data.shape[:2]
            assert safe_mask.dtype == bool
            mask_tensor = torch.from_numpy(safe_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            self.register_buffer('safe_mask_grid', mask_tensor)
        else:
            self.register_buffer('safe_mask_grid', None)

    def safe_check(self, coords):
        device = self.grid.device
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords.astype(np.float32)).to(device)
        if self.safe_mask_grid is not None:
            norm_x = 2 * (coords[:, 0] - self.x_min) / (self.x_max - self.x_min) - 1
            norm_y = 2 * (coords[:, 1] - self.y_min) / (self.y_max - self.y_min) - 1
            
            normalized_coords = torch.stack([norm_x, norm_y], dim=1).unsqueeze(0).unsqueeze(0)

            mask_values = F.grid_sample(self.safe_mask_grid, normalized_coords, mode='bilinear', padding_mode='border', align_corners=True).squeeze()
            return mask_values
        else:
            return torch.ones(coords.shape[0], device=device)

    def forward(self, coords, strict=False, warn=True):
        device = self.grid.device
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords.astype(np.float32)).to(device)

        norm_x = 2 * (coords[:, 0] - self.x_min) / (self.x_max - self.x_min) - 1
        norm_y = 2 * (coords[:, 1] - self.y_min) / (self.y_max - self.y_min) - 1
        
        normalized_coords = torch.stack([norm_x, norm_y], dim=1).unsqueeze(0).unsqueeze(0)

        if self.safe_mask_grid is not None:
            mask_values = F.grid_sample(self.safe_mask_grid, normalized_coords, mode='bilinear', padding_mode='border', align_corners=True).squeeze()
            out_of_bounds_indices = torch.where(mask_values < 0.999)[0]

            if out_of_bounds_indices.numel() > 0:
                if strict:
                    raise ValueError(f"{out_of_bounds_indices.numel()} query points are outside the safe operating mask.")
                elif warn:
                    print(f"Warning: {out_of_bounds_indices.numel()} of {coords.shape[0]} query points are outside the safe operating mask.")

        interpolated_values = F.grid_sample(
            self.grid, 
            normalized_coords, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )
        
        num_queries = coords.shape[0]
        num_channels = self.grid.shape[1]
        # Permute from (N, C, H_out, W_out) -> (N, H_out, W_out, C) and reshape
        return interpolated_values.permute(0, 2, 3, 1).reshape(num_queries, num_channels)

# --- 2. Deployment API ---
class AnkleAPI:
    """
    A high-level API for accessing the ankle mappings for either the left or right foot.
    Handles the mirroring logic internally. Side is specified at call time.
    """
    def __init__(self, model_dir="trained_models", device='cpu'):
        self.device = device
        self.models = {}
        
        print(f"--- Initializing AnkleAPI ---")
        
        for name in ['fk', 'ik', 'jt', 'jt_inv', 'g']:
            self.models[name] = self._load_model(name, model_dir)
            self.models[name].to(self.device)
            self.models[name].eval()

    def _load_model(self, mapping_name, model_dir):
        """Loads a single packaged model."""
        model_path = os.path.join(model_dir, f"{mapping_name}_final_lookup_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        state_dict = torch.load(model_path, map_location='cpu')
        
        grid_tensor = state_dict['grid']
        res_y, res_x = grid_tensor.shape[2], grid_tensor.shape[3]
        num_channels = grid_tensor.shape[1]
        
        placeholder_grid = np.zeros((res_y, res_x, num_channels))
        x_coords = np.linspace(state_dict['x_min'], state_dict['x_max'], res_x)
        y_coords = np.linspace(state_dict['y_min'], state_dict['y_max'], res_y)
        placeholder_axes = [x_coords, y_coords]

        safe_mask = None
        if 'safe_mask_grid' in state_dict and state_dict['safe_mask_grid'] is not None:
            safe_mask = state_dict['safe_mask_grid'].squeeze().numpy().astype(bool)

        model = FastGridInterpolator(placeholder_grid, placeholder_axes, safe_mask=safe_mask)
        model.load_state_dict(state_dict)
        return model

    def _predict(self, model_name, data_in, **kwargs):
        with torch.no_grad():
            return self.models[model_name](data_in, **kwargs).cpu().numpy()

    def get_fk(self, q_in, side='left', **kwargs):
        """
        Get forward kinematics.
        
        Args:
            q_in: Joint angles
            side: 'left' or 'right'
        """
        assert side in ['left', 'right'], "Side must be 'left' or 'right'"
        
        if side == 'left':
            return self._predict('fk', q_in, **kwargs)
        else:  # Right side
            q_l = -q_in
            pred = self._predict('fk', q_l, **kwargs)
            pred[:, 1] *= -1
            return pred

    def get_ik(self, ankle_in, side='left', **kwargs):
        """
        Get inverse kinematics.
        
        Args:
            ankle_in: Ankle position
            side: 'left' or 'right'
        """
        assert side in ['left', 'right'], "Side must be 'left' or 'right'"
        
        if side == 'left':
            return self._predict('ik', ankle_in, **kwargs)
        else:  # Right side
            ankle_l = ankle_in.copy()
            ankle_l[:, 1] *= -1
            return -self._predict('ik', ankle_l, **kwargs)

    def get_jt(self, q_in, side='left', **kwargs):
        """
        Get Jacobian transpose.
        
        Args:
            q_in: Joint angles
            side: 'left' or 'right'
        """
        assert side in ['left', 'right'], "Side must be 'left' or 'right'"
        
        if side == 'left':
            return self._predict('jt', q_in, **kwargs)
        else:  # Right side
            q_l = -q_in
            Jt_l = self._predict('jt', q_l, **kwargs).reshape(-1, 2, 2)
            Jt_r = Jt_l.copy()
            Jt_r[:, :, 0] *= -1  # Negate first column
            return Jt_r.reshape(-1, 4)

    def get_jt_inv(self, q_in, side='left', **kwargs):
        """
        Get inverse Jacobian transpose.
        
        Args:
            q_in: Joint angles
            side: 'left' or 'right'
        """
        assert side in ['left', 'right'], "Side must be 'left' or 'right'"
        
        if side == 'left':
            return self._predict('jt_inv', q_in, **kwargs)
        else:  # Right side
            q_l = -q_in
            Jt_inv_l = self._predict('jt_inv', q_l, **kwargs).reshape(-1, 2, 2)
            Jt_inv_r = Jt_inv_l.copy()
            Jt_inv_r[:, 0, :] *= -1  # Negate first row
            return Jt_inv_r.reshape(-1, 4)

    def get_g(self, q_in, side='left', **kwargs):
        """
        Get gravity compensation.
        
        Args:
            q_in: Joint angles
            side: 'left' or 'right'
        """
        assert side in ['left', 'right'], "Side must be 'left' or 'right'"
        
        if side == 'left':
            return self._predict('g', q_in, **kwargs)
        else:  # Right side
            q_l = -q_in
            return -self._predict('g', q_l, **kwargs)
