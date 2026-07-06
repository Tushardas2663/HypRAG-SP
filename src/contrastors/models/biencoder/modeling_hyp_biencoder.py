

from contextlib import nullcontext
import os
from functools import partial
import math
import numpy as np
import torch
import torch.nn as nn
#import hyplib.nn as hnn
import torch.nn.functional as F
#from flash_attn.bert_padding import pad_input, unpad_input
# from flash_attn.ops.rms_norm import RMSNorm
from torch.nn import RMSNorm
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

from layers.activations import quick_gelu
from layers.attention import FlashAttentionPooling
from layers.block import Block
from layers.mlp import MLP, GatedMLP
from models.encoder import ALBertModel, ELBertModel, CONFIG_CONVERTER_REGISTRY, ALBertConfig
from models.encoder import convert_base_model_config_to_elbert_config


def update_model_config(model_config, config):
    model_config.model_type = 'albert'
    for key, value in config.__dict__.items():
        if key.startswith("_"):
            continue
        if hasattr(model_config, key) and model_config.__dict__[key] != value and value is not None:
            print(f"Setting {key} to {value}")
            setattr(model_config, key, value)
    return model_config


class LogitScale(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(config.logit_scale), requires_grad=config.trainable_logit_scale
        )

    def forward(self, x):
        return x * self.logit_scale.exp()

    def __repr__(self):
        return f"LogitScale(logit_scale={self.logit_scale.exp().item()}, trainable={self.logit_scale.requires_grad})"


class ClsSelector(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, input_ids, attention_mask):
        return hidden_states[:, 0]


class LastTokenPooling(nn.Module):
    def __init__(self, eos_token_id):
        super().__init__()
        self.eos_token_id = eos_token_id

    def forward(self, hidden_states, input_ids, attention_mask):
        # get the embedding corresponding to the first eos token
        # we don't substract 1 because the eos token is already included in the input_ids and attention_mask
        # and we want to get the embedding of the last token
        sequence_lengths = attention_mask.sum(-1) - 1
        selected_tokens = input_ids[torch.arange(input_ids.shape[0]), sequence_lengths]

        if not torch.all(selected_tokens == self.eos_token_id):
            raise ValueError(
                f"The last token of the input_ids is not the eos token: {selected_tokens}\n{input_ids}\n{sequence_lengths}"
            )
        prev_token = input_ids[torch.arange(input_ids.shape[0]), sequence_lengths - 1]
        if torch.any(prev_token == self.eos_token_id):
            raise ValueError(
                f"The second to last token of the input_ids is the eos token: {selected_tokens}\n{input_ids}\n{sequence_lengths}"
            )

        embs = hidden_states[torch.arange(hidden_states.shape[0]), sequence_lengths]

        return embs


class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, input_ids, attention_mask):
        if attention_mask is None:
            # for vit, no attention mask is provided
            return torch.mean(hidden_states, dim=1)

        s = torch.sum(hidden_states * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(axis=1, keepdim=True).float()
        return s / d


class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, config):
        # adapted from https://github.com/google-research/big_vision/blob/474dd2ebde37268db4ea44decef14c7c1f6a0258/big_vision/models/vit.py#L158
        super().__init__()
        self.attn = FlashAttentionPooling(config)
        activation = (
            F.sigmoid
            if config.activation_function == "glu"
            else (
                F.silu
                if config.activation_function == "swiglu"
                else (quick_gelu if config.activation_function == "quick_gelu" else F.gelu)
            )
        )
        if config.activation_function in ["glu", "swiglu"]:
            self.mlp = GatedMLP(
                config.n_embd,
                hidden_features=config.n_inner,
                bias1=config.mlp_fc1_bias,
                bias2=config.mlp_fc2_bias,
                activation=activation,
                fused_bias_fc=config.fused_bias_fc,
            )
        else:
            self.mlp = MLP(
                config.n_embd,
                hidden_features=config.n_inner,
                bias1=config.mlp_fc1_bias,
                bias2=config.mlp_fc2_bias,
                activation=activation,
                fused_bias_fc=config.fused_bias_fc,
            )
        norm_cls = partial(
            nn.LayerNorm if not config.use_rms_norm else RMSNorm,
            eps=config.layer_norm_epsilon,
        )
        self.norm1 = norm_cls(config.n_embd)

    def forward(self, hidden_states, input_ids, attention_mask):
        if attention_mask is not None:
            hidden_states, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(hidden_states, attention_mask)
        else:
            indices = None
            cu_seqlens = None
            max_seqlen_in_batch = None

        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            is_padded_inputs=True,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_k=max_seqlen_in_batch,
        )

        normed = self.norm1(attn_outputs)
        hidden_states = hidden_states + self.mlp(normed)
        if attention_mask is not None:
            hidden_states = pad_input(hidden_states, indices, cu_seqlens, max_seqlen_in_batch)

        return hidden_states[:, 0]

class MonotonicSpline(nn.Module):
    def __init__(self, grid_size=10, spline_order=3, grid_range=[-15.0, 15.0]):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # 1. CREATE A CLAMPED KNOT VECTOR
        h = (grid_range[1] - grid_range[0]) / grid_size #h is step size
        inner_knots = torch.linspace(grid_range[0], grid_range[1], grid_size + 1)#1d tensor of size grid_size+1
        
        left_pad = torch.full((spline_order,), grid_range[0]) #(spline_order,) defines shape of output tensor, fill with grid_range[0]
        right_pad = torch.full((spline_order,), grid_range[1])
        
        grid = torch.cat([left_pad, inner_knots, right_pad])  #first and last order+1 knots same value
        self.register_buffer("grid", grid.unsqueeze(0))  #unsqueeze adds a single dimension at 0th position here, move to gpu
        
        self.num_coeffs = grid_size + spline_order  #13 here, formulaically
        
        # 2. IDENTITY INITIALIZATION (y = x)
        self.raw_weights = nn.Parameter(torch.zeros(self.num_coeffs)) #model doesn't train the control points but the raw weights
        #self.raw_weights = nn.Parameter(torch.randn(grid_size + spline_order))
        
        with torch.no_grad():                         #le init
            self.raw_weights[0] = grid_range[0]
            target_step = h - 1e-4
            w_init = math.log(math.exp(target_step) - 1.0)
            self.raw_weights[1:] = w_init

        
    def get_monotonic_coeffs(self):
        start = self.raw_weights[0:1]
        steps = F.softplus(self.raw_weights[1:]) + 1e-4
        return torch.cat([start, start + torch.cumsum(steps, dim=0)]) #build the monotonic control points dynamically, raw weights not control points, rather distance between them that is trained

    def forward(self, x):
        original_shape = x.shape
        x_flat = x.view(-1)
        
        # The valid domain is exactly the grid range we specified
        valid_min = self.grid[0, self.spline_order].item()       #-15, got grid from buffer
        valid_max = self.grid[0, -self.spline_order - 1].item() 
        
        x_clamped = torch.clamp(x_flat, min=valid_min, max=valid_max) #clamp input temporarily to -15,15
        x_unsq = x_clamped.unsqueeze(-1) #for tensor broadcasting and comparison below
        
        # Standard Cox-de Boor evaluation base case 0, self.grid[:,:-1] is the left bound and 1: gives the right bound for each interval for each eigen value
        bases = ((x_unsq >= self.grid[:, :-1]) & (x_unsq < self.grid[:, 1:])).to(x_flat.dtype) #base is a boolean array True when eigen value in range
        
        # Handle the edge case where x == valid_max
        #bases[..., -1] = torch.where(x_unsq[..., 0] == valid_max, 
         #                            torch.ones_like(bases[..., -1]), 
          #                           bases[..., -1])

        for k in range(1, self.spline_order + 1):   #recursion evaluation
            left_den = self.grid[:, k:-1] - self.grid[:, :-(k+1)]  #left t denominator for all basis i intervals at once
            right_den = self.grid[:, k+1:] - self.grid[:, 1:-k]
            
            # THE FIX: Replace 0 with 1 in the denominator BEFORE division to prevent Inf/NaN in Autograd
            safe_left_den = torch.where(left_den == 0, torch.ones_like(left_den), left_den) #if denom is 0 anywhere replace it with 1
            safe_right_den = torch.where(right_den == 0, torch.ones_like(right_den), right_den)
            
            left_term = torch.where(left_den > 0, (x_unsq - self.grid[:, :-(k+1)]) / safe_left_den, torch.zeros_like(left_den)) * bases[..., :-1]  #multiply  bases of previous step as per formua
            right_term = torch.where(right_den > 0, (self.grid[:, k+1:] - x_unsq) / safe_right_den, torch.zeros_like(right_den)) * bases[..., 1:]
            
            bases = left_term + right_term 
            
        c = self.get_monotonic_coeffs()
        out = (bases * c).sum(dim=-1) #bases are now the b-splines, mul c and add for fx.
        out = torch.where(x_clamped == valid_max, c[-1].detach(), out)
        out = torch.where(x_clamped == valid_min, c[0].detach(), out)
        # 3. EXACT ANALYTICAL LINEAR EXTRAPOLATION
        left_mask = (x_flat < valid_min).float()
        right_mask = (x_flat > valid_max).float()
        
        degree = self.spline_order
        h_left = self.grid[0, degree + 1] - self.grid[0, degree]
        h_right = self.grid[0, -degree - 1] - self.grid[0, -degree - 2]
        
        left_slope = (degree / h_left) * (c[1] - c[0])
        right_slope = (degree / h_right) * (c[-1] - c[-2])
        
        out = out + left_mask * left_slope * (x_flat - valid_min)
        out = out + right_mask * right_slope * (x_flat - valid_max)

        return out.view(original_shape)


class LightweightSplineManifold:
    def __init__(self, spline_net):
        self.spline_net = spline_net
        self.name = 'spline'

    def pairwise_spline_dist(self, x, y):
        # x shape: (N, D), y shape: (M, D)
        # For your first pass, use a straight-line Riemann sum approximation
        # to avoid the complex Geodesic ODE solver during initial training.
        
        N, M = x.size(0), y.size(0)
        x_exp = x.unsqueeze(1).expand(N, M, -1)
        y_exp = y.unsqueeze(0).expand(N, M, -1)
        
        # Euclidean distance
        euc_dist = torch.norm(x_exp - y_exp, dim=-1)
        
        # Midpoint approximation for the integral: \lambda((x+y)/2) * ||x-y||
        # (You can upgrade this to a 5-point Gauss-Legendre quadrature later)
        midpoints = (x_exp + y_exp) / 2.0
        mid_radii = torch.norm(midpoints, dim=-1)
        
        # Query your spline for the conformal factor at the midpoints
        conformal_factors = self.spline_net(mid_radii)
        
        return euc_dist * conformal_factors

class SplineProjection(nn.Module):
    def __init__(self, in_features, out_features, spline_net):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.spline_net = spline_net
        
    def forward(self, x):
        # 1. Euclidean transformation
        v = self.linear(x)
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp_min(1e-6)
        
        # 2. The exp0 map proxy (Mapping tangent length to radial depth)
        # If you write the L^{-1} root finder, replace tanh(v_norm) with it here.
        r_mapped = torch.tanh(v_norm) 
        
        # 3. Project to the unit ball
        return r_mapped * (v / v_norm)
class HypBiEncoder(PreTrainedModel):
    _supports_flash_attn_2 = True
    _supports_flex_attn = True

    def __init__(self, config):
        super().__init__(config)

        if config.use_fused_kernels:
            print(f"Initializing {config.model_name}, pretrained={config.pretrained}")
            # set default to true for backward compatibility with old models?
            if "elbert" in config.model_type:
                model_config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)

                if model_config.model_type == "elbert":
                    if config.pretrained:
                        state_dict = torch.load(os.path.join(config.model_name, 'pytorch_model.bin'), map_location="cpu")
                        # trim state_dict if 'trunk.' prefix exists
                        if any(key.startswith("trunk.") for key in state_dict.keys()):
                            state_dict = {key[len("trunk.") :]: value for key, value in state_dict.items()}
                        self.trunk = ELBertModel.from_pretrained(
                            config.model_name,
                            add_pooling_layer=True,
                            config=model_config,
                        )
                        self.trunk.load_state_dict(state_dict, strict=False)
                    else:
                        self.trunk = ELBertModel(
                            config=model_config,
                            add_pooling_layer=True,
                        )
                else:
                    model_config = convert_base_model_config_to_elbert_config(model_config, config)
                    self.trunk = ELBertModel(
                        config=model_config,
                        add_pooling_layer=True,
                        base_pretrained=config.pretrained,
                    )
            elif "albert" in config.model_type:
                model_config = ALBertConfig.from_pretrained(config.model_name, trust_remote_code=True)
                config.attn_implementation = "flash_attention_2"
                model_config = update_model_config(model_config, config)
                
                self.trunk = ALBertModel(
                    config=model_config,
                    add_pooling_layer=True,
                )
                if config.pretrained:
                    print(f"Loading weights from {config.model_name}")

                    state = torch.load(
                        os.path.join(config.model_name, "pytorch_model.bin"),
                        map_location="cpu",
                    )

                    state_dict = state.get("state_dict", state)

                    trunk_state_dict = {
                        k[len("trunk."):]: v
                        for k, v in state_dict.items()
                        if k.startswith("trunk.")
                    }
                    missing, unexpected = self.trunk.load_state_dict(
                        trunk_state_dict,
                        strict=False,
                        assign=True,
                    )
        else:
            #raise ValueError(f"Model type {config.model_type} not supported")
            print(f"Loading standard HF model: {config.model_name}")
            self.trunk = AutoModel.from_pretrained(config.model_name, trust_remote_code=True)
            # Standard models don't have a hyperbolic manifold by default
            self.trunk.manifold = None

        if config.freeze:
            self.trunk.eval()
            for param in self.trunk.parameters():
                param.requires_grad = False

            self.frozen_trunk = True
        else:
            self.frozen_trunk = False

        if config.gradient_checkpointing:
            self.trunk.gradient_checkpointing_enable({"use_reentrant": False})

        if config.projection_dim:
            # Initialize your custom spline (assuming you paste MonotonicSpline in this file)
            self.spline_net = MonotonicSpline(grid_size=10, spline_order=3, grid_range=[0.0, 1.0])
            
            # Use your new projection
            self.proj = SplineProjection(self.trunk.config.hidden_size, config.projection_dim, self.spline_net)
            
            # Initialize the ferry manifold
            self.spline_manifold = LightweightSplineManifold(self.spline_net)
        else:
            self.proj = nn.Identity()
            self.spline_manifold = self.trunk.manifold # Fallback

    @property
    def manifold(self):
        # The trainer will grab this and pass it to hyperbolic_loss.py
        if hasattr(self, 'spline_manifold'):
            return self.spline_manifold
        return self.trunk.manifold

    def save_pretrained(self, save_directory, **kwargs):
        self.trunk.save_pretrained(save_directory, **kwargs)

    def load_pretrained(self, pretrained_model_name_or_path, **kwargs):
        # print("Loading using load_pretrained")
        # config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        # attn_implementation = self.trunk.config.attn_implementation
        # self.trunk = self.trunk.from_pretrained(
        #     pretrained_model_name_or_path,
        #     config=config,
        #     **kwargs
        # )
        # self.trunk.config.attn_implementation = attn_implementation
        # return self
        weights_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        self.trunk.load_state_dict(torch.load(weights_path), **kwargs)
    def forward(self, input_ids, attention_mask=None, return_cone_info=False, **kwargs):
        context = torch.no_grad if self.frozen_trunk else nullcontext
        with context():
            trunk_output = self.trunk(input_ids, attention_mask=attention_mask, **kwargs)

        #manifold = self.trunk.manifold
        manifold = getattr(self.trunk, 'manifold', None)

        embedding = trunk_output.pooler_output
        embedding = self.proj(embedding)

        if return_cone_info:
            hidden_states = trunk_output.hidden_states
            # randomly select a substring of the hidden states and then run pooling on it
            min_length = int(0.6 * hidden_states.shape[1])
            length = torch.randint(low=min_length, high=hidden_states.shape[1] + 1, size=(1,)).item()
            start_idx = torch.randint(low=0, high=hidden_states.shape[1] - length + 1, size=(1,)).item()
            sub_hidden_states = hidden_states[:, start_idx : start_idx + length, :]
            sub_pooled_output = self.trunk.pooler(sub_hidden_states)
            sampled_embedding = self.proj(sub_pooled_output)
        else:
            sampled_embedding = None

        info = trunk_output.info

        return {
            "embedding": embedding,
            "manifold": manifold,
            "sub_embedding": sampled_embedding,
            "info": info,
        }