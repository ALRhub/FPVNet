import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import einops
import math
from typing import Optional
from torch.nn import functional as F
import logging

logger = logging.getLogger(__name__)


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(
            self,
            n_embd: int,
            n_heads: int,
            attn_pdrop: float,
            resid_pdrop: float,
            block_size: int,
    ):
        super().__init__()
        assert n_embd % n_heads == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )
        self.n_head = n_heads

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
             )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class CausalSelfCrossAttention(nn.Module):
    def __init__(self, n_embd, cross_embed, n_heads, attn_pdrop, resid_pdrop, block_size):
        super().__init__()

        assert n_embd % n_heads == 0

        # Self-Attention Projections
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # Cross-Attention Projections
        self.cross_key = nn.Linear(cross_embed, n_embd)
        self.cross_query = nn.Linear(n_embd, n_embd)
        self.cross_value = nn.Linear(cross_embed, n_embd)

        # Regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # Output Projection
        self.proj = nn.Linear(n_embd, n_embd)

        # Causal mask for Self-Attention
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

        self.n_head = n_heads

    def forward(self, x, cross_input=None):
        B, T, C = x.size()

        # calculate query, key, values for self-attention
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v

        if cross_input is not None:
            # calculate query, key, values for cross-attention
            T_C = cross_input.size(1)
            k_cross = self.cross_key(cross_input).view(B, T_C, self.n_head, C // self.n_head).transpose(1, 2)
            v_cross = self.cross_value(cross_input).view(B, T_C, self.n_head, C // self.n_head).transpose(1, 2)

            q_cross = self.cross_query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            # cross-attention
            att_cross = (q_cross @ k_cross.transpose(-2, -1)) * (1.0 / math.sqrt(k_cross.size(-1)))
            att_cross = F.softmax(att_cross, dim=-1)
            att_cross = self.attn_drop(att_cross)
            y_cross = att_cross @ v_cross

            # combine self-attention and cross-attention
            y = y + y_cross  # or any other combination strategy

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))

        return y


class DecoderBlock(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
            self,
            n_embd: int,
            cross_embd: int,
            n_heads: int,
            attn_pdrop: float,
            resid_pdrop: float,
            block_size: int,

    ):
        super().__init__()
        self.ln1 = LayerNorm(n_embd, bias=False)
        self.ln2 = LayerNorm(n_embd, bias=False)
        self.attn = CausalSelfCrossAttention(
            n_embd,
            cross_embd,
            n_heads,
            attn_pdrop,
            resid_pdrop,
            block_size,
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, cond=None):
        x = x + self.attn(self.ln1(x), cross_input=cond)
        x = x + self.mlp(self.ln2(x))
        return x


class EncoderBlock(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
            self,
            n_embd: int,
            n_heads: int,
            attn_pdrop: float,
            resid_pdrop: float,
            block_size: int,

    ):
        super().__init__()
        self.ln1 = LayerNorm(n_embd, bias=False)
        self.ln2 = LayerNorm(n_embd, bias=False)
        self.attn = SelfAttention(
            n_embd,
            n_heads,
            attn_pdrop,
            resid_pdrop,
            block_size,
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerEncoder(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            n_heads: int,
            attn_pdrop: float,
            resid_pdrop: float,
            n_layers: int,
            block_size: int,
            bias: bool = False,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[EncoderBlock(
                embed_dim,
                n_heads,
                attn_pdrop,
                resid_pdrop,
                block_size,
            )
                for _ in range(n_layers)]
        )
        self.ln = LayerNorm(embed_dim, bias)

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        x = self.ln(x)
        return x


class TransformerDecoder(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            cross_embed: int,
            n_heads: int,
            attn_pdrop: float,
            resid_pdrop: float,
            n_layers: int,
            block_size: int,
            bias: bool = False,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[DecoderBlock(
                embed_dim,
                cross_embed,
                n_heads,
                attn_pdrop,
                resid_pdrop,
                block_size,
            )
                for _ in range(n_layers)]
        )
        self.ln = LayerNorm(embed_dim, bias)

    def forward(self, x, cond=None):
        for layer in self.blocks:
            x = layer(x, cond=cond)
        x = self.ln(x)
        return x


class Enc_only(nn.Module):
    """Diffusion model with transformer architecture for state, goal, time and action tokens,
    with a context size of block_size"""

    def __init__(
            self,
            encoder: DictConfig,
            state_dim: int,
            action_dim: int,
            device: str,
            goal_conditioned: bool,
            embed_dim: int,
            embed_pdrob: float,
            goal_seq_len: int,
            obs_seq_len: int,
            action_seq_len: int,
            goal_drop: float = 0.1,
            linear_output: bool = False,
    ):
        super().__init__()

        self.encoder = hydra.utils.instantiate(encoder)

        self.device = device
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            goal_seq_len = 0
        # input embedding stem
        # first we need to define the maximum block size
        # it consists of the goal sequence length plus 1 for the sigma embedding and 2 the obs seq len
        block_size = goal_seq_len + action_seq_len + obs_seq_len + 1
        # the seq_size is a little different since we have state action pairs for every timestep
        seq_size = goal_seq_len + obs_seq_len + action_seq_len

        self.tok_emb = nn.Linear(state_dim, embed_dim)
        self.tok_emb.to(self.device)

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        # needed for calssifier guidance learning
        self.cond_mask_prob = goal_drop

        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim

        self.block_size = block_size
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

        # get an action embedding
        self.query_embed = nn.Embedding(action_seq_len, embed_dim)

        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, self.action_dim)
            )
        # self.action_pred = nn.Linear(embed_dim, action_dim) # less parameters, worse reward
        self.action_pred.to(self.device)

        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
                len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    # x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: torch.Tensor
    # def forward(self, x, t, state, goal):
    def forward(
            self,
            states,
            goals=None,
            uncond: Optional[bool] = False,
            keep_last_actions: Optional[bool] = False
    ):

        if len(states.size()) != 3:
            states = states.unsqueeze(0)

        b, t, dim = states.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        if self.goal_conditioned:

            if self.training:
                goals = self.mask_cond(goals)
            # we want to use unconditional sampling during clasisfier free guidance
            if uncond:
                goals = torch.zeros_like(goals).to(self.device)

            goal_embed = self.tok_emb(goals)

        # embed them into linear representations for the transformer
        state_embed = self.tok_emb(states)

        position_embeddings = self.pos_emb[:, :(t + self.goal_seq_len + self.action_seq_len - 1), :]
        # note, that the goal states are at the beginning of the sequence since they are available
        # for all states s_1, ..., s_t otherwise the masking would not make sense
        if self.goal_conditioned:
            goal_x = self.drop(goal_embed + position_embeddings[:, :self.goal_seq_len, :])

        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len + t), :])

        # state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:, :])
        # # the action get the same position embedding as the related states
        # action_x = self.drop(action_embed + position_embeddings[:, self.goal_seq_len:, :])

        # decode the action sequence with cross attention over the encoder output
        action_seq = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)

        input_seq = torch.cat([state_x, action_seq], dim=1)

        # encode the state, goal and latent z into the hidden dim
        encoder_output = self.encoder(input_seq)

        pred_actions = self.action_pred(encoder_output[:, t:, :])

        return pred_actions

    def get_params(self):
        return self.parameters()


class EncDec(nn.Module):
    """Diffusion model with transformer architecture for state, goal, time and action tokens,
    with a context size of block_size"""

    def __init__(
            self,
            encoder: DictConfig,
            decoder: DictConfig,
            state_dim: int,
            goal_dim: int,
            action_dim: int,
            obs_tokens: int,
            device: str,
            embed_dim: int,
            embed_pdrob: float,
            goal_seq_len: int,
            obs_seq_len: int,
            action_seq_len: int,
            goal_drop: float = 0.1,
            linear_output: bool = False,
    ):
        super().__init__()

        self.encoder = hydra.utils.instantiate(encoder)
        self.decoder = hydra.utils.instantiate(decoder)

        self.device = device

        # input embedding stem
        # first we need to define the maximum block size
        # it consists of the goal sequence length plus 1 for the sigma embedding and 2 the obs seq len
        block_size = goal_seq_len + action_seq_len + obs_seq_len + 1
        # the seq_size is a little different since we have state action pairs for every timestep
        seq_size = obs_tokens + action_seq_len

        self.tok_emb = nn.Linear(state_dim, embed_dim)
        self.tok_emb.to(self.device)

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        self.goal_emb = nn.Linear(goal_dim, embed_dim)

        # needed for calssifier guidance learning
        self.cond_mask_prob = goal_drop

        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim

        self.block_size = block_size
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

        self.obs_tokens = obs_tokens

        # get an action embedding
        self.action_token = nn.Embedding(action_seq_len, embed_dim)

        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, self.action_dim)
            )
        # self.action_pred = nn.Linear(embed_dim, action_dim) # less parameters, worse reward
        self.action_pred.to(self.device)

        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
            self,
            states,
            goals=None,
            return_encoder_embedding=False
    ):  
        # goal is language embedding
        # states is image embedding

        if len(states.size()) != 3:
            states = states.unsqueeze(0)

        b, t, dim = states.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # with torch.no_grad():

        # embed them into linear representations for the transformer
        state_embed = self.tok_emb(states)
        goal_embed = self.goal_emb(goals)

        position_embeddings = self.pos_emb

        goal_x = self.drop(goal_embed + position_embeddings[:, :self.goal_seq_len, :])
        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len + t), :])

        # encode the state, goal and latent z into the hidden dim
        encoder_output = self.encoder(torch.cat([goal_x, state_x], dim=1))

        if return_encoder_embedding:
            return encoder_output

        # decode the action sequence with cross attention over the encoder output
        action_seq = self.action_token.weight.unsqueeze(0).repeat(b, 1, 1)

        decoder_output = self.decoder(action_seq, encoder_output)

        pred_actions = self.action_pred(decoder_output)

        return pred_actions

    def get_params(self):
        return self.parameters()
