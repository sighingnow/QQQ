import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaMLP,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaPreTrainedModel,
    LlamaModel,
    LlamaForCausalLM,
    LlamaRotaryEmbedding,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RMSNorm,
    Qwen2MLP,
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2PreTrainedModel,
    Qwen2Model,
    Qwen2ForCausalLM,
    Qwen2RotaryEmbedding,
)
from transformers.models.qwen2 import Qwen2Config
from transformers.activations import ACT2FN
from typing import Dict, Optional
from ..qlinear import QuantLinear
from ..gptq import *
from ..quant import *
from QQQ.utils import find_layers
from transformers.utils import logging

logger = logging.get_logger(__name__)



@torch.no_grad()
def gptq_qwen2_func(model, dataloader, dev, args, force_to_cpu=False):
    print("Starting GPTQ quantization ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = []
    # inps = torch.zeros(
    #     (1, args.seqlen, model.config.hidden_size),
    #     dtype=dtype,
    #     device=dev,
    # )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            # inps[cache["i"]] = inp
            inps.append(inp.cpu())
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    if force_to_cpu:
        layers[0] = layers[0].cpu()
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
        torch.cuda.empty_cache()

    # outs = torch.zeros_like(inps)
    outs = [None] * len(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    quantizers = {}
    for i, layer in enumerate(layers):
        if layer.input_layernorm.weight.device == torch.device("cpu"):
            layer = layer.to(dev)
        cur_device = layer.input_layernorm.weight.device
        # inps.to(cur_device)
        # outs.to(cur_device)
        attention_mask.to(cur_device)
        position_ids.to(cur_device)
        
        full = find_layers(layer)
        sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits,
                    perchannel=True,
                    sym=args.sym,
                    mse=args.mse,
                    groupsize=args.groupsize,
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(
                    inps[j].to(cur_device),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0].cpu()
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print("Quantizing ...")
                scale, zero, g_idx, scale_extra = gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=args.groupsize,
                    actorder=args.act_order,
                    static_groups=args.static_groups,
                )
                quantizers["model.layers.%d.%s" % (i, name)] = (
                    scale,
                    zero,
                    g_idx,
                    scale_extra
                )
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].to(cur_device),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0].cpu()

        if force_to_cpu:
            layers[i] = layer.cpu()
            del layer
        else:
            layers[i] = layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers


class QuantizedQwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
        self,
        config: Qwen2Config,
        quant_config: Dict[str, str],
        layer_idx: Optional[int] = None
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        group_size = quant_config["group_size"]
        wbits = quant_config["wbits"]
        self.q_proj = QuantLinear(wbits, group_size, self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = QuantLinear(wbits, group_size, self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = QuantLinear(wbits, group_size, self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = QuantLinear(wbits, group_size, self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2RotaryEmbedding(config=self.config)

    forward = Qwen2Attention.forward

class QuantizedQwen2MLP(nn.Module):
    def __init__(self, config: Qwen2Config, quant_config: Dict[str, str]):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        group_size = quant_config["group_size"]
        wbits = quant_config["wbits"]
        self.gate_proj = QuantLinear(wbits, group_size, self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = QuantLinear(wbits, group_size, self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = QuantLinear(wbits, group_size, self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
    
    forward = Qwen2MLP.forward

    
class QuantizedQwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, quant_config: Dict[str, str], layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        # only support Qwen2Attention for now. TODO: support Qwen2FlashAttention2 and Qwen2SdpaAttention
        self.self_attn = QuantizedQwen2Attention(config, quant_config, layer_idx)
        self.mlp = QuantizedQwen2MLP(config, quant_config)
        self.input_layernorm = Qwen2RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    forward = Qwen2DecoderLayer.forward

    
class QuantizedQwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config, quant_config: Dict[str, str]):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([QuantizedQwen2DecoderLayer(config, quant_config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    get_input_embeddings = Qwen2Model.get_input_embeddings
    set_input_embeddings = Qwen2Model.set_input_embeddings
    forward = Qwen2Model.forward
    

class QuantizedQwen2ForCausalLM(Qwen2PreTrainedModel):
    def __init__(self, config, quant_config):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.model = QuantizedQwen2Model(config, quant_config)
        # no need to quant
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    get_input_embeddings = Qwen2ForCausalLM.get_input_embeddings
    set_input_embeddings = Qwen2ForCausalLM.set_input_embeddings
    get_output_embeddings = Qwen2ForCausalLM.get_output_embeddings
    set_output_embeddings = Qwen2ForCausalLM.set_output_embeddings
    set_decoder = Qwen2ForCausalLM.set_decoder
    get_decoder = Qwen2ForCausalLM.get_decoder
    forward = Qwen2ForCausalLM.forward
    prepare_inputs_for_generation = Qwen2ForCausalLM.prepare_inputs_for_generation
    _reorder_cache = Qwen2ForCausalLM._reorder_cache  

