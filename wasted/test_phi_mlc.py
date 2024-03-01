import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tvm.relax.frontend import nn

from mlc_chat.compiler.model.phi import modeling_phi as hf_phi
from mlc_chat.compiler.model.phi import phi_model

config = phi_model.PhiConfig()


def test_rotary():
    rotary_embed = phi_model.RotaryEmbedding(config)
    print(config.n_head)
    print(config.head_dim)
    print(rotary_embed.position_embedding_base)
    print(rotary_embed.rotary_dim)
    mod = rotary_embed.jit(
        spec={
            "forward": {
                "q": nn.spec.Tensor([1, 7, config.n_head, config.head_dim], "float16"),
                "k": nn.spec.Tensor([1, 7, config.n_head, config.head_dim], "float16"),
                "offset": int,
            }
        }
    )
    q = torch.rand(1, 7, config.n_head, config.head_dim, dtype=torch.float16)
    k = torch.rand(1, 7, config.n_head, config.head_dim, dtype=torch.float16)
    q_embed, k_embed = mod["forward"](q, k, 0)

    hf_rotary = hf_phi.RotaryEmbedding(config.rotary_dim)
    kv = torch.stack([k, k], dim=2)
    hf_q, hf_kv = hf_rotary.forward(q, kv, 0)
    print(torch.allclose(hf_q, q_embed))
    # print(hf_q)
    # print(q_embed)
    hf_k, _ = torch.unbind(hf_kv, 2)
    print(torch.allclose(hf_k, k_embed))
    # print(q_embed.shape)


def test_e2e():
    hf_model = AutoModelForCausalLM.from_pretrained(
        "/opt/scratch/lesheng/phi-2", torch_dtype=torch.float16, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("/opt/scratch/lesheng/phi-2", trust_remote_code=True)

    state_dict = hf_model.state_dict()
    state_dict["transformer.embd.weight"] = state_dict["transformer.embd.wte.weight"]
    state_dict.pop("transformer.embd.wte.weight")
    phi = phi_model.PhiForCausalLM(config)
    phi.to("float16")
    phi.load_state_dict(state_dict)
    mod = phi.jit(spec=phi.get_default_spec())

    inputs = tokenizer(
        "What is the meaning of life?",
        return_tensors="pt",
        return_attention_mask=False,
    )

    mod["prefill"](inputs, len(inputs[0]))


if __name__ == "__main__":
    test_rotary()
    # test_e2e()
    # test_e2e()
