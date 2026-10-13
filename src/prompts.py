import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate_completion(
    prompt: str,
    model_name: str = "gpt2",
    max_new_tokens: int = 60,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str | None = None,
) -> str:
    """Genera una continuacion del prompt con un modelo causal de HuggingFace."""
    device = device or pick_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # gpt2 no trae pad token y generate() se queja sin uno
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
