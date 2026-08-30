"""
RLHF from Scratch on DistilGPT2

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_distilgpt2_tokenizer
def load_distilgpt2_tokenizer(model_name="sshleifer/tiny-gpt2"):
    # Load and return the Hugging Face tokenizer for the given model.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

# Step 2 - load_distilgpt2_model
def load_distilgpt2_model(model_name="sshleifer/tiny-gpt2"):
    # Load a causal language model from the Hugging Face Hub.
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    
    return model

# Step 3 - set_pad_token_to_eos
def set_pad_token_to_eos(tokenizer):
    # GPT-2 does not define a pad token by default.
    # Use the EOS token for padding.
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

# Step 4 - generate_and_decode
def generate_and_decode(model, tokenizer, prompt, max_new_tokens=8):
    # GPT-2-family tokenizers do not define a pad token by default.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the prompt.
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate deterministically using greedy decoding.
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode the generated sequence into a single string.
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Step 5 - greedy_decode
import torch

def greedy_decode(logits):
    """Return the argmax token id from a single-row logits vector."""
    return int(torch.argmax(logits).item())

# Step 6 - sample_with_temperature
def sample_with_temperature(logits, temperature):
    # Rescale logits by temperature.
    scaled_logits = logits / temperature

    # Convert logits to a probability distribution.
    probabilities = torch.softmax(scaled_logits, dim=-1)

    # Sample one token index from the distribution.
    token_id = torch.multinomial(probabilities, num_samples=1)

    return int(token_id.item())

# Step 7 - top_k_filter
def top_k_filter(logits, k):
    # If k covers the whole vocabulary, return a copy unchanged.
    if k >= logits.numel():
        return logits.clone()

    # Find the indices of the k largest logits.
    _, top_k_indices = torch.topk(logits, k)

    # Create a new tensor filled with -inf.
    filtered_logits = torch.full_like(logits, float("-inf"))

    # Preserve only the top-k logits.
    filtered_logits[top_k_indices] = logits[top_k_indices]

    return filtered_logits

# Step 8 - top_p_filter
def top_p_filter(logits, p):
    # Accept both Python lists and PyTorch tensors.
    logits = torch.as_tensor(logits)

    # p=1 keeps the entire vocabulary unchanged.
    if p >= 1.0:
        return logits.clone()

    # Sort logits from highest to lowest.
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    # Convert sorted logits to probabilities.
    sorted_probs = torch.softmax(sorted_logits, dim=-1)

    # Compute cumulative probability.
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens only after the cumulative probability crosses p.
    # The token that first reaches/exceeds p is retained.
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False

    # Map the sorted mask back to the original vocabulary order.
    indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
    indices_to_remove[sorted_indices] = sorted_indices_to_remove

    # Do not mutate the original tensor.
    filtered_logits = logits.clone()
    filtered_logits[indices_to_remove.bool()] = float("-inf")

    return filtered_logits

# Step 9 - build_synthetic_instruction_dataset
def build_synthetic_instruction_dataset():
    # Deterministic, small in-memory dataset for supervised fine-tuning.
    return [
        {
            "prompt": "Explain what a variable is in programming.",
            "response": "A variable is a named storage location used to hold a value that a program can read or change."
        },
        {
            "prompt": "What is the capital of France?",
            "response": "The capital of France is Paris."
        },
        {
            "prompt": "Give one benefit of regular exercise.",
            "response": "Regular exercise can improve physical fitness and support overall health."
        },
        {
            "prompt": "What does Python's len function do?",
            "response": "Python's len function returns the number of items in an object, such as the number of elements in a list or characters in a string."
        },
        {
            "prompt": "Define supervised learning.",
            "response": "Supervised learning is a machine learning approach where a model learns from examples containing input data paired with the correct target outputs."
        },
        {
            "prompt": "Why is water important for the human body?",
            "response": "Water is essential for hydration and helps the body regulate temperature, transport nutrients, and remove waste."
        },
    ]

# Step 10 - format_example
def format_example(example):
    # Render the instruction example using the exact required template.
    return (
        f"### Instruction:\n"
        f"{example['prompt']}\n\n"
        f"### Response:\n"
        f"{example['response']}"
    )

# Step 11 - apply_template
def apply_template(examples):
    # Apply the existing formatting helper to each example
    # while preserving the input order.
    return [format_example(example) for example in examples]

# Step 12 - tokenize_example
def tokenize_example(tokenizer, text, max_length=64):
    # Encode the text with truncation and no padding.
    token_ids = tokenizer.encode(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    return list(token_ids)

# Step 13 - build_labels
def build_labels(input_ids):
    # Return an independent copy of input_ids for causal LM labels.
    return list(input_ids)

# Step 14 - mask_prompt_labels
def mask_prompt_labels(labels, prompt_length):
    # Create a fresh copy so the input list is not mutated.
    masked_labels = list(labels)

    # Mask the prompt positions so they are ignored by cross-entropy loss.
    for i in range(min(prompt_length, len(masked_labels))):
        masked_labels[i] = -100

    return masked_labels

# Step 15 - pad_batch
def pad_batch(sequences, pad_id):
    # Find the length of the longest sequence.
    max_length = max(len(sequence) for sequence in sequences)

    # Right-pad each sequence to max_length without mutating the input.
    return [
        list(sequence) + [pad_id] * (max_length - len(sequence))
        for sequence in sequences
    ]

# Step 16 - make_attention_mask
def make_attention_mask(padded_ids, pad_id):
    # Mark real tokens with 1 and padding tokens with 0.
    return [
        [1 if token_id != pad_id else 0 for token_id in sequence]
        for sequence in padded_ids
    ]

# Step 17 - collate_lm_batch
def collate_lm_batch(batch, pad_id):
    # Extract input IDs and labels from each example.
    input_ids = [example["input_ids"] for example in batch]
    labels = [example["labels"] for example in batch]

    # Pad input IDs with pad_id using the upstream helper.
    padded_input_ids = pad_batch(input_ids, pad_id)

    # Pad labels with -100 so padding positions are ignored by the loss.
    padded_labels = pad_batch(labels, -100)

    # Build the attention mask from the padded input IDs.
    attention_mask = make_attention_mask(padded_input_ids, pad_id)

    # Stack everything into torch.long tensors.
    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }

# Step 18 - iterate_minibatches
import random

def iterate_minibatches(examples, batch_size, seed=0):
    # Make a copy so the original input list is not mutated.
    shuffled = list(examples)

    # Shuffle deterministically based only on seed.
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    # Yield successive minibatches, allowing the final batch to be smaller.
    for start in range(0, len(shuffled), batch_size):
        yield shuffled[start:start + batch_size]

# Step 19 - train_val_split
def train_val_split(examples, val_ratio=0.2, seed=0):
    # Make a copy so the original input list is not mutated.
    shuffled = list(examples)

    # Shuffle deterministically using the provided seed.
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    # Validation size is floor(len(examples) * val_ratio).
    val_size = int(len(shuffled) * val_ratio)

    # Carve off the validation portion; the remainder is training data.
    val = shuffled[:val_size]
    train = shuffled[val_size:]

    return train, val

# Step 20 - shift_logits_and_labels
def shift_logits_and_labels(logits, labels):
    # Drop the last logit position and the first label position
    # so that each token prediction is scored against the next token.
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    return shift_logits, shift_labels

# Step 21 - cross_entropy_loss
import torch.nn.functional as F

def cross_entropy_loss(shift_logits, shift_labels):
    """Mean next-token cross-entropy, ignoring label positions equal to -100."""
    # Flatten the batch and sequence dimensions so each token prediction
    # becomes one row of logits and one target label.
    logits = shift_logits.reshape(-1, shift_logits.size(-1))
    labels = shift_labels.reshape(-1)

    # Ignore masked positions (-100) while computing the mean loss.
    return F.cross_entropy(logits, labels, ignore_index=-100)

# Step 22 - adamw_update
def adamw_update(
    param,
    grad,
    state,
    lr,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
):
    """Apply one in-place AdamW step to `param` using `grad` and persistent `state`."""
    beta1, beta2 = betas

    # Initialize optimizer state on the first call.
    if "step" not in state:
        state["step"] = 0
        state["m"] = torch.zeros_like(param)
        state["v"] = torch.zeros_like(param)

    # Increment step count.
    state["step"] += 1
    step = state["step"]

    m = state["m"]
    v = state["v"]

    # Update first and second moments.
    m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    # Bias-corrected moments.
    m_hat = m / (1.0 - beta1 ** step)
    v_hat = v / (1.0 - beta2 ** step)

    # Decoupled weight decay.
    if weight_decay != 0.0:
        param.mul_(1.0 - lr * weight_decay)

    # AdamW parameter update.
    param.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

    return state

# Step 23 - linear_warmup_schedule
def linear_warmup_schedule(step, warmup_steps):
    # No warmup: immediately use the full learning rate.
    if warmup_steps <= 0:
        return 1.0

    # Linearly increase from 0 to 1 during warmup.
    if step < warmup_steps:
        return step / warmup_steps

    # After warmup, keep the multiplier at 1.
    return 1.0

# Step 24 - clip_grad_norm
def clip_grad_norm(grads, max_norm):
    # Compute the global L2 norm across all gradient tensors.
    total_norm_sq = sum(torch.sum(grad.detach() ** 2) for grad in grads)
    total_norm = torch.sqrt(total_norm_sq)

    # Convert the original norm to a Python float for logging.
    total_norm_value = float(total_norm.item())

    # Clip gradients in place if the global norm exceeds max_norm.
    if total_norm_value > max_norm:
        scale = max_norm / (total_norm_value + 1e-12)

        for grad in grads:
            grad.mul_(scale)

    return total_norm_value

# Step 25 - accumulate_gradients
def accumulate_gradients(grad_list):
    """Average a list of equally-shaped gradient tensors across micro-batches."""
    if not grad_list:
        raise ValueError("grad_list must be non-empty")

    # Stack gradients and take the mean across micro-batches.
    return torch.stack(grad_list, dim=0).mean(dim=0)

# Step 26 - sft_train_step
def sft_train_step(model, batch, optimizer):
    """Run one SFT forward/backward/step and return the loss as a float."""
    # Clear gradients from any previous update.
    optimizer.zero_grad()

    # Forward pass.
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    # Align predictions with next-token targets.
    shift_logits, shift_labels = shift_logits_and_labels(
        outputs.logits,
        batch["labels"],
    )

    # Compute the masked next-token cross-entropy loss.
    loss = cross_entropy_loss(shift_logits, shift_labels)

    # Backpropagate and update parameters.
    loss.backward()
    optimizer.step()

    return float(loss.item())

# Step 27 - evaluate_loss
def evaluate_loss(model, batches):
    """Mean LM loss over validation batches, no grad."""
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in batches:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            shift_logits, shift_labels = shift_logits_and_labels(
                outputs.logits,
                batch["labels"],
            )

            loss = cross_entropy_loss(shift_logits, shift_labels)

            total_loss += float(loss.item())
            num_batches += 1

    if num_batches == 0:
        raise ValueError("batches must contain at least one batch")

    return total_loss / num_batches

# Step 28 - lora_delta
def lora_delta(A, B, alpha, r):
    # Compute the low-rank update B @ A and scale it by alpha / r.
    return (alpha / r) * (B @ A)

# Step 29 - lora_linear_forward
def lora_linear_forward(x, base_weight, A, B, alpha, r, bias=None):
    # Build the LoRA weight update using the upstream helper.
    delta = lora_delta(A, B, alpha, r)

    # Combine the frozen base weight with the LoRA update.
    weight = base_weight + delta

    # Apply the linear transformation.
    output = x @ weight.T

    # Add the optional bias.
    if bias is not None:
        output = output + bias

    return output

# Step 30 - init_lora_weights
def init_lora_weights(in_features, out_features, r, seed=0):
    """Return (A, B) LoRA factors with random A and zero B so the initial delta is zero."""
    torch.manual_seed(seed)

    # A contains small random values.
    A = torch.randn(r, in_features, dtype=torch.float32) * 0.01

    # B is initialized to zero so the initial LoRA update is exactly zero.
    B = torch.zeros(out_features, r, dtype=torch.float32)

    return A, B

# Step 31 - freeze_base_params
def freeze_base_params(model):
    # Freeze all base parameters while keeping LoRA parameters trainable.
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model

# Step 32 - count_trainable_params (not yet solved)
# TODO: implement

# Step 33 - merge_lora (not yet solved)
# TODO: implement

# Step 34 - build_synthetic_preference_dataset (not yet solved)
# TODO: implement

# Step 35 - format_preference (not yet solved)
# TODO: implement

# Step 36 - reward_head_forward (not yet solved)
# TODO: implement

# Step 37 - pairwise_reward_loss (not yet solved)
# TODO: implement

# Step 38 - reward_bce_loss (not yet solved)
# TODO: implement

# Step 39 - pairwise_accuracy (not yet solved)
# TODO: implement

# Step 40 - reward_train_step (not yet solved)
# TODO: implement

# Step 41 - sequence_logprob (not yet solved)
# TODO: implement

# Step 42 - per_token_kl (not yet solved)
# TODO: implement

# Step 43 - compute_returns (not yet solved)
# TODO: implement

# Step 44 - gae_advantages (not yet solved)
# TODO: implement

# Step 45 - policy_ratio (not yet solved)
# TODO: implement

# Step 46 - clipped_surrogate (not yet solved)
# TODO: implement

# Step 47 - value_function_loss (not yet solved)
# TODO: implement

# Step 48 - entropy_bonus (not yet solved)
# TODO: implement

# Step 49 - ppo_loss (not yet solved)
# TODO: implement

# Step 50 - kl_penalized_reward (not yet solved)
# TODO: implement

# Step 51 - batch_sequence_logprob (not yet solved)
# TODO: implement

# Step 52 - dpo_logratios (not yet solved)
# TODO: implement

# Step 53 - dpo_ref_logratios (not yet solved)
# TODO: implement

# Step 54 - dpo_loss (not yet solved)
# TODO: implement

# Step 55 - ipo_loss (not yet solved)
# TODO: implement

# Step 56 - kto_loss (not yet solved)
# TODO: implement

# Step 57 - orpo_loss (not yet solved)
# TODO: implement

# Step 58 - simpo_loss (not yet solved)
# TODO: implement

# Step 59 - build_eval_prompt_set (not yet solved)
# TODO: implement

# Step 60 - generate_completions (not yet solved)
# TODO: implement

# Step 61 - score_with_reward (not yet solved)
# TODO: implement

# Step 62 - win_rate (not yet solved)
# TODO: implement

# Step 63 - stream_tokens (not yet solved)
# TODO: implement

# Step 64 - apply_stop_tokens (not yet solved)
# TODO: implement

# Step 65 - chat (not yet solved)
# TODO: implement

