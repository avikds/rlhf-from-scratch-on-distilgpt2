# RLHF from Scratch on DistilGPT2

Build the full Reinforcement Learning from Human Feedback pipeline on distilgpt2 from scratch: decoding, supervised fine-tuning, LoRA adapters, reward modeling, PPO, and preference-optimization methods like DPO, IPO, KTO, ORPO, and SimPO. Ends with evaluation tooling and a minimal chat interface to compare aligned and unaligned models.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** load_distilgpt2_tokenizer
- [x] **2.** load_distilgpt2_model
- [x] **3.** set_pad_token_to_eos
- [x] **4.** generate_and_decode
- [x] **5.** greedy_decode
- [x] **6.** sample_with_temperature
- [x] **7.** top_k_filter
- [x] **8.** top_p_filter
- [x] **9.** build_synthetic_instruction_dataset
- [x] **10.** format_example
- [x] **11.** apply_template
- [x] **12.** tokenize_example
- [x] **13.** build_labels
- [ ] **14.** mask_prompt_labels
- [ ] **15.** pad_batch
- [ ] **16.** make_attention_mask
- [ ] **17.** collate_lm_batch
- [ ] **18.** iterate_minibatches
- [ ] **19.** train_val_split
- [ ] **20.** shift_logits_and_labels
- [ ] **21.** cross_entropy_loss
- [ ] **22.** adamw_update
- [ ] **23.** linear_warmup_schedule
- [ ] **24.** clip_grad_norm
- [ ] **25.** accumulate_gradients
- [ ] **26.** sft_train_step
- [ ] **27.** evaluate_loss
- [ ] **28.** lora_delta
- [ ] **29.** lora_linear_forward
- [ ] **30.** init_lora_weights
- [ ] **31.** freeze_base_params
- [ ] **32.** count_trainable_params
- [ ] **33.** merge_lora
- [ ] **34.** build_synthetic_preference_dataset
- [ ] **35.** format_preference
- [ ] **36.** reward_head_forward
- [ ] **37.** pairwise_reward_loss
- [ ] **38.** reward_bce_loss
- [ ] **39.** pairwise_accuracy
- [ ] **40.** reward_train_step
- [ ] **41.** sequence_logprob
- [ ] **42.** per_token_kl
- [ ] **43.** compute_returns
- [ ] **44.** gae_advantages
- [ ] **45.** policy_ratio
- [ ] **46.** clipped_surrogate
- [ ] **47.** value_function_loss
- [ ] **48.** entropy_bonus
- [ ] **49.** ppo_loss
- [ ] **50.** kl_penalized_reward
- [ ] **51.** batch_sequence_logprob
- [ ] **52.** dpo_logratios
- [ ] **53.** dpo_ref_logratios
- [ ] **54.** dpo_loss
- [ ] **55.** ipo_loss
- [ ] **56.** kto_loss
- [ ] **57.** orpo_loss
- [ ] **58.** simpo_loss
- [ ] **59.** build_eval_prompt_set
- [ ] **60.** generate_completions
- [ ] **61.** score_with_reward
- [ ] **62.** win_rate
- [ ] **63.** stream_tokens
- [ ] **64.** apply_stop_tokens
- [ ] **65.** chat

---

Built on Deep-ML.
