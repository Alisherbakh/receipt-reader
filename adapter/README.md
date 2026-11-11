---
base_model: Qwen/Qwen2-VL-2B-Instruct
library_name: peft
tags:
- lora
- sft
- trl
- receipts
---

# Receipt Reader LoRA adapter

LoRA adapter for `Qwen/Qwen2-VL-2B-Instruct` that reads a receipt image and returns
`{"items": [{"name", "count", "price"}], "total"}` as JSON.

- Trained on 400 receipts from CORD v2 with TRL `SFTTrainer` (completion-only loss), 4-bit NF4 base, float16
- LoRA r=8, alpha=16, dropout 0.05, target modules `q_proj`, `v_proj`
- Processor image size: 128×28×28 to 768×28×28 pixels (saved in `processor_config.json`)

Results and loading code are in the repository README.
