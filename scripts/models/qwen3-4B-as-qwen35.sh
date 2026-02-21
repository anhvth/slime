# Qwen3-4B architecture with Qwen3.5 vocabulary (248320).
# Used after converting a Qwen3-4B checkpoint with convert_qwen3_to_qwen35_vocab.py.
MODEL_ARGS=(
   --swiglu
   --num-layers 36
   --hidden-size 2560
   --ffn-hidden-size 9728
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base "${MODEL_ARGS_ROTARY_BASE:-1000000}"
   --vocab-size 248320
   --kv-channels 128
   --qk-layernorm
   --untie-embeddings-and-output-weights
)
