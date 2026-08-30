import random

from basics.bpe_tokenizer.tokenizer_experiments import get_compression_ratio


random.seed(20260828)

n_doc = 10

tinystories_ratio = get_compression_ratio(
    "TinyStoriesV2-GPT4-train", n_doc, tokenizer_dataset="owt_train"
)
owt_ratio = get_compression_ratio("owt_train", n_doc, tokenizer_dataset="TinyStoriesV2-GPT4-train")

print(f"tinystories ratio is {tinystories_ratio:.2f}, owt ratio is {owt_ratio:.2f}")
