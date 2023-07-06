def get_stop_tokens(tokenizer):
    stop_tokens = (
        [tokenizer.eos_id],
        [tokenizer.token_to_id("User"), tokenizer.token_to_id(":")],
        [tokenizer.token_to_id("Human"), tokenizer.token_to_id(":")],
        [tokenizer.token_to_id("Human:")],
        [tokenizer.token_to_id("User:")],
    )
    return stop_tokens
