def get_stop_tokens(tokenizer):
    stop_tokens = (
        [tokenizer.eos_id],
        [187, 187],  # '\n', '\n'
        [535],  # '\n\n'
        [2756],  # '\n\n\n',
        [tokenizer.token_to_id("User"), tokenizer.token_to_id(":")],
        [tokenizer.token_to_id("Human"), tokenizer.token_to_id(":")],
    )
    return stop_tokens
