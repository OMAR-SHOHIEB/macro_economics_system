from tensorflow.keras.layers import (
    MultiHeadAttention,
    LayerNormalization
)


def attention_block(
    x,
    num_heads=4,
    key_dim=32
):

    attn_out = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=0.1
    )(x, x)

    x = LayerNormalization()(
        x + attn_out
    )

    return x


