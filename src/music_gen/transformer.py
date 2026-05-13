import sys
import logging
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers


from music_gen.main import Config, parse_args, prepare_data, generate, save_results
from music_gen.post_processing import NoteSampler

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def positional_encoding(
    sequence_length: int, d_model: int
) -> tf.Tensor:
    positions = tf.range(sequence_length, dtype=tf.float32)[:, tf.newaxis]
    indices = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
    angles = positions / tf.pow(10000.0, (2 * (indices // 2)) / tf.cast(d_model, tf.float32))

    sin_part = tf.sin(angles[:, 0::2])
    cos_part = tf.cos(angles[:, 1::2])

    encoding = tf.concat([sin_part, cos_part], axis=-1)
    return encoding[tf.newaxis, ...]


def build_transformer_model(
    n_notes: int,
    n_voices: int,
    cfg: Config,
) -> tf.keras.Model:
    n_notes = int(n_notes)
    n_voices = int(n_voices)
    d_model = 64
    num_heads = 4
    num_transformer_blocks = 4
    ffn_dim = 256
    dropout_rate = cfg.dropout_rate

    seq_length = cfg.window_size
    input_dim = n_voices * 5

    inputs = layers.Input(shape=cfg.input_shape, name="input_notes")
    x = layers.Reshape((seq_length, input_dim))(inputs)

    x = layers.Dense(d_model, name="input_projection")(x)

    pos_enc = positional_encoding(seq_length, d_model)
    x = x + pos_enc[:, :seq_length, :]

    for block_idx in range(num_transformer_blocks):
        residual = x

        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            name=f"mha_{block_idx}",
        )(query=x, key=x, value=x)

        attn_output = layers.Add(name=f"attn_add_{block_idx}")([residual, attn_output])
        attn_output = layers.LayerNormalization(
            epsilon=1e-6, name=f"attn_norm_{block_idx}"
        )(attn_output)

        ffn = layers.Dense(
            ffn_dim, activation="gelu", name=f"ffn_dense_{block_idx}"
        )(attn_output)
        ffn = layers.Dropout(dropout_rate, name=f"ffn_dropout_{block_idx}")(ffn)
        ffn = layers.Dense(d_model, name=f"ffn_proj_{block_idx}")(ffn)

        x = layers.Add(name=f"ffn_add_{block_idx}")([attn_output, ffn])
        x = layers.LayerNormalization(
            epsilon=1e-6, name=f"ffn_norm_{block_idx}"
        )(x)

    x = layers.GlobalAveragePooling1D(name="global_pool")(x)

    x = layers.Dense(256, activation="relu", name="dense_head_1")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    x = layers.Dense(n_notes * n_voices, name="output_logits")(x)
    x = layers.Reshape((n_voices, n_notes), name="output_reshape")(x)
    outputs = layers.Softmax(axis=-1, name="output_softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="transformer_bach")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


def train_model(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cfg: Config,
    model_path_override: str | None = None,
) -> tf.keras.callbacks.History:
    model_path = model_path_override or cfg.model_path

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=cfg.log_dir,
        histogram_freq=1,
    )

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=cfg.early_stopping_patience,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1,
    )

    callbacks = [checkpoint_cb, tensorboard_cb, early_stopping_cb, reduce_lr_cb]

    return model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=cfg.epochs,
        callbacks=callbacks,
    )


def run(cfg: Config | None = None) -> None:
    if cfg is None:
        cfg = parse_args()

    cfg.model_path = cfg.model_path.replace(".keras", "_transformer.keras")
    cfg.midi_dir = cfg.midi_dir + "_transformer"
    cfg.output_file = cfg.output_file.replace(".txt", "_transformer.txt")

    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)

    voices, n_notes, x_train, y_train, x_test, y_test = prepare_data(cfg)
    note_sampler = NoteSampler(voices)

    model = build_transformer_model(n_notes, len(voices), cfg)

    history = train_model(model, x_train, y_train, x_test, y_test, cfg)

    best_val_acc = max(history.history["val_accuracy"])
    logger.info("[+] best val accuracy: %.4f", best_val_acc)

    logger.info("[*] loading best model for generation...")
    model = tf.keras.models.load_model(cfg.model_path)

    prediction = generate(model, voices, note_sampler, cfg)
    save_results(prediction, note_sampler, voices, cfg)


if __name__ == "__main__":
    run()
