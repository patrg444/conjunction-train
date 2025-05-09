#!/usr/bin/env python3
"""
Convert a stateless Keras LSTM model into a stateful version by cloning
architecture, transferring weights, and saving the new model for streaming.
Usage:
    python3 scripts/convert_to_stateful.py \
        --input target_model/model_85.6_accuracy.keras \
        --output target_model/model_85.6_accuracy_stateful.keras \
        --feat_dim  (dimension of feature vector per timestep)
"""
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, LSTM
import os

def make_stateful_model(orig_model: Model, feat_dim: int):
    """
    Rebuild the original model with stateful LSTM layers.
    Assumes the original uses a single LSTM; for multiple, adapt accordingly.
    """
    # Identify input shape
    # batch_input_shape = (1, None, feat_dim)
    inputs = Input(batch_shape=(1, None, feat_dim), name="stateful_input")
    x = inputs
    # Traverse layers of original and recreate them
    for layer in orig_model.layers[1:]:  # skip original input
        config = layer.get_config()
        cls = layer.__class__
        if isinstance(layer, tf.keras.layers.LSTM):
            # create a stateful version
            x = cls.from_config({
                **config,
                "stateful": True,
                "batch_input_shape": (1, None, feat_dim)
            })(x)
        else:
            x = cls.from_config(config)(x)
    model = Model(inputs, x, name=orig_model.name + "_stateful")
    return model

def main(input_path, output_path, feat_dim):
    if not os.path.exists(input_path):
        print(f"Input model not found: {input_path}")
        return
    print(f"Loading original model from {input_path}")
    orig = load_model(input_path, compile=False)
    print("Building stateful clone")
    stateful = make_stateful_model(orig, feat_dim)
    print("Transferring weights")
    for src_layer, dst_layer in zip(orig.layers, stateful.layers):
        if src_layer.get_weights():
            dst_layer.set_weights(src_layer.get_weights())
    print("Saving stateful model to", output_path)
    stateful.save(output_path, include_optimizer=False)
    print("Done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert LSTM model to stateful")
    p.add_argument("--input",  type=str, required=True, help="Stateless model path")
    p.add_argument("--output", type=str, required=True, help="Stateful model path")
    p.add_argument("--feat_dim", type=int, required=True,
                   help="Feature dimensionality per time step")
    args = p.parse_args()
    main(args.input, args.output, args.feat_dim)
