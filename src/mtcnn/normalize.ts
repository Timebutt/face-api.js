import * as tf from '@tensorflow/tfjs-node';

export function normalize(x: tf.Tensor4D): tf.Tensor4D {
  return tf.tidy(
    () => tf.mul(tf.sub(x, tf.scalar(127.5)), tf.scalar(0.0078125))
  )
}