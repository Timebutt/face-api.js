import * as tf from '@tensorflow/tfjs-node';

import { ScaleLayerParams } from './types';

export function scale(x: tf.Tensor4D, params: ScaleLayerParams): tf.Tensor4D {
  return tf.add(tf.mul(x, params.weights), params.biases)
}
