import { isTensor3D, isTensor4D } from '../utils';
import { awaitMediaLoaded } from './awaitMediaLoaded';
import { isMediaElement } from './isMediaElement';
import { NetInput } from './NetInput';
import { resolveInput } from './resolveInput';
import { TNetInput } from './types';

/**
 * Validates the input to make sure, they are valid net inputs and awaits all media elements
 * to be finished loading.
 *
 * @param input The input, which can be a media element or an array of different media elements.
 * @returns A NetInput instance, which can be passed into one of the neural networks.
 */
export async function toNetInput(inputs: TNetInput): Promise<NetInput> {
  if (inputs instanceof NetInput) {
    return inputs
  }

  let inputArgArray = Array.isArray(inputs)
      ? inputs
      : [inputs]

  if (!inputArgArray.length) {
    throw new Error('toNetInput - empty array passed as input')
  }

  const getIdxHint = (idx: number) => Array.isArray(inputs) ? ` at input index ${idx}:` : ''

  const inputArray = inputArgArray.map(resolveInput)

  for (let index = 0; index < inputArray.length; index++) {
    let input = inputArray[index]

    if ('then' in input && typeof input.then === 'function') {
      input = await input
      inputArray[index] = input
    }

    if (!isMediaElement(input) && !isTensor3D(input) && !isTensor4D(input)) {

      if (typeof inputArgArray[index] === 'string') {
        throw new Error(`toNetInput -${getIdxHint(index)} string passed, but could not resolve HTMLElement for element id ${inputArgArray[index]}`)
      }

      throw new Error(`toNetInput -${getIdxHint(index)} expected media to be of type HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | tf.Tensor3D, or to be an element id`)
    }

    if (isTensor4D(input)) {
      // if tf.Tensor4D is passed in the input array, the batch size has to be 1
      const batchSize = input.shape[0]
      if (batchSize !== 1) {
        throw new Error(`toNetInput -${getIdxHint(index)} tf.Tensor4D with batchSize ${batchSize} passed, but not supported in input array`)
      }
    }
  }

  // wait for all media elements being loaded
  await Promise.all(
    inputArray.map(input => isMediaElement(input) && awaitMediaLoaded(input))
  )

  return new NetInput(inputArray, Array.isArray(inputs))
}