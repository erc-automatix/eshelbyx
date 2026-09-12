from jaxmat.tensors import Tensor


def flatten_batch_dims(T: Tensor):
    return T.__class__(array=T.array.reshape((-1, *T.base_array_shape)))


def reshape_batch_dims(T: Tensor, shape):
    return T.__class__(array=T.array.reshape(shape + T.base_array_shape))
