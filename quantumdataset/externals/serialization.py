"""Quantum Inspire library

Copyright 2019 QuTech Delft

qilib is available under the [MIT open-source license](https://opensource.org/licenses/MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import base64
from functools import partial
from json import JSONDecoder, JSONEncoder
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

# A callable type for transforming a given argument with a type to another type
TransformFunctionResult = Any
TransformFunction = Callable[[Any], TransformFunctionResult]


class JsonSerializeKey:
    """The custom value types for the JSON serializer"""

    OBJECT = "__object__"
    CONTENT = "__content__"


class Encoder(JSONEncoder):
    """A JSON encoder"""

    def __init__(self, **kwargs: Any) -> None:
        """Constructs a JSON Encoder"""
        super().__init__(**kwargs)
        # creates a new transform table
        self.encoders: Dict[type, TransformFunction] = {}

    def default(self, o: Any) -> TransformFunctionResult:
        if type(o) in self.encoders:
            return self.encoders[type(o)](o)

        return JSONEncoder.default(self, o)


class Decoder(JSONDecoder):
    """A JSON decoder"""

    def __init__(self) -> None:
        """Constructs a JSON Decoder"""
        super().__init__(object_hook=self._object_hook)
        # creates a new transform table
        self.decoders: Dict[str, TransformFunction] = {}

    def _object_hook(self, obj: Any) -> TransformFunctionResult:
        if isinstance(obj, dict):
            if JsonSerializeKey.OBJECT in obj:
                if obj[JsonSerializeKey.OBJECT] in self.decoders:
                    return self.decoders[obj[JsonSerializeKey.OBJECT]](obj)
                else:
                    raise ValueError(f"object key {obj[JsonSerializeKey.OBJECT]} not in decoders")

        return obj


def _encode_bytes_base64(data: bytes) -> Dict[str, Any]:
    return {JsonSerializeKey.OBJECT: "bytes_base64", JsonSerializeKey.CONTENT: base64.b64encode(data).decode("utf-8")}


def _decode_bytes_base64(data: Dict[str, Any]) -> bytes:
    return base64.b64decode(data[JsonSerializeKey.CONTENT].encode("utf-8"))


def _encode_numpy_number(item: Any) -> Dict[str, Any]:
    if isinstance(item, (np.float32, np.float64)):
        return float(item)
    if isinstance(item, np.bool_):
        return bool(item)

    return {
        JsonSerializeKey.OBJECT: "__npnumber__",
        JsonSerializeKey.CONTENT: {
            "__npnumber__": base64.b64encode(item.tobytes()).decode("ascii"),
            "__data_type__": item.dtype.str,
        },
    }


def _decode_numpy_number(item: Dict[str, Any]) -> Any:
    obj = item[JsonSerializeKey.CONTENT]
    return np.frombuffer(base64.b64decode(obj["__npnumber__"]), dtype=np.dtype(obj["__data_type__"]))[0]


numpy_ndarray_type = npt.NDArray[Any]


class NumpyKeys:
    """The custom values types for encoding and decoding numpy arrays."""

    OBJECT: str = "__object__"
    CONTENT: str = "__content__"
    DATA_TYPE: str = "__data_type__"
    SHAPE: str = "__shape__"
    ARRAY: str = "__ndarray__"


def encode_numpy_array(array: numpy_ndarray_type, encode_to_bytes: bool = True) -> Dict[str, Any]:
    """Encode numpy array to store in database.
    Args:
        array: Numpy array to encode.
        encode_to_bytes: If True, encode to bytes to, otherwise to str type

    Returns:
        The encoded array.

    """
    if encode_to_bytes:
        data: Union[str, bytes] = array.tobytes()
    else:
        data = base64.b64encode(array.tobytes()).decode("ascii")
    return {
        NumpyKeys.OBJECT: np.array.__name__,
        NumpyKeys.CONTENT: {
            NumpyKeys.ARRAY: data,
            NumpyKeys.DATA_TYPE: array.dtype.str,
            NumpyKeys.SHAPE: list(array.shape),
        },
    }


def decode_numpy_array(encoded_array: Dict[str, Any]) -> numpy_ndarray_type:
    """Decode a numpy array from database.

    Args:
        encoded_array: The encoded array to decode.

    Returns:
        The decoded array.
    """
    array: numpy_ndarray_type
    content = encoded_array[NumpyKeys.CONTENT]
    data = content[NumpyKeys.ARRAY]
    if isinstance(data, str):
        # decode
        data = base64.b64decode(data)
    array = np.frombuffer(data, dtype=np.dtype(content[NumpyKeys.DATA_TYPE])).reshape(content[NumpyKeys.SHAPE])
    # recreate the array to make it writable
    array = np.array(array)

    return array


class Serializer:
    """A general serializer to serialize data to JSON and vice versa. It allows
    extending the types with a custom encoder and decoder."""

    def __init__(
        self,
        encoders: Optional[Dict[type, TransformFunction]] = None,
        decoders: Optional[Dict[str, TransformFunction]] = None,
    ):
        """Creates a serializer

        Args:
            encoders: The default encoders if any
            decoders: The default decoders if any
        """

        self.encoder = Encoder()
        self.decoder = Decoder()

        if encoders is None:
            encoders = {}
        self.encoder.encoders = encoders
        if decoders is None:
            decoders = {}
        self.decoder.decoders = decoders

        self.register(bytes, _encode_bytes_base64, "bytes_base64", _decode_bytes_base64)
        self.register(
            np.ndarray, partial(encode_numpy_array, encode_to_bytes=False), np.array.__name__, decode_numpy_array
        )
        self.register(tuple, self._encode_tuple, tuple.__name__, self._decode_tuple)
        for numpy_integer_type in [np.int16, np.int32, np.int64, np.float16, np.float32, np.float64, np.bool_]:
            self.register(numpy_integer_type, _encode_numpy_number, "__npnumber__", _decode_numpy_number)

    def _encode_tuple(self, item: Tuple[Any, Any]) -> Dict[str, Any]:
        return {
            JsonSerializeKey.OBJECT: tuple.__name__,
            JsonSerializeKey.CONTENT: [self.encode_data(value) for value in item],
        }

    def _decode_tuple(self, data: Dict[str, Any]) -> Tuple[Any, ...]:
        return tuple(data[JsonSerializeKey.CONTENT])

    def register(
        self, type_: type, encode_func: TransformFunction, type_name: str, decode_func: TransformFunction
    ) -> None:
        """Registers an encoder and decoder for a given type

        An encode function is expected to return a JSON valid type or a dictionary with the keys
        `JsonSerializeKey.OBJECT` (the name of the type) and `JsonSerializeKey.CONTENT` (the JSON valid encoded content)

        Args:
            type_: The type to encode
            encode_func: The transform function for encoding that type
            type_name: The type name to decode
            decode_func: The transform function for decoding
        """

        self.encoder.encoders[type_] = encode_func
        self.decoder.decoders[type_name] = decode_func

    def encode_data(self, data: Any) -> Any:
        """Recursively transform a Python object and apply transform functions to it

        Args:
            data: Any Python object that can be handled by an encode/transform function for that type

        Returns:
            The transformed data
        """

        if isinstance(data, dict):
            new_dict = {}

            for key, value in data.items():
                new_dict[key] = self.encode_data(value)

            return new_dict

        elif isinstance(data, list):
            new_list = []
            for item in data:
                new_list.append(self.encode_data(self._encode(item)))

            return new_list

        return self._encode(data)

    def _encode(self, data: Any) -> Any:
        type_ = type(data)
        if type_ in self.encoder.encoders:
            return self.encoder.encoders[type_](data)

        return data

    def decode_data(self, data: Any) -> Any:
        """Recursively transform an object and apply transform functions to it

        Args:
            data: Any Python object that can be handled by an decode/transform function for that type

        Returns:
            The transformed data
        """

        if isinstance(data, dict):
            new_dict = {}
            for key, value in data.items():
                new_dict[key] = self.decode_data(value)

            return self._decode(new_dict)

        if isinstance(data, list):
            new_list = []
            for item in data:
                new_list.append(self.decode_data(item))
            return new_list

        return data

    def _decode(self, data: Any) -> Any:
        if isinstance(data, dict):
            if JsonSerializeKey.OBJECT in data:
                if data[JsonSerializeKey.OBJECT] in self.decoder.decoders:
                    return self.decoder.decoders[data[JsonSerializeKey.OBJECT]](data)
                else:
                    raise ValueError(f"object key {data[JsonSerializeKey.OBJECT]} not in decoders")

        return data


if __name__ == "__main__":
    serializer = Serializer()
    self = serializer
    data = {"a": np.array([1.0, 2.0, 3.0]), "str": "s", "bytes": b"\x00\x00\x00\x00\x00\x00\xf0?\x00\x00"}
    encoded_data = serializer.encode_data(data)
    data2 = serializer.decode_data(encoded_data)
    b = encoded_data["a"]["__content__"]["__ndarray__"]

    d = serializer.encode_data([1, 2, np.array([1.0, 2.0, np.nan])])
