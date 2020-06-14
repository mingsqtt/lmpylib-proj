import base64
import re
import numpy as np
import array


def base64_to_bytes(b64_data, altchars=b'+/'):
    """Decode base64, padding being optional.

    :param bytes_data: Base64 data as an ASCII byte array, or just the string
    :returns: The decoded bytes.
    """

    if type(b64_data) == str:
        byte_arr = bytearray()
        byte_arr.extend(map(ord, b64_data))
        data = byte_arr
    elif type(b64_data) == bytes:
        byte_arr = bytearray()
        byte_arr.extend(b64_data)
        data = byte_arr
    elif type(b64_data) == bytearray:
        data = b64_data
    else:
        raise Exception("data must be either str or bytearray.")

    if altchars is not None:
        data = re.sub(rb'[^a-zA-Z0-9%s]+' % altchars, b'', data)  # normalize
    missing_padding = len(data) % 4
    if missing_padding:
        data += b'=' * (4 - missing_padding)
    if altchars is not None:
        return base64.b64decode(data, altchars)
    else:
        return base64.b64decode(data)


def bytes_to_base64(bytes_data):
    if (type(bytes_data) == bytes) or (type(bytes_data) == bytearray):
        return base64.b64encode(bytes_data)
    elif type(bytes_data) == list:
        return base64.b64encode(array.array('B', bytes_data).tobytes())
    elif type(bytes_data) == np.ndarray:
        return base64.b64encode(array.array('B', list(bytes_data)).tobytes())
    else:
        return None


def char_to_code(chars):
    if len(chars) == 1:
        return ord(chars)
    elif len(chars) > 1:
        return [ord(c) for c in chars]
    else:
        return None


def code_to_char(char_codes):
    if type(char_codes) == int:
        return chr(char_codes)
    elif (type(char_codes) == list) or (type(char_codes) == np.ndarray):
        return [chr(c) for c in char_codes]
    else:
        return None


def codes_to_bytes(char_codes):
    if type(char_codes) == list:
        return array.array('B', char_codes).tobytes()
    elif type(char_codes) == np.ndarray:
        return array.array('B', list(char_codes)).tobytes()
    elif type(char_codes) == int:
        return array.array('B', [char_codes]).tobytes()
    else:
        return None


def bytes_to_int(bytes_data):
    return list(bytes_data)


def to_hex(bytes_data, prefix=None):
    if type(bytes_data) == int:
        h = hex(bytes_data)[2:]
        if prefix is None:
            return h.rjust(int(np.ceil(len(h)/2)*2), "0")
        else:
            return prefix + h.rjust(int(np.ceil(len(h)/2)*2), "0")
    elif (type(bytes_data) == bytes) or (type(bytes_data) == bytearray):
        if prefix is None:
            return bytes_data.hex()
        else:
            return prefix + bytes_data.hex()
    elif (type(bytes_data) == list) or (type(bytes_data) == np.ndarray):
        if prefix is None:
            return "".join([to_hex(num, None) for num in bytes_data])
        else:
            return prefix + "".join([to_hex(num, None) for num in bytes_data])
    else:
        return None


def hex_to_bytes(hex_str):
    return bytes.fromhex(hex_str.replace("0x", "").replace("#", ""))


def hex_to_int(hex_str):
    removed = hex_str.replace("0x", "").replace("#", "")
    if len(removed) <= 2:
        return int(removed, base=16)
    else:
        return [i for i in bytes.fromhex(removed)]


def construct_int_from_bytes(bytes_data, zeros_on_left=True, signed=True):
    if (type(bytes_data) == bytes) or (type(bytes_data) == bytearray) or (type(bytes_data) == np.ndarray):
        return int.from_bytes(list(bytes_data), byteorder='big' if zeros_on_left else "little", signed=signed)
    elif type(bytes_data) == list:
        return int.from_bytes(bytes_data, byteorder='big' if zeros_on_left else "little", signed=signed)
    else:
        return None
