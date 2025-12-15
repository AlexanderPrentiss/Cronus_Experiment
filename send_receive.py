import struct
import torch
import numpy as np

def send_tensor(sock, tensor):
    """
    Send a torch.Tensor over a socket.
    """
    # Ensure CPU & contiguous
    tensor = tensor.detach().cpu().contiguous()

    array = tensor.numpy()
    data = array.tobytes()

    shape = array.shape
    dtype_str = str(array.dtype)
    data_len = len(data)

    header = str((shape, dtype_str, data_len)).encode("utf-8")
    header_len = len(header)

    sock.sendall(struct.pack("!I", header_len))
    sock.sendall(header)
    sock.sendall(data)

#Receive Torch Tensor

def recv_all(sock, length):
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
        data += chunk
    return data


def recv_tensor(sock, device="cpu"):
    """
    Receive a torch.Tensor over a socket.
    """
    header_len_bytes = recv_all(sock, 4)
    (header_len,) = struct.unpack("!I", header_len_bytes)

    header_bytes = recv_all(sock, header_len)
    shape, dtype_str, data_len = eval(header_bytes.decode("utf-8"))

    data_bytes = recv_all(sock, data_len)

    array = np.frombuffer(data_bytes, dtype=np.dtype(dtype_str)).reshape(shape)
    tensor = torch.from_numpy(array).to(device)

    return tensor