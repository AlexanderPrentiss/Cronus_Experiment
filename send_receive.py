import struct
import numpy as np

def send_numpy(sock, array):
    """Send a NumPy array over a socket."""
    # Convert array to raw bytes
    data = array.tobytes()

    # Metadata
    shape = array.shape
    dtype_str = str(array.dtype)
    data_len = len(data)

    # Construct header as a Python tuple
    header = str((shape, dtype_str, data_len)).encode("utf-8")
    header_len = len(header)

    # --- Send header length (4 bytes big-endian) ---
    sock.sendall(struct.pack("!I", header_len))

    # --- Send header ---
    sock.sendall(header)

    # --- Send raw array bytes ---
    sock.sendall(data)

def recv_all(sock, length):
    """Receive exactly `length` bytes from the socket."""
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
        data += chunk
    return data

def recv_numpy(sock):
    """Receive a NumPy array over a socket."""
    # --- Step 1: Read header length ---
    header_len_bytes = recv_all(sock, 4)
    (header_len,) = struct.unpack("!I", header_len_bytes)

    # --- Step 2: Read header ---
    header_bytes = recv_all(sock, header_len)
    shape, dtype_str, data_len = eval(header_bytes.decode("utf-8"))

    # --- Step 3: Read array bytes ---
    data_bytes = recv_all(sock, data_len)

    # --- Step 4: Reconstruct array ---
    array = np.frombuffer(data_bytes, dtype=np.dtype(dtype_str))
    return array.reshape(shape)