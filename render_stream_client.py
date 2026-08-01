"""
Thin Redis client that streams robot state to the stair render server and
receives egocentric camera frames back.

Runs inside the ``bm`` env alongside ``run_controller.py``; the heavy Isaac-Sim
renderer (``openpi-vlk/rendering/stair_render_server.py``) runs in the
``rendering`` env. The two only ever exchange a 36-float state vector and a JPEG
through Redis, so the envs stay fully isolated.

State packet layout (36 float32, little-endian), matching the server::

    [ root_pos(3) | root_quat_xyzw(4) | joint_pos(29, MuJoCo order) ]

Typical use from run_controller.py::

    client = StairRenderStreamClient()
    if client.connect(timeout=5):          # non-fatal if the server is absent
        ...
        client.send_state(frame_id, robot_state.root_pos,
                          robot_state.root_orn, robot_state.q)
        frame = client.try_recv_frame()    # non-blocking; None if not ready yet
        if frame is not None:
            client.show(frame)             # optional cv2 preview window
"""

import time

import numpy as np
import redis

NUM_JOINTS = 29
STATE_DIM = 3 + 4 + NUM_JOINTS  # 36


class StairRenderStreamClient:
    def __init__(self, host="localhost", port=6379,
                 input_stream="stair_render:input",
                 output_stream="stair_render:output",
                 status_key="stair_render:status",
                 window_name="robot egocentric view"):
        self.rdb = redis.Redis(host=host, port=port, decode_responses=False)
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.status_key = status_key
        self.window_name = window_name
        self._last_output_id = "0-0"
        self._frame_id = 0
        self._window_ok = False
        self.connected = False

    def connect(self, timeout=10.0):
        """Wait until the server reports ready. Returns False (no exception) if
        the server never appears, so the controller can run render-free."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if self.rdb.get(self.status_key) == b"ready":
                    # Skip any stale frames from a previous run.
                    if self.rdb.exists(self.output_stream):
                        last = self.rdb.xinfo_stream(self.output_stream).get(
                            b"last-generated-id", b"0-0")
                        self._last_output_id = last.decode() if isinstance(last, bytes) else str(last)
                    self.connected = True
                    print(f"[render_client] Connected; output cursor {self._last_output_id}")
                    return True
            except redis.exceptions.RedisError as exc:
                print(f"[render_client] Redis not reachable yet: {exc}")
            time.sleep(0.5)
        print(f"[render_client] Render server not ready after {timeout}s; "
              "continuing without rendering.")
        return False

    def send_state(self, frame_id, root_pos, root_quat_xyzw, joints):
        """Publish one robot state. quat is xyzw (scipy convention), joints are
        the 29 MuJoCo-order angles (robot_state.q)."""
        state = np.empty(STATE_DIM, dtype=np.float32)
        state[:3] = np.asarray(root_pos, dtype=np.float32)
        state[3:7] = np.asarray(root_quat_xyzw, dtype=np.float32)
        state[7:7 + NUM_JOINTS] = np.asarray(joints, dtype=np.float32)
        self.rdb.xadd(self.input_stream, {
            b"frame_id": str(int(frame_id)).encode(),
            b"state": state.tobytes(),
        })

    def try_recv_frame(self, block_ms=0):
        """Return the next rendered BGR image, or None if none is ready.

        block_ms=0 polls without blocking — the controller never stalls waiting
        on the renderer. Pass a positive value to wait up to that many ms.
        """
        result = self.rdb.xread({self.output_stream: self._last_output_id},
                                count=1, block=block_ms if block_ms > 0 else None)
        if not result:
            return None
        import cv2
        _, messages = result[0]
        msg_id, fields = messages[-1]   # keep only the freshest frame
        self._last_output_id = msg_id
        img_bytes = fields.get(b"image")
        if img_bytes is None:
            return None
        return cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    def show(self, bgr):
        """Display a frame in a cv2 window (best-effort; no-op without a display)."""
        import cv2
        try:
            if not self._window_ok:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self._window_ok = True
            cv2.imshow(self.window_name, bgr)
            cv2.waitKey(1)
        except Exception as exc:
            print(f"[render_client] Preview disabled ({exc}).")
            self._window_ok = False

    def next_frame_id(self):
        fid = self._frame_id
        self._frame_id += 1
        return fid

    def close(self):
        if self._window_ok:
            try:
                import cv2
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
