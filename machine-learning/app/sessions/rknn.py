from __future__ import annotations

import platform
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from rknn.api import RKNN
from app.schemas import SessionNode

from ..config import log, settings


DEVICE_COMPATIBLE_NODES = [
    '/proc/device-tree/compatible',
    '/device-tree/compatible',
]

def is_available() -> bool:
    system = platform.system()
    machine = platform.machine()
    os_machine = system + '-' + machine
    if os_machine != 'Linux-aarch64':
        return False
    for p in DEVICE_COMPATIBLE_NODES:
        path = Path(p)
        if not path.exists():
            continue
        with p.open() as f:
            device_compatible_str = f.read()
            if 'rk3562' in device_compatible_str:
                host = 'RK3562'
            elif 'rk3576' in device_compatible_str:
                host = 'RK3576'
            elif 'rk3588' in device_compatible_str:
                host = 'RK3588'
            else:
                host = 'RK3566_RK3568'
                return True
    return False

class RknnSession:
    """
    Wrapper for RKNN to be drop-in replacement for ONNX session.
    """

    def __init__(self, model_path: Path, cache_dir: Path = settings.cache_folder) -> None:
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.rknn = RKNN()

        log.info("Loading RKNN model %s ...", model_path)
        self.model = self.rknn.load(model_path.as_posix())
        log.info("Loaded RKNN model with ID %d", self.model)

    def __del__(self) -> None:
        self.rknn.release()
        log.info("Unloaded RKNN model %d", self.model)

    def get_inputs(self) -> list[SessionNode]:
        inputs = []
        for a in self.rknn.rknn_base.inputs_meta.values():
            for key, val in dict(a).items():
                if val["is_output"] == False:
                    inputs.append(RknnNode(key, val["shape"]))
        return inputs

    def get_outputs(self) -> list[SessionNode]:
        outputs = []
        for a in self.rknn.rknn_base.inputs_meta.values():
            for key, val in dict(a).items():
                if val["is_output"] == True:
                    outputs.append(RknnNode(key, val["shape"]))
        return outputs

    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, NDArray[np.float32]] | dict[str, NDArray[np.int32]],
        run_options: Any = None,
    ) -> list[NDArray[np.float32]]:
        inputs: list[NDArray[np.float32]] = [np.ascontiguousarray(v) for v in input_feed.values()]
        return self.rknn.inference(inputs)


class RknnNode(NamedTuple):
    name: str | None
    shape: tuple[int, ...]
