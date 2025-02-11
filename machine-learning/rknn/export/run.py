import glob
import os
import onnx
from rknn.api import RKNN

PLATFORM = "rk3568"
DEFAULT_DIMENSIONS = [1, 3, 112, 112]


def getShapes(model):
    names = []
    shapes = []
    for input in onnx.load(model).graph.input:
        names.append(input.name)
        shape = []
        shapes.append(shape)
        tensor_type = input.type.tensor_type
        if tensor_type.HasField("shape"):
            i = 0
            for d in tensor_type.shape.dim:
                if d.HasField("dim_value"):
                    shape.append(d.dim_value)
                else:
                    shape.append(DEFAULT_DIMENSIONS[i])
                i += 1
    return names, shapes


def export(onnx_model_path, platform):
    name, _ = os.path.splitext(os.path.basename(onnx_model_path))
    rknn_model_path = "output/" + name + ".rknn"
    names, shapes = getShapes(onnx_model_path)

    rknn = RKNN()
    rknn.config(target_platform=platform)
    rknn.load_onnx(onnx_model_path, names, shapes)
    rknn.build(do_quantization=False)
    rknn.export_rknn(rknn_model_path)
    rknn.release()

def main():
    models = [
        ("buffalo_l_rec.onnx", PLATFORM),
        ("buffalo_l_det.onnx", PLATFORM),
    ]

    for model in models:
        export(model[0], model[1])


if __name__ == "__main__":
    main()
