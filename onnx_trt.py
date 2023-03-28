import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import os
import time
import cv2
import sys
from config import Config

max_batch_size = 1

TRT_LOGGER = trt.Logger()  # This logger is required to build an engine
EXPLICIT_BATCH = []
if trt.__version__[0] >= '7':
    EXPLICIT_BATCH.append(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))


def get_img_np_nchw(width, height):
    # pfe_input = np.ones([1, 3, 480, 640], dtype=np.float32)
    width = int(width)
    height = int(height)
    pfe_input = np.ones([1, 3, height, width], dtype=np.float32)
    # print(pfe_input.shape)
    return pfe_input


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def get_engine(max_batch_size=1, onnx_file_path="", engine_file_path="", \
               fp16_mode=False, int8_mode=False, save_engine=False,
               ):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine(onnx_file_path, engine_file_path, verbose=True):
        """Takes an ONNX file and creates a TensorRT engine."""
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(
                network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 14
            builder.max_batch_size = 1
            builder.fp16_mode = True

            if not os.path.exists(onnx_file_path):
                print(
                    'ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            if trt.__version__[0] >= '7':
                # Reshape input to batch size 1
                shape = list(network.get_input(0).shape)
                shape[0] = 1
                network.get_input(0).shape = shape
            print('Completed parsing of ONNX file')

            print('Building an engine; this may take a while...')
            engine = builder.build_cuda_engine(network)
            print('Completed creating engine')
            with open(engine_file_path, 'wb') as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(onnx_file_path, engine_file_path, verbose=True)


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs


if __name__ == '__main__':
    width = Config.get_input_w()
    height = Config.get_input_h()
    onnx_model_path = Config.get_onnx_path()
    img_np_nchw = get_img_np_nchw(width, height)
    print(img_np_nchw.shape)

    fp16_mode = True
    int8_mode = False
    strsp = onnx_model_path.split('.')
    trt_engine_path = '{}_fp16_new_{}_{}_pp-1.trt'.format(strsp[0], fp16_mode, int8_mode)
    # Build an engine
    engine = get_engine(max_batch_size, onnx_model_path, trt_engine_path, fp16_mode, int8_mode, True)
    # Create the context for this engine
    context = engine.create_execution_context()
    # Allocate buffers for input and output
    inputs, outputs, bindings, stream = allocate_buffers(engine)  # input, output: host # bindings

    inputs[0].host = img_np_nchw  # .reshape(-1)

    # inputs[1].host = ... for multiple input
    t1 = time.time()

    output = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)  # numpy data

    t2 = time.time()
    # feat = postprocess_the_outputs(trt_outputs[1], shape_of_output)

    print('All completed!')
