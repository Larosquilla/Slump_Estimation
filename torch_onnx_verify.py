import torch
from model import ResNet
from config import Config
import cv2
from PIL import Image
import torchvision.transforms as transforms
resize = transforms.Resize([224, 224])
to_tensor = transforms.ToTensor()
# Export to onnx
def to_onnx():
    net = ResNet(pretrained=False, postprocess=True)
    #print(net)
    checkpoint = torch.load(Config.get_pth_path())
    state_dict = checkpoint['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name = k[7:]
        new_state_dict[name]=v
    net.load_state_dict(new_state_dict)
    net.eval()

    dummy_input = torch.zeros([1, ] + [3, Config.get_input_h(), Config.get_input_w()])
    out = net(dummy_input)
    print(out)

    torch.onnx.export(net, dummy_input, Config.get_onnx_path(), verbose=True, input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                      opset_version=10)
    print('Exported the model to ', Config.get_onnx_path())

    import onnx
    import numpy as np
    onnx_model = onnx.load(Config.get_onnx_path())
    onnx.checker.check_model(onnx_model)
    import onnxruntime

    ort_session = onnxruntime.InferenceSession(Config.get_onnx_path())
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    ort_session_slump = onnxruntime.InferenceSession(Config.get_onnx_path())
    test_img=cv2.imread("1.jpg")
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB) 
    test_img = Image.fromarray(test_img)
    test_img = resize(test_img)
    test_img = to_tensor(test_img)
    #test_face = normalize(test_face)
    test_img.unsqueeze_(0)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(test_img)}
    ort_outs = ort_session_slump.run(None, ort_inputs)
    print(ort_outs)
    print(max(max(ort_outs[0])))


if __name__ == '__main__':
    to_onnx()
