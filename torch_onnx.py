import torch
from model import ResNet
from config import Config


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
    onnx_model = onnx.load('weights/model_slump_16_v4_2022_03_11.onnx')
    onnx.checker.check_model(onnx_model)
if __name__ == '__main__':
    to_onnx()
