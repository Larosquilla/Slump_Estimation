import configparser


class Config():
    config = configparser.ConfigParser()
    config.read('config.ini')

    @staticmethod
    def get_class_num():
        value = Config.config['dataset']['class_num']
        value = int(value)
        return value

    @staticmethod
    def get_class_label():
        value = Config.config['dataset']['class_label']
        value = value.split(',')
        value = [float(i) for i in value]
        return value

    @staticmethod
    def get_train_path():
        value = Config.config['dataset']['train_path']
        return value

    @staticmethod
    def get_val_path():
        value = Config.config['dataset']['val_path']
        return value

    @staticmethod
    def get_input_h():
        value = Config.config['image_size']['input_h']
        value = int(value)
        return value

    @staticmethod
    def get_input_w():
        value = Config.config['image_size']['input_w']
        value = int(value)
        return value

    @staticmethod
    def get_batch_size():
        value = Config.config['train']['batch_size']
        value = int(value)
        return value

    @staticmethod
    def get_epoch():
        value = Config.config['train']['epoch']
        value = int(value)
        return value

    @staticmethod
    def get_learning_rate():
        value = Config.config['train']['lr']
        value = float(value)
        return value

    @staticmethod
    def get_pth_path():
        value = Config.config['save_path']['pth_model_path']
        return value

    @staticmethod
    def get_onnx_path():
        value = Config.config['save_path']['onnx_model_path']
        return value
