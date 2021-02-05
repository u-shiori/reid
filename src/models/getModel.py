import torch

from models._MultipletNet import TripletNet, QuadrupletNet
from models._ResNet import ResNet18


def getModel(model_name, num_out, weight_path=None):
    """Modelを返す
        
    Parameters
    ----------
    model_name : str
        モデル名
    num_out : int
        出力層のunit数
    weight_path : str, default None
        モデルをロードしたいときに重みが保存されたpickleファイルのpathを入力


    Returns
    -------
    model : model

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if weight_path is None:
        resnet18 = ResNet18(num_out)
        if model_name=="Triplet" or model_name=="ImprovedTriplet" or model_name=="ClusterTriplet":
            model = TripletNet(resnet18)
        elif model_name == "Quadruplet":
            model = QuadrupletNet(resnet18)
        elif model_name == "Arcface" or model_name == "UIR":
            model = resnet18
    else:
        assert "embeddingNet" in weight_path
        model = ResNet18(num_out)
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    model.to(device)

    return model
