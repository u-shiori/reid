
from distances._ObjectDistances import triplet_euclidean, doublet_euclidean, triplet_cosin, doublet_cosin
from distances._ClusterDistances import single_method, complete_method, average_method, centroid_method, ward_method

def getObjectDistance(model_name, triplet=True):
    """要素間距離の関数を返す
        
    Parameters
    ----------
    model_name : str
        モデル名
    triplet : bool
        triplet用の距離関数が欲しいかどうか

    Returns
    -------
    if triplet is True:
        triplet_distance
    if triplet is False:
        doublet_distance : 2サンプルの距離を測る

    """
    if triplet:
        if model_name=="Arcface" or model_name=="UIR":
            return triplet_cosin
        else:
            return triplet_euclidean
    
    else:
        if model_name=="Arcface" or model_name=="UIR":
            return doublet_cosin
        else:
            return doublet_euclidean


def getClusterDistance(model_name, method_name):
    """クラスタ間距離の関数を返す
        
    Parameters
    ----------
    model_name : str
        モデル名
    method_name : str
        クラスタ間距離法の名前

    Returns
    -------
    cluster_distance : 距離関数

    """
    object_distance = getObjectDistance(model_name, triplet=False)

    if method_name == "Single":
        return single_method(object_distance)
    elif method_name == "Complete":
        return complete_method(object_distance)
    elif method_name == "Average":
        return average_method(object_distance)
    elif method_name == "Centroid":
        return centroid_method(object_distance)
    elif method_name == "Ward":
        return ward_method(object_distance)

