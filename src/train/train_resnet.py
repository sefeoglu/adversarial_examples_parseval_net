import sys
sys.path.insert(1,'/home/sefika/AE_Parseval_Network/src/models')
from wideresnet.wresnet import WideResidualNetwork


def train(type, model):
    pass
if __name__ == "__main__":
    wideresnet_instance = WideResidualNetwork()
    ### TODO ###
    input_dim=(100,68,1)
    wrn_16_2 = wideresnet_instance.create_wide_residual_network()
    ###TODO ###
    # write parameters here ##
    hist= wrn_16_2.fit()
    ######TODO#####
    ## store hist##
    ## store model##
    ################
    