import sys
sys.path.insert(1,'/home/sefika/AE_Parseval_Network/src/models')
from Parseval_Networks_OC.parsnet_oc import ParsevalNetwork


def train(type, model):
    pass
if __name__ == "__main__":
    parseval_instance = ParsevalNetwork()
    ### TODO ###
    input_dim=(100,68,1)
    pars_oc_16_2 = parseval_instance.create_wide_residual_network()
    ###TODO ###
    # write parameters here ##
    hist= pars_oc_16_2.fit()
    ######TODO#####
    ## store hist##
    ## store model##
    ################
