#CONSTANT TO FILTER RESULTS TO FIND AND DOWNLOAD IMAGES

LISTE_BANDE=[["VV","VH"],["B04", "B03", "B02","B08"]] #for downloading the data avoid changing its value

    #[["vh","vv"],["B04", "B03", "B02","B08","cm"]]

##PROJECTION SENTINEL 2
EPSG="EPSG:32755"
#Land calssif EPSG
EPSG_LANDCLASS="EPSG:3577"

## DISPALY CONSTANT
BOUND_X=[100,1000]
BOUND_Y=[100,1000]

## CONVERT Uint16 2 Float 32
CONVERTOR=10000 #apply when displaying the tif tile but also when creating the train, test, val dataset
SCALE_S1=1
## Dataset tiles shape

##The data rescaling before going into the NN
DICT_BAND_LABEL={"R":[0],"G":[1],"B":[2],"NIR":[3]}
DICT_BAND_X={"VV":[0,2],"VH":[1,3],"R":[4],"G":[5],"B":[6],"NIR":[7]}

DICT_METHOD={"standardization": "mean_std","standardization11": "mean_std", "centering": "mean_std",
             "normalization": "min_max"," ": "min_max","centering_r":"mean_std","normalization11_r":"min_max",
             "center_norm11":"min_max","center_norm11_r":"min_max"}


#TRAINING CONSTANT
NAME_LOGS=[]
PREFIX_IM="im"
PREFIX_HIST="hist"



## CONSTANTS FOR THE EVI COMPUTATION

DICT_EVI_PARAM={"L":1, "G":2.5, "C1":6, "C2":7.5}

#CONSTANT ABOUT MINMAX EXTRACTION OF THE S2 AND NDVI DATA
DICT_TRANSLATE_BAND={"B2": "B", "B4": "R","B8": "NIR","B3":"G"}
NDVI_BAND=["B4","B8"]
EVI_BAND=["B4","B8","B2"]
NB_VI_CSV=2


#BURN SEVERITY dNDVI constant
