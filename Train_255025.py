import MODULE_Train_the_model as tr

datadir = './Training_data_mix/255025'
modelname1 = 'vgg_255025'
networkname1 = 'vgg16'
tr.Train_model(datadir,modelname1, networkname1)

modelname2 = 'res_255025'
networkname2 = 'resnet18'
tr.Train_model(datadir,modelname2, networkname2)

