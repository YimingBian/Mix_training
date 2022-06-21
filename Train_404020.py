import MODULE_Train_the_model as tr

datadir = './Training_data_mix/404020'
modelname1 = 'vgg_404020'
networkname1 = 'vgg16'
tr.Train_model(datadir,modelname1, networkname1)

modelname2 = 'res_404020'
networkname2 = 'resnet18'
tr.Train_model(datadir,modelname2, networkname2)

