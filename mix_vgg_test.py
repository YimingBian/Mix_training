import MODULE_Test_model_accuracy as tst
import MODULE_TNT as tnt
import torch

data_dirs = [   '../../ProjectAlpha/Testing_data/Original/test',  # ori
                # SNP data
                '../../ProjectAlpha/Testing_data/SNP/SNP_0.1/test', 
                '../../ProjectAlpha/Testing_data/SNP/SNP_0.2/test',
                '../../ProjectAlpha/Testing_data/SNP/SNP_0.3/test',
                '../../ProjectAlpha/Testing_data/SNP/SNP_0.4/test',
                '../../ProjectAlpha/Testing_data/SNP/SNP_0.5/test'
]

model_dir = [   './Results/Re-trained_models/vgg_204040/vgg_204040.pth',
                './Results/Re-trained_models/vgg_255025/vgg_255025.pth',
                './Results/Re-trained_models/vgg_333333/vgg_333333.pth',
                './Results/Re-trained_models/vgg_404020/vgg_404020.pth'                
                ]
model_name = ['vgg_204040', 'vgg_255025', 'vgg_333333', 'vgg_404020']
networkname = 'vgg16'
output_dim = 5
noise = 'SNP' # does not matter
tst.Test_accuracy_retrained_models(model_dir, model_name, networkname, output_dim, noise, data_dirs)
