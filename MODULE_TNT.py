import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import torch
from torchvision import datasets, transforms
from MODULE_MEAN_STD import Mean_and_std_of_dataset


def train_model(model, criterion, optimizer, scheduler, DATALOADER, DEVICE, DATASETSIZE, checkpoint_model = None, num_epochs = 25, store_path = None):
    """
    This function trains the model. Input parameters are:
    1. model: pre-trained model, or a model with random initialization 
    2. criterion: loss function, e.g. "nn.CrossEntropyLoss()"
    3. optimizer: optimizer, e.g. "optim.SGD(model.parameters(), lr=0.001, momentum=0.9)"
    4. scheduler: slowly decrease the learning rate, e.g. "lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)"
    5. DATALOADER: as suggested by name
    6. DEVIDE: as suggested by name
    7. DATASETSIZE: total number of images, not the individual category
    8. num_epochs: # of training epochs, 25 by default
    9. store_path: not needed for Nova
    """

    since = time.time()
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    dataset_sizes = DATASETSIZE
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in DATALOADER[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
            
            torch.save( {
                        'Break epoch' : epoch,
                        'model_state_dict' : best_model_weights,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss
                        }, checkpoint_model)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model

def test_model(model1, datadir, noisetype = 'SPE', filename = None):
    """
    This function test the accuracy of the model.
    Parameters:
    1. model1: model to be tested
    2. datadir: the path of the folder that contains the images. e.g. ".../test"
    3. noisetype: 'SPE' by default
    5. filename: 'None' by default. If writemode is True, it needs to be specified. It is the name of the text file without extension.   
    """
    ds_mean, ds_std = Mean_and_std_of_dataset(datadir, 1)
    data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(ds_mean, ds_std)])

    testset = datasets.ImageFolder(datadir,transform=data_transforms)
    #classes = testset.classes
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
    correct1 = 0
    total = 0
    err_dist = list()
    label_dic = list()
    model1.eval()
    model1=model1.to('cuda')

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            total += labels.size(0)
            images = images.to('cuda')
            outputs1 = model1(images)
            #for retrained model
            _,predicted1 = torch.max(outputs1.data,1)
            predicted1 = predicted1.to('cpu')
            correct1 += (predicted1 == labels).sum().item()

    acc = round(100 * correct1 / total, 2)
    print(f'Accuracy of the retrained network on {total} test images: {acc} %')
    result_dir = f'./Results/Re-trained_models/{filename}/{filename}.txt'
    with open(result_dir, 'a') as F:
        F.write(f'{acc}\n')
    F.close

    err_dist.append(correct1)        
    err_dist.append(total-correct1)        
    label_dic.append("Correct")
    label_dic.append("error")
    return err_dist, label_dic

 



if __name__ == "__main__":
    pass
