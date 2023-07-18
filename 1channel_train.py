import torch
from torch import nn
from torch.utils.data import DataLoader
from bainarydatasest import SoundDataset
# from models.conformer import Conformer
# from models.crnn import CRNN
from models.transepertrain import TransferTrainModel
from models.vgg16 import VGG_16
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import torchvision.transforms.functional as F
import os
from tqdm import tqdm
from sklearn.metrics import f1_score
# from torchsummary import summary
# import utils

#----------------------------------------------------------------------------------------------------------------
BATCH_SIZE = 12
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 50  # Number of epochs to wait for improvement
SNR= 10
TIME = 4
image = False

file_path = 'binary_class.csv'
save_dir = "test"
data_dir = "./binarysoundset"
#----------------------------------------------------------------------------------------------------------------
def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    return train_dataloader
#----------------------------------------------------------------------------------------------------------------
# input = torch.Size([24, 1, 64, 345])
def rolling_augmentation(features,shift):

    rolled_tensor = torch.roll(features, shifts=int(shift/2), dims=-1)

    feat_np = features.squeeze().detach().cpu().numpy()
    plot_rolled_np = rolled_tensor.squeeze().detach().cpu().numpy()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))       
    librosa.display.specshow(feat_np, x_axis='time', y_axis='mel', sr=44100, ax=axs[0], cmap='inferno')
    axs[0].set_title('org')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Mel Filter')
    librosa.display.specshow(plot_rolled_np, x_axis='time', y_axis='mel', sr=44100, ax=axs[1], cmap='inferno')
    axs[1].set_title('roll')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Mel Filter')
    plt.tight_layout()
    plt.show()

    # fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    # axs[0].imshow(features.permute(1,2,0).cpu().numpy().astype(np.uint8), aspect='auto', cmap='inferno')
    # axs[0].set_title('org')
    # axs[0].set_xlabel('Time')
    # axs[0].set_ylabel('Mel Filter')
    # axs[1].imshow(rolled_tensor.permute(1,2,0).cpu().numpy().astype(np.uint8), aspect='auto', cmap='inferno')
    # axs[1].set_title('noise')
    # axs[1].set_xlabel('Time')
    # axs[1].set_ylabel('Mel Filter')
    # plt.tight_layout()
    # plt.show()
    return rolled_tensor
#----------------------------------------------------------------------------------------------------------------
def G_noise(features, snr=None):
    # channel = 1
    if snr is None:
        return features
    
    new_features = features.clone().to(device)
    new_features_cpu = new_features.cpu()
    new_features_numpy = new_features_cpu.numpy()
    nbatch, channel, nfrq, nfrm = new_features_cpu.shape
    noise = torch.zeros(nbatch, channel, nfrq, nfrm).to(device)
    for bter in range(nbatch):
        feat = new_features_numpy[bter]
        std = np.sqrt(np.mean((feat ** 2) * (10 ** (-snr/10)), axis=-2))
            # temp= feat[i] + std * np.random.randn(nfrq, nfrm) #multi channel noise
        temp = feat.squeeze() + std * np.random.randn(nfrq, nfrm) #single channel noise

            # noise[bter,i] = torch.tensor(temp).float().to(device) #multi channel noise
        noise[bter, 0] = torch.tensor(temp).float().to(device) #single channel noise
       
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].imshow(feat.transpose(1, 2, 0).astype(np.uint8), aspect='auto', cmap='inferno')
        axs[0].set_title('org')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Mel Filter')
        noise_shape = noise.squeeze(0).permute(1, 2, 0)
        noise_shape = noise_shape.cpu().numpy()
        axs[1].imshow(noise_shape.astype(np.uint8), aspect='auto', cmap='inferno')
        axs[1].set_title('noise')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Mel Filter')
        plt.tight_layout()
        plt.show()

    return noise
#----------------------------------------------------------------------------------------------------------------
def mixup(input,roll):
    input_mix = input*0.5 + roll*0.5
    
    feat_np = input.squeeze().detach().cpu().numpy()
    plot_rolled_np = roll.squeeze().detach().cpu().numpy()
    input_mix_np = input_mix.squeeze().detach().cpu().numpy()
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs[0].imshow(input.permute(1,2,0).cpu().numpy().astype(np.uint8), aspect='auto', cmap='inferno', origin='lower')
    axs[0].set_title('org')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Mel Filter')
    axs[1].imshow(roll.permute(1,2,0).cpu().numpy().astype(np.uint8), aspect='auto', cmap='inferno', origin='lower')
    axs[1].set_title('roll')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Mel Filter')
    axs[2].imshow(input_mix.permute(1,2,0).cpu().numpy().astype(np.uint8), aspect='auto', cmap='inferno', origin='lower')
    axs[2].set_title('mix')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Mel Filter')
    plt.tight_layout()
    plt.show()

    return input_mix
#----------------------------------------------------------------------------------------------------------------
def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        unknown_list = []
        # print(inputs.shape)
        inputs = inputs.float().to(device)
        labels = labels.to(device)
        inputs_arg = inputs.clone()
        labels_arg = labels.clone()
        data_index = torch.argmax(labels, dim=1)
#----------------------------------------------------------------------------------------------------------------
#another another mix
        for i,bter in enumerate(data_index):
            if bter == 0:
                inputs_noise = G_noise(inputs[i].unsqueeze(dim=0) ,SNR)
                inputs_arg = torch.cat([inputs_arg,inputs_noise],dim=0)
                labels_arg = torch.cat([labels_arg,labels[i].unsqueeze(dim=0)], dim=0)

                inputs_roll = rolling_augmentation(inputs[i],(inputs.shape[-1]))
                inputs_arg = torch.cat([inputs_arg, inputs_roll.unsqueeze(dim=0)], dim=0)
                labels_arg = torch.cat([labels_arg,labels[i].unsqueeze(dim=0)], dim=0)
            elif bter == 1:
                unknown_list.append(inputs[i])
        
        for k,_ in enumerate(unknown_list):
            if len(unknown_list) == 0 or k > 1 :
                break
            if len(unknown_list) == 2 or len(unknown_list) == 3:
                inputs_mix = mixup(unknown_list[0+k],unknown_list[len(unknown_list)-1-k])
                inputs_arg = torch.cat([inputs_arg, inputs_mix.unsqueeze(dim=0)], dim=0)
                label_mix = torch.Tensor([[0, 1]]).to(device)
                labels_arg = torch.cat([labels_arg,label_mix], dim=0)
                break
            inputs_mix = mixup(unknown_list[0+k],unknown_list[len(unknown_list)-1-k])
            inputs_arg = torch.cat([inputs_arg, inputs_mix.unsqueeze(dim=0)], dim=0)
            label_mix = torch.Tensor([[0, 1]]).to(device)
            labels_arg = torch.cat([labels_arg,label_mix], dim=0) 
#----------------------------------------------------------------------------------------------------------------
        # print(labels_arg.shape)
        # print(labels_arg.shape,labels_arg.shape)
        # len_inputs += inputs_arg.shape[0]
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs_arg)
        # _, predicted = torch.max(outputs.data, 1)

        # Compute loss
        loss = criterion(outputs.float(), labels_arg.float())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        torch.cuda.synchronize()
        total += labels_arg.size(0)
        
        predicted_labels = torch.round(outputs)  # Convert predictions to binary (0 or 1)
        
        correct += (torch.sum(predicted_labels == labels_arg).item())/2 # Calculate accuracy

    accuracy = correct / total
    average_loss = total_loss / len(train_loader)
    return accuracy, average_loss
#----------------------------------------------------------------------------------------------------------------
def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.float()
            inputs = inputs.to(device)

            labels = labels.to(device)
        
            outputs = model(inputs)
    
            loss = criterion(outputs.float(), labels.float())

            total_loss += loss.item()
            total += labels.size(0)
            
            predicted_labels = torch.round(outputs)
            
            correct += (torch.sum(predicted_labels == labels).item())/2

    # Compute accuracy
    accuracy = correct / total

    # Compute average loss
    average_loss = total_loss / len(val_loader)

    return accuracy, average_loss
#----------------------------------------------------------------------------------------------------------------
def test(cnn, test_loader, save_dir):
    class_mapping = [
    'wildboar',
    'unknown'
]
    state_dict = torch.load(f"./saved_model/{save_dir}.pth")
    cnn.load_state_dict(state_dict)

    y_true = []
    y_pred = []
    probabilities_list = []
    PT=0
    PF=0
    NT=0
    NF=0

    for inputs, targets in test_loader:
        inputs = inputs.float().to(device)
        targets = targets.float().to(device)

        target_index = targets.nonzero(as_tuple=False).item()
        predicted, expected, probability = predict(cnn, inputs, target_index, class_mapping)

        if expected == 'wildboar' and predicted == 'wildboar':
            PT+=1
        elif expected == 'wildboar' and predicted == 'unknown':
            PF+=1
        elif expected == 'unknown' and predicted == 'unknown':
            NT+=1
        elif expected == 'unknown' and predicted == 'wildboar':
            NF+1 

        print(f"Predicted: '{predicted}', Expected: '{expected}', Probability: {probability:.2f}%")
        
        y_true.append(target_index)
        y_pred.append(class_mapping.index(predicted))
        probabilities_list.append(probability)

    f1 = f1_score(y_true, y_pred, average='weighted')
    average_probability = np.mean(probabilities_list)
    print(f"F1 Score: {(100*f1):.2f}%")
    print(f"Average Probability: {average_probability:.2f}%")
    plot_confusion_matrix(PT, NT, PF, NF)

    return 
#----------------------------------------------------------------------------------------------------------------
def plot_confusion_matrix(PT, NT, PF, NF):
    # Create a 2x2 confusion matrix
    confusion_matrix = np.array([[PT, PF], [NF, NT]])

    # Set the labels for the matrix
    labels = ['wildboar', 'unknown']

    # Create a figure and axis
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, cmap='Blues')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Set ticks and tick labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Loop over data dimensions and create text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, confusion_matrix[i, j],
                           ha="center", va="center", color="black")

    # Set title and labels
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    # Display the plot
    plt.show()
#----------------------------------------------------------------------------------------------------------------
def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # probabilities = torch.softmax(predictions, dim=1)[0]
        # probabilities = torch.sigmoid(predictions)[0]
        # print(torch.softmax(predictions, dim=1)[0])
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
        predictions= predictions.squeeze()
        probability = predictions[predicted_index].item() * 100
    return predicted, expected, probability
#----------------------------------------------------------------------------------------------------------------
# MAIN
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    os.mkdir(f"./loss/{save_dir}")

    animaltrain_dataset = SoundDataset(file_path=file_path, image=image, time=TIME, device=device, train="Train")
    animalvalid_dataset = SoundDataset(file_path=file_path, image=image, time=TIME, device=device, train="Valid")
    animaltest_dataset = SoundDataset(file_path=file_path, image=image, time=TIME, device=device, train='Test')
    train_loader = create_data_loader(animaltrain_dataset, BATCH_SIZE)
    valid_loader = create_data_loader(animalvalid_dataset, BATCH_SIZE)
    test_loader = create_data_loader(animaltest_dataset, batch_size=1)

    model = VGG_16().to(device)
    # summary(model, input_size=(1, 128, 345))
    print(model)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#----------------------------------------------------------------------------------------------------------------
# Train
    valid_loss_history = []
    train_loss_history = []

    valid_accuracy_history = []
    train_accuracy_history = []

    best_valid_loss = float('inf')
    best_model_params = None
    patience_counter = 0    

    for epoch in range(EPOCHS):
         # Training
        train_time = tqdm(
                          train_loader,
                          desc='train time',
                          mininterval=0.01,
                          total = int(animaltrain_dataset.__len__()/BATCH_SIZE)+1,
                        #   ascii = ' =', 
                          ncols = 100,  
                          leave = True
                          )
        train_acc, train_loss = train(model, train_time, criterion, optimizer)

        # Validation
        val_acc, val_loss = validate(model, valid_loader, criterion)

        # Print epoch results
        print(f"Epoch {epoch+1}/{EPOCHS}:")
        print(f"Train Loss: {train_loss:.4f} - Train Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_acc:.4f}")
        print()

        train_loss_history.append(train_loss)
        valid_loss_history.append(val_loss)
        train_accuracy_history.append(train_acc)
        valid_accuracy_history.append(val_acc)

        plt.plot(range(epoch + 1), train_loss_history, label='Train Loss')
        plt.plot(range(epoch + 1), valid_loss_history, label='Valid Loss')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss and Validation Loss and Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'./loss/{save_dir}/plot{epoch + 1}.png')
        plt.clf()

        # Early stopping check
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            best_model_params = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered. No improvement in {PATIENCE} epochs.")
                break
        train_time.close()
    torch.save(best_model_params, f"./saved_model/{save_dir}.pth")
    print(f"Best model parameters saved at {save_dir}.pth")

#----------------------------------------------------------------------------------------------------------------
    #Test
    test(cnn=model, test_loader=test_loader, save_dir=save_dir)
#----------------------------------------------------------------------------------------------------------------