import torch
import torchaudio
from bainarydatasest import SoundDataset
# from seperble import CNN
from sklearn.metrics import f1_score
import numpy as np
# from models.conformer import Conformer
# from models.crnn import CRNN
from models.vgg16 import VGG_16
from models.transepertrain import TransferTrainModel
import matplotlib.pyplot as plt
import torch

TIME = 4
load_file = "1channel_vgg16_128.173_train_standconv"
image=False
class_mapping = [
    'wildboar',
    'unknown'
]
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

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        input = input.to(model.device)
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

if __name__ == "__main__":
    cnn = VGG_16().to("cuda")
    state_dict = torch.load(f"./saved_model/{load_file}.pth")
    cnn.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn.device = device  # Add 'device' attribute to the model

    print(f"Using device {device}")
    file_path = 'binary_class.csv'
    animal_dataset = SoundDataset(file_path=file_path, image=image, time=TIME, device=device, train='Test')

    y_true = []
    y_pred = []
    probabilities_list = []
    PT=0
    PF=0
    NT=0
    NF=0
    for i in range(animal_dataset.__len__()):
        input, target = animal_dataset[i][0], animal_dataset[i][1]
        # print(input.shape)
        target_index = target.nonzero(as_tuple=False).item()
        input_tensor = torch.Tensor(input)
        input_tensor.unsqueeze_(0)

        predicted, expected, probability = predict(cnn, input_tensor, target_index, class_mapping)
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
