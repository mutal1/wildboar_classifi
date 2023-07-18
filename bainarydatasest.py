#----------------------------------------------------------------------------------------------------------------
#import
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from scipy.io import wavfile
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchaudio
import pandas as pd
from scipy import ndimage
#----------------------------------------------------------------------------------------------------------------
# Dataset object creation

class SoundDataset(Dataset):
#---------------------------------------------------------------------------------------------------------------- 
    def __init__(self, file_path,time,device,image = False,cut_samples = 44100 ,target_sample_rate=44100 ,transformation=True ,train = "Train"):
        
        self.df = pd.read_csv(file_path)
        self.train = train
        if train == "Train":
            self.data_path = self.df.loc[self.df['path'].str.contains('train')]
        elif train =='Valid':
            self.data_path = self.df.loc[self.df['path'].str.contains('valid')]
        elif train =='Test':
            self.data_path = self.df.loc[self.df['path'].str.contains('test')]
        self.data_path = self.data_path.reset_index(drop=True)

        self.time = time
        self.device = device
        self.target_sample_rate = target_sample_rate
        self.max_len = cut_samples * self.time
        self.image = image

        if transformation is True :
            self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100,
            n_fft=2048,
            hop_length=1024,
            n_mels=128,
            f_min = 20,
            f_max = 8300,
        ).to(self.device)
#----------------------------------------------------------------------------------------------------------------
    def get_melspec_torch(self,signal):
        
        signal = self.mel_spectrogram(signal)
        power_to_db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
        mel_spec_db = power_to_db_transform(signal)

        if self.image == False:
            mel_spec_db = mel_spec_db.cpu().numpy()

        if self.image == True:
            mel_spec_db = mel_spec_db.squeeze(dim=0)
            mel_spec_db = mel_spec_db.cpu().numpy()
            desired_size = (299, 299)
            mel_spec_db = ndimage.zoom(mel_spec_db, (desired_size[0] / mel_spec_db.shape[0], desired_size[1] / mel_spec_db.shape[1]))

        return mel_spec_db
#----------------------------------------------------------------------------------------------------------------
    def spec_to_image(self,spec):
        eps=1e-6

        # Z-score normalization
        mean = spec.mean()
        std = spec.std()
        spec_norm = (spec - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()

        # Min-max scaling
        spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
        spec_scaled = spec_scaled.astype(np.uint8)
        if self.image == True:
            spec_scaled = torch.Tensor(spec_scaled).unsqueeze(0)
        return spec_scaled
#----------------------------------------------------------------------------------------------------------------
    def get_labels(self,item):
        class_name = self.data_path.loc[item]['class_name']
        if class_name == 'wildboar':
            label = torch.tensor([1,0])
        elif class_name == 'unknown':
            label = torch.tensor([0,1]) 
        return label
#----------------------------------------------------------------------------------------------------------------    
    def __len__(self):

        pig_len = self.data_path.loc[self.data_path['class_name'] == 'wildboar'].shape[0]
        unknown_len = self.data_path.loc[self.data_path['class_name'] == 'unknown'].shape[0]

        return self.data_path.shape[0]
#----------------------------------------------------------------------------------------------------------------    
    def __getitem__(self, item):
        file_path = self.data_path.loc[item]['path']
        label = self.get_labels(item)
        signal, sr = torchaudio.load(file_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.get_melspec_torch(signal)            
        signal = self.spec_to_image(signal)
    
        return signal, label
#----------------------------------------------------------------------------------------------------------------            
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal
#----------------------------------------------------------------------------------------------------------------    
    def _mix_down_if_necessary(self, signal):
        # signal = (channels, num_samples) -> (2, 16000) -> (1, 16000)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
#----------------------------------------------------------------------------------------------------------------    
    def _cut_if_necessary(self, signal):
        # signal -> (1, num_sample)
        if signal.shape[1] > self.max_len:
            signal = signal[:, :(self.max_len)]
        return signal 
#----------------------------------------------------------------------------------------------------------------    
    def _right_pad_if_necessary(self, signal):
        len_signal = signal.shape[1]
        if len_signal < (self.max_len): # apply right pad
            pad_missing_samples = (self.max_len) - len_signal
            last_dim_padding = (0, pad_missing_samples)
            signal = F.pad(signal, last_dim_padding)#value=1e-10
        return signal
#----------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":        
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    data_dir = "./binarysoundset"
    file_path = './binary_class.csv'
#----------------------------------------------------------------------------------------------------------------
# Object dataset
# train_dataset = SoundDataset(data_dir,train="Train",device=device)
# valid_dataset = SoundDataset(file_path=file_path, data_dir=data_dir ,train="Valid",device=device)
# test_dataset = SoundDataset(file_path=file_path,time=4, image=False, train="Train",device=device)
#----------------------------------------------------------------------------------------------------------------
# file_len
# print(train_dataset.__len__())
# print(valid_dataset.__len__())
# print(test_dataset.__len__())
#----------------------------------------------------------------------------------------------------------------
# test code
# hist_list = []
# valid = DataLoader(test_dataset,batch_size=1,shuffle=True)
# test = DataLoader(test_dataset,batch_size=1,shuffle=False)
# k=0
# for i,(log_mel_spectrogram,label) in enumerate(test):
    # if label[0][0] == 0:
        # k+=1
    # print(i,label)
    # print(log_mel_spectrogram.shape,label.shape)
    # hist_list.append(log_mel_spectrogram.shape[3])
    # hist_list.append(log_mel_spectrogram[3])
#    print(i)
    # plt.figure(figsize=(10, 4))
    # plt.imshow(log_mel_spectrogram.cpu().squeeze().numpy(), origin='lower', aspect='auto', cmap='inferno')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel Spectrogram')
    # plt.xlabel('Frame')
    # plt.ylabel('Mel Filter')
    # plt.tight_layout()
    # plt.show()
#----------------------------------------------------------------------------------------------------------------
# histgram
# print(hist_list)
# plt.hist(hist_list,bins=len(hist_list))
# counts, edges = np.histogram(hist_list, bins=len(hist_list))
# max_index = np.argmax(counts)
# x_at_max = edges[max_index], edges[max_index + 1]
# print("X at max count:", x_at_max)
# plt.xlabel('Value')
# plt.ylabel('count')
# plt.title('Histogram')
# plt.show()
# X at max count: (65.65367965367966, 71.1103896103896) 1[sec] = 87[frame] 
#----------------------------------------------------------------------------------------------------------------