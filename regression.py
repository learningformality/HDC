import torch
import torch.nn as nn
import torchmetrics
import torchhd
import pandas as pd
from sklearn import preprocessing
from torchhd.models import Centroid
from torchhd import embeddings
from tqdm import tqdm

if __name__ == "__main__":

    class CustomDataset(torch.utils.data.Dataset):
    
        def __init__(self, inputs, labels):

            self.inputs = inputs
            self.labels = labels

        def __len__(self):

            return len(self.inputs)

        def __getitem__(self, idx):

            input_tensor = self.inputs[idx]
            label_tensor = self.labels[idx]
            return input_tensor, label_tensor
        
    class Encoder(nn.Module):

        def __init__(self, out_features, size, levels):

            super(Encoder, self).__init__()
            self.flatten = torch.nn.Flatten()
            self.position = embeddings.Random(size, out_features)
            self.value = embeddings.Level(levels, out_features)

        def forward(self, x):

            x = self.flatten(x)
            sample_hv = torchhd.bind(self.position.weight, self.value(x))
            sample_hv = torchhd.multiset(sample_hv)
            return torchhd.hard_quantize(sample_hv)


    # Use the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))

    DIMENSIONS = 10000
    NUM_LEVELS = 1000
    BATCH_SIZE = 100
    features = 528

    data_df = pd.read_csv("door_training.csv", index_col=0)

    size = int(data_df.shape[0])
    scaler = preprocessing.MinMaxScaler()
    norm_df = scaler.fit_transform(data_df)
    data_df = pd.DataFrame(norm_df, columns=data_df.columns)
    labindex = data_df.columns[-1]

    train_X, train_Y = data_df.iloc[:, 0:features], data_df.iloc[:, -1]
    train_Y = train_Y.to_numpy().astype(float)
    train_X = train_X.to_numpy().astype(float)
    train_Y = train_Y.flatten()
    inputs = torch.tensor(train_X).float()
    labels = torch.tensor(train_Y).float()
    torch_train = CustomDataset(inputs, labels)

    data_df[labindex] = pd.Categorical(data_df[labindex])

    train_ds, test_ds = torch.utils.data.random_split(torch_train, [int(9 * size/10), int(size/10) + 1])

    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, prefetch_factor=4, persistent_workers=True, pin_memory=True)
    test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print("---Data processing finished---")

    encode = Encoder(DIMENSIONS, features, NUM_LEVELS)
    encode = encode.to(device)

    num_classes = 1
    model = Centroid(DIMENSIONS, num_classes)
    model = model.to(device)

    with torch.no_grad():

        for samples, labels in tqdm(train_ld, desc="Training"):

            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            model.add(samples_hv, labels)

    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

    with torch.no_grad():

        model.normalize()

        for samples, labels in tqdm(test_ld, desc="Testing"):

            samples = samples.to(device)

            samples_hv = encode(samples)
            outputs = model(samples_hv, dot=True)
            accuracy.update(outputs.cpu(), labels)

    print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")