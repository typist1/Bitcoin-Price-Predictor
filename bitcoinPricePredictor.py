#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

data = pd.read_csv('BTC-USD.csv')


# In[4]:


data


# In[5]:


data = data[['Date', 'Close']]
data


# In[6]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


# In[11]:


data['Date'] = pd.to_datetime(data['Date'])

plt.plot(data['Date'], data['Close'])


# In[97]:


from copy import deepcopy as dc

def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)

    df.set_index('Date', inplace=True)

    for i in range(1, n_steps+1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)

    df.dropna(inplace=True)

    return df

lookback = 30
shifted_df = prepare_dataframe_for_lstm(data, lookback)
shifted_df


# In[98]:


shifted_df_as_np = shifted_df.to_numpy()
shifted_df_as_np


# In[99]:


#scaled the data into a range between -1 and 1
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)

shifted_df_as_np


# In[100]:


x = shifted_df_as_np[:, 1:]
y = shifted_df_as_np[:, 0]

#want to flip t-1 t-2 etc to go in reverse, t-7 t-6 becuase for the LSTM model you want to go from the oldest to most recent data
x = dc(np.flip(x, axis=1))


# In[101]:


split_index = int(len(x) * .95)
split_index


# In[102]:


x_train, x_test = x[:split_index], x[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[103]:


x_train = x_train.reshape((-1, lookback, 1))
x_test = x_test.reshape((-1, lookback, 1))

y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[104]:


#convert to PyTorch Tensors
x_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train).float()
x_test = torch.tensor(x_test).float()
y_test = torch.tensor(y_test).float()

x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[105]:


#make dataset
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

train_dataset = TimeSeriesDataset(x_train, y_train)
test_dataset = TimeSeriesDataset(x_test, y_test)


# In[106]:


#make dataloader
from torch.utils.data import DataLoader

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# In[107]:


for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    print(x_batch.shape, y_batch.shape)
    break


# In[108]:


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model = LSTM(1, 4, 1)
model.to(device)
model


# In[109]:


def train_one_epoch():
    model.train(True)
    running_loss = 0.0
    print(f'Epoch: {epoch+1}')
    for batch_index, batch in enumerate(train_loader):

        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output=model(x_batch)
        loss= loss_function(output, y_batch)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:
            avg_loss_across_batches = running_loss / 100
            print(f'Batch: {batch_index+1}, Loss: {avg_loss_across_batches}')

            running_loss = 0.0
    print()


# In[110]:


def validate_one_epoch():
    model.train(False)
    running_loss=0.0

    for batch_index,batch in enumerate(test_loader):
        x_batch, y_batch = batch[0], batch[1]

        with torch.inference_mode():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss +=loss.item()

    avg_loss_across_batches = running_loss/ len(test_loader)

    print(f'Val loss: {avg_loss_across_batches:.5f}')
    print("********************************")
    print()


# In[111]:


learning_rate = 0.001
num_epochs = 100
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range (num_epochs):
    train_one_epoch()
    validate_one_epoch()


# In[90]:


with torch.inference_mode():
    predicted = model(x_train).numpy()

plt.plot(y_train, label='Actual Close')
plt.plot(predicted, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()


# In[91]:


train_predictions = predicted.flatten()
#need to do inverse transform
dummies = np.zeros((x_train.shape[0], lookback+1))
dummies[:, 0] = train_predictions
dummies = scaler.inverse_transform(dummies)

train_predictions = dc(dummies[:, 0])
train_predictions


# In[92]:


#need to do inverse transform
dummies = np.zeros((x_train.shape[0], lookback+1))
dummies[:, 0] = y_train.flatten()
dummies = scaler.inverse_transform(dummies)

new_y_train = dc(dummies[:, 0])
new_y_train


# In[93]:


with torch.inference_mode():
    predicted = model(x_train).numpy()

plt.plot(new_y_train, label='Actual Close')
plt.plot(train_predictions, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()


# In[94]:


test_predictions = model(x_test).detach().numpy().flatten()
dummies = np.zeros((x_test.shape[0], lookback+1))
dummies[:, 0] = test_predictions
dummies = scaler.inverse_transform(dummies)

test_predictions = dc(dummies[:, 0])
test_predictions


# In[95]:


dummies = np.zeros((x_test.shape[0], lookback+1))
dummies[:, 0] = y_test.flatten()
dummies = scaler.inverse_transform(dummies)

new_y_test = dc(dummies[:, 0])
new_y_test


# In[96]:


plt.plot(new_y_test, label='Actual Close')
plt.plot(test_predictions, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()

plt.show()
# In[ ]:




