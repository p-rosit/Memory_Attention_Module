import torch
from torch import nn, optim
from attention_memory_layer import MemoryAttentionModule
from utils import generate_sequence, moving_average

from tqdm import tqdm
import matplotlib.pyplot as plt

class MAMnet(nn.Module):
    def __init__(self, size, hidden_size):
        super().__init__()
        self.linear_1 = nn.Linear(size, hidden_size)
        self.mam = MemoryAttentionModule(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, size)
        self.relu = nn.ReLU()

    def forward(self, x, memory):
        # x = self.relu(self.linear_1(x))
        x, memory, mam_mag = self.mam(x, memory)
        x = self.relu(x)
        # x = self.relu(self.linear_2(x))
        return x, memory, mam_mag

class LSTMnet(nn.Module):
    def __init__(self, size, hidden_size):
        super().__init__()
        self.linear_1 = nn.Linear(size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.linear_2 = nn.Linear(hidden_size, size)
        self.relu = nn.ReLU()

    def forward(self, x, hidden):
        # x = self.relu(self.linear_1(x))
        x, hidden = self.lstm(x, hidden)
        x = self.relu(x)
        # x = self.relu(self.linear_2(x))
        return x, hidden

def main():
    size = 4
    # hidden_size = 20
    hidden_size = 4
    min_length, max_length = 10, 15
    min_repeat, max_repeat = 2, 10

    lstm_mem_size = 10
    mam_mem_height = 4
    mam_mem_length = 20

    lstm = LSTMnet(size, hidden_size)
    mam = MAMnet(size, hidden_size)

    epochs = 1000
    batch_size = 256
    lstm_max_grad = 100
    mam_max_grad = 10
    criterion = nn.MSELoss()
    optimizer = optim.AdamW((*lstm.parameters(), *mam.parameters()), lr=1e-3)

    lstm_history = torch.zeros(epochs)
    mam_history = torch.zeros(epochs)
    mam_mags = []
    for ep in tqdm(range(epochs)):
        seq_length = 10  # torch.randint(min_length, max_length, (1,))
        seq_repeats = 3  # torch.randint(min_repeat, max_repeat, (1,))
        batch_seq = generate_sequence(batch_size, size, seq_length + 1)

        lstm_hidden = (torch.randn(1, batch_size, hidden_size), torch.randn(1, batch_size, hidden_size))
        mam_memory = 1e-2 * torch.randn(batch_size, mam_mem_length, mam_mem_height)
        lstm_loss = torch.zeros(1)
        mam_loss = torch.zeros(1)
        mem_loss = torch.zeros(1)
        for i in range(seq_length + 1):
            lstm_last, lstm_hidden = lstm(batch_seq[:, :, i].unsqueeze(1), lstm_hidden)
            mam_last, mam_memory, mam_mag = mam(batch_seq[:, :, i], mam_memory)
            # mem_loss += 1e-6 * mam_memory.abs().sum()
            mam_mags.append(mam_mag)
            # print(lstm_last)
            # print(mam_last)

        # if lstm_last.isnan().any() or mam_last.isnan().any():
        #     print(batch_seq)
        #     print(lstm_last)
        #     print(mam_last)
        #     error(':)')

        for _ in range(seq_repeats):
            for i in range(seq_length):
                lstm_loss += criterion(lstm_last.squeeze(), batch_seq[:, :, i])
                mam_loss += criterion(mam_last, batch_seq[:, :, i])

                lstm_last, lstm_hidden = lstm(batch_seq[:, :, i].unsqueeze(1), lstm_hidden)
                mam_last, mam_memory, mam_mag = mam(batch_seq[:, :, i], mam_memory)
                # mem_loss += 1e-6 * mam_memory.abs().sum()
                mam_mags.append(mam_mag)
                # print(lstm_last)
                # print(mam_last)

        lstm_loss += criterion(lstm_last.squeeze(), batch_seq[:, :, -2])
        mam_loss += criterion(mam_last, batch_seq[:, :, -2])
        total_loss = lstm_loss + mam_loss + mem_loss
        lstm_history[ep] = lstm_loss.item() / (seq_length * seq_repeats * size)
        mam_history[ep] = mam_loss.item() / (seq_length * seq_repeats * size)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(lstm.parameters(), lstm_max_grad)
        torch.nn.utils.clip_grad_norm_(mam.parameters(), mam_max_grad)
        optimizer.step()

    print(mam_memory)

    plt.plot(mam_mags)
    plt.show()

    # plt.plot(moving_average(lstm_history))
    # plt.plot(moving_average(mam_history))
    print(lstm_history)
    print(mam_history)
    plt.plot(torch.arange(epochs), lstm_history)
    plt.plot(torch.arange(epochs), mam_history)
    plt.legend(['LSTM', 'MAM'])
    plt.show()

if __name__ == '__main__':
    main()
