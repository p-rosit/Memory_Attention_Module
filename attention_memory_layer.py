import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, input, hidden_sizes, output):
        super().__init__()
        layer_sizes = (input, *hidden_sizes, output)
        layers = []
        for layer_in, layer_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(layer_in, layer_out))
            layers.append(nn.ReLU())
        layers.pop(-1)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MemoryAttentionModule(nn.Module):
    def __init__(self, input_size, output_size, memory_size=4):
        super().__init__()
        self.normalize_commit = nn.Sigmoid()
        self.forget_net = MLP(input_size + memory_size, [], 1)
        self.remember_net = MLP(input_size + memory_size, [], memory_size + 1)

        self.normalize_recall = nn.Softmax(dim=1)
        self.active_recall_net = MLP(input_size, [], memory_size)
        self.passive_recall_net = MLP(memory_size, [], 1)

        self.layer = nn.Linear(input_size + 2 * memory_size, output_size)

    def forward(self, x, memory):
        x = x.unsqueeze(dim=1)

        # Active recall
        mem_score = self.active_recall_net(x)
        mem_score = memory * mem_score
        mem_score = mem_score.sum(dim=-1)
        mem_score = self.normalize_recall(mem_score).unsqueeze(dim=-1)
        active_recall = (memory * mem_score).sum(dim=1)

        # Passive recall
        mem_score = self.normalize_recall(self.passive_recall_net(memory))
        passive_recall = (memory * mem_score).sum(dim=1)

        # Compute output
        intermediate = torch.cat((x.squeeze(), active_recall, passive_recall), dim=1)
        output = self.layer(intermediate)

        # Forgetting and committing to memory
        x = x.repeat(1, memory.size(1), 1)
        inp_mem_stack = torch.cat((x, memory), dim=2)

        # Forget mechanism
        forget_score = 1.1 * self.normalize_commit(self.forget_net(inp_mem_stack))
        memory *= forget_score

        # Remember mechanism
        remember_res = self.remember_net(inp_mem_stack)
        modification_to_mem = remember_res[:, :, :-1]
        modification_scale = remember_res[:, :, -1].unsqueeze(-1)

        memory += modification_scale * modification_to_mem

        return output, memory

if __name__ == '__main__':
    input_size = 3
    output_size = 3
    memory_length = 10
    memory_size = 4

    inp = torch.zeros((16, input_size))
    mem = torch.zeros((16, memory_length, memory_size))
    mam = MemoryAttentionModule(input_size, output_size, memory_size=memory_size)

    out, mem = mam(inp, mem)

    print(out.size())
