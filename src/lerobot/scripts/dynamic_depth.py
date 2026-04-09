import torch
import torch.nn as nn

class DynamicDepth(nn.Module):
    def __init__(self):
        super(DynamicDepth, self).__init__()
        self.linear_1 = nn.Linear(32, 16)
        self.linear_2 = nn.Linear(16, 8)
        self.linear_3 = nn.Linear(8, 1)
    
    def forward(self, x, sample_rate):
        if sample_rate > 32:
            x = x[:,-32:]
            x = self.linear_1(self.linear_2(self.linear_3(x)))
        elif sample_rate > 16:
            x = x[:,-16:]
            x = self.linear_2(self.linear_3(x))
        else:
            x = x[:,-8:]
            print(x.shape)
            x = self.linear_3(x)
        return x
    
def main():
    model = DynamicDepth()
    x = torch.randn((32, 16))
    sample_rate = 16

    output = model(x, sample_rate)
    print(output)

if __name__ == "__main__":
    main()