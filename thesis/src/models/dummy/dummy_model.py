import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self, hidden_size=768, num_labels=2, vocab_size=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 16)
        self.linear = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input_ids, attention_mask = None, labels=None, **kwargs):
        if attention_mask:
            input_ids = input_ids[attention_mask]
        input_ids = self.embedding(input_ids)
        input_ids = input_ids.mean(dim=1)
        logits = self.linear(self.dropout(input_ids))
        logits = self.softmax(logits) 
        return logits
        
        
#%%
# %%
#%%
# from torch.utils.data import Dataset
# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, random_split

# class Data(Dataset):
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#     def __len__(self):
#         return len(self.x)
#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]
    

# num_examples = 1000
# max_len = 5
# vocab_size = 100
# hidden_dim = 16
# embedding_dim = 8
# output_dim = 2

# x_low = torch.randint(0, 40, (num_examples, max_len))
# y_low = torch.zeros(x_low.shape[0], dtype=torch.long)
# x_high = torch.randint(30, 70, (num_examples, max_len))
# y_high = torch.ones(x_high.shape[0], dtype=torch.long)
# x = torch.cat((x_low, x_high))
# y = torch.cat((y_low, y_high))
# dataset = Data(x,y)
# train, eval = random_split(dataset, [0.8, 0.2]) 
# for i in train:
#     print(i)
#     break

# model = DummyModel(hidden_dim)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# for _ in range(10):
#     model.train()
#     for i, (x, y) in enumerate(torch.utils.data.DataLoader(train, batch_size=16, shuffle=True)):
#         optimizer.zero_grad()
#         out = model(x)
#         loss = criterion(out, y)
#         if i % 100 == 0:
#             print(loss)
#         loss.backward()
#         optimizer.step()
    
#     model.eval()
#     with torch.no_grad():
#         correct = 0
#         for x, y in torch.utils.data.DataLoader(train, batch_size=16, shuffle=False):
#                 out = model(x)
#                 predictions = torch.argmax(out, dim=1)
#                 correct += torch.sum(predictions == y)
#         print(f"Train accuracy: {correct/len(train)}")
#         correct = 0
#         for x, y in torch.utils.data.DataLoader(eval, batch_size=16, shuffle=False):
#                 out = model(x)
#                 predictions = torch.argmax(out, dim=1)
#                 correct += torch.sum(predictions == y)
#         print(f"Eval Accuracy: {correct/len(eval)}")