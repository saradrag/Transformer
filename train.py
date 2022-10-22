import torch
from torch import nn
import matplotlib.pyplot as plt
from transformer import Transformer
from dataloader import training_loader
from parameters import NUM_EPOCHS
import tqdm

tr_loss_list = []
val_loss_list = []


def train_fn(model, loader, optimizer, loss_fn):
    total_loss = 0.0
    for batch in tqdm.tqdm(loader):
        encoder_input, decoder_input, decoder_output = batch
        optimizer.zero_grad()
        output = model(encoder_input, decoder_input)
        loss = loss_fn(output, decoder_output)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        tr_loss_list.append(loss.item())
        torch.save(model.state_dict(), 'model/version_1.pt')
        with open("loss.txt", "a") as file_object:
            file_object.write(f"\n{loss.item()}")
    return total_loss / len(loader)


def valid_fn(model, loader, loss_fn):
    total_loss = 0.0
    for batch in loader:
        encoder_input, decoder_input, labels = batch
        output = model(encoder_input, decoder_input)
        prediction = torch.argmax(output[0], dim=-1)
        print(f"in validation: input: {encoder_input[0]}, prediction: {prediction}, label: {labels[0]}")
        output = torch.log(output)
        output = torch.transpose(output, -1, -2)
        labels = labels.to(torch.long)
        loss = loss_fn(output, labels)
        total_loss += loss.item()
        val_loss_list.append(loss.item())
    return total_loss / len(loader)


model = Transformer()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
for i in range(NUM_EPOCHS):
    loss = train_fn(model, training_loader, optimizer, loss_fn)
    print(f"Epoch {i}: training loss: {loss}")
    #model.eval()
    #loss = valid_fn(model, validation_loader, loss_fn)
    #print(f"Epoch {i}: validation loss: {loss}")

plt.plot(tr_loss_list)
plt.show()
#plt.plot(val_loss_list)
#plt.show()
