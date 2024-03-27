import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, conv_channels, conv_kernel_size, lstm_hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        print("input_dim ", input_dim)
        print("embedding_dim ", embedding_dim)
        self.conv_stack = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=conv_channels, kernel_size=conv_kernel_size, padding=(conv_kernel_size - 1) // 2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=conv_kernel_size, padding=(conv_kernel_size - 1) // 2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=conv_kernel_size, padding=(conv_kernel_size - 1) // 2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU()
        ])
        self.lstm = nn.LSTM(conv_channels, lstm_hidden_size, bidirectional=True)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # Reshape for Conv1D: (batch_size, embedding_dim, seq_len)
        for layer in self.conv_stack:
            x = layer(x)
        x = x.permute(0, 2, 1)  # Reshape for LSTM: (batch_size, seq_len, conv_channels)
        encoder_output, (hidden, cell) = self.lstm(x)
        print("Encoder Output Shape:", encoder_output.shape)
        return encoder_output, hidden, cell

class Attention(nn.Module):
    def __init__(self, lstm_hidden_size, attention_hidden_size, encoder_output_size):
        super(Attention, self).__init__()
        self.W1 = nn.Linear(lstm_hidden_size * 2, attention_hidden_size)
        self.W2 = nn.Linear(encoder_output_size, attention_hidden_size)
        self.V = nn.Linear(attention_hidden_size, 1)

    def forward(self, decoder_hidden, encoder_output):
        decoder_hidden = decoder_hidden.squeeze(1)  # Squeeze to remove the extra dimension
        decoder_hidden = decoder_hidden.view(decoder_hidden.size(0), -1) 
        print("Decoder Hidden Shape:", decoder_hidden.shape)
        print("Encoder Output Shape:", encoder_output.shape)
        print("W1 Weight Shape:", self.W1.weight.shape)
        print("W2 Weight Shape:", self.W2.weight.shape)
        energy = torch.tanh(self.W1(decoder_hidden) + self.W2(encoder_output))
        attention_weights = F.softmax(self.V(energy), dim=1)
        context_vector = torch.sum(attention_weights * encoder_output, dim=1)

        
        return context_vector, attention_weights



class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, lstm_hidden_size, attention_hidden_size, encoder_output_size, conv_channels):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.prenet = nn.Sequential(
            nn.Linear(embedding_dim + encoder_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(256, lstm_hidden_size, batch_first=True)
        self.attention = Attention(lstm_hidden_size, attention_hidden_size, encoder_output_size)
        self.fc_out = nn.Linear(lstm_hidden_size + encoder_output_size, output_dim)

    def forward(self, x, encoder_output, hidden, cell):
        x = x.long()
        x = self.embedding(x)
        print("Decoder Input Shape:", x.shape)
        context_vector, _ = self.attention(hidden[0], encoder_output)  # Ensure hidden has the correct shape
        print("Context Vector Shape:", context_vector.shape)
        x = torch.cat((context_vector.unsqueeze(1), x), -1)
        print("Input to Prenet Shape:", x.shape)
        x = self.prenet(x)
        print("Output from Prenet Shape:", x.shape)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        print("LSTM Output Shape:", output.shape)
        prediction = self.fc_out(torch.cat((output.squeeze(1), context_vector), -1))
        print("Decoder Output Shape:", prediction.shape)
        return prediction, hidden, cell
    

    # def forward(self, x, encoder_output, hidden, cell):
    #     x = x.long()
    #     x = self.embedding(x)
    #     print("Decoder Input Shape:", x.shape)
    #     context_vector, _ = self.attention(hidden, encoder_output)  # Ensure hidden has the correct shape
    #     print("Context Vector Shape:", context_vector.shape)
    #     x = torch.cat((context_vector.unsqueeze(1), x), -1)
    #     print("Input to Prenet Shape:", x.shape)
    #     x = self.prenet(x)
    #     print("Output from Prenet Shape:", x.shape)
    #     output, (hidden, cell) = self.lstm(x, (hidden, cell))
    #     print("LSTM Output Shape:", output.shape)
    #     prediction = self.fc_out(torch.cat((output.squeeze(1), context_vector), -1))
    #     print("Decoder Output Shape:", prediction.shape)
    #     return prediction, hidden, cell



class Tacotron2(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim, conv_channels, conv_kernel_size, lstm_hidden_size, attention_hidden_size):
        super(Tacotron2, self).__init__()
        self.encoder = Encoder(input_dim, embedding_dim, conv_channels, conv_kernel_size, lstm_hidden_size)
        self.decoder = Decoder(output_dim, embedding_dim, lstm_hidden_size, attention_hidden_size, lstm_hidden_size * 2, conv_channels)

    def forward(self, text, speech, teacher_forcing_ratio=0.5):
        batch_size = text.shape[0]
        max_len = speech.shape[1]
        output_dim = speech.shape[2]

        encoder_output, hidden, cell = self.encoder(text)

        outputs = torch.zeros(batch_size, max_len, output_dim).to(text.device)
        input = torch.tensor([0] * batch_size).unsqueeze(1).to(text.device)  # Start token index
        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, encoder_output, hidden, cell)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = speech[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        return outputs

if __name__ == "__main__":

    # Define model hyperparameters
    input_dim = 100  # Example input dimension (vocabulary size)
    output_dim = 80  # Example output dimension (mel spectrogram channels)
    embedding_dim = 512
    conv_channels = 512
    conv_kernel_size = 5
    lstm_hidden_size = 512
    attention_hidden_size = 128

    # Instantiate the model
    model = Tacotron2(input_dim, output_dim, embedding_dim, conv_channels, conv_kernel_size, lstm_hidden_size, attention_hidden_size)

    # Generate random input tensors for testing
    batch_size = 2
    max_text_length = 20
    max_speech_length = 100
    text_input = torch.randint(0, input_dim, (batch_size, max_text_length))  # Random text input tensor
    speech_input = torch.rand(batch_size, max_speech_length, output_dim)  # Random speech input tensor

    with torch.no_grad():
        model.eval()
        encoder_output, hidden, cell = model.encoder(text_input)
        print("Shapes of input tensors:")
        print("Text Input Shape:", text_input.shape)
        print("Speech Input Shape:", speech_input.shape)
        print("Encoder Output Shape:", encoder_output.shape)
        for t in range(max_speech_length):
            print("Decoder Input Step", t+1)
            output, hidden, cell = model.decoder(speech_input[:, t].unsqueeze(1), encoder_output, hidden, cell)

    # Inspect the output
    print("Output shape:", output.shape)