import torch
import torch.nn as nn
from torch.nn import functional as F

class SpeechEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_size, num_layers=1, dropout=0.1):
        super(SpeechEncoder, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, embedding_dim)

    def forward(self, input):
        output, (hidden, cell) = self.rnn(input)
        print("-- output --", output.shape)
        encoded_output = self.fc(hidden[-1])  # Use the hidden state of the last layer
        print("-- encoded_output --", encoded_output.shape)
        return encoded_output

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1, dropout=0.1):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)

    def forward(self, input):
        embedded = self.embedding(input)
        output, (hidden, cell) = self.rnn(embedded)
        return output, (hidden, cell)

class CustomTacotron2Decoder(nn.Module):
    def __init__(self, speech_embedding_dim, word_embedding_dim, decoder_input_size, decoder_hidden_size):
        super(CustomTacotron2Decoder, self).__init__()
        self.speech_embedding_dim = speech_embedding_dim
        self.word_embedding_dim = word_embedding_dim
        
        # Dummy decoder
        self.decoder_rnn = nn.LSTMCell(decoder_input_size, decoder_hidden_size)
        self.fc_output = nn.Linear(decoder_hidden_size, output_size)

    def forward(self, speech_embeddings, word_embeddings, memory, memory_lengths):
        # Dummy decoder forward pass
        decoder_hidden = (torch.zeros(speech_embeddings.size(0), decoder_hidden_size), 
                          torch.zeros(speech_embeddings.size(0), decoder_hidden_size))
        
        print("1speech_embeddings size: ",speech_embeddings.size())
        print("1word_embeddings size: ", word_embeddings.size())

        # Unsqueeze speech embeddings along a new dimension to match the sequence length of word embeddings
        # speech_embeddings = speech_embeddings.unsqueeze(1).expand(-1, word_embeddings.size(1), -1)
        speech_embeddings = speech_embeddings.unsqueeze(1)
        speech_embeddings = speech_embeddings.expand(-1, word_embeddings.size(1), -1)

        word_embeddings = word_embeddings[:, :, :speech_embeddings.size(2)]


        print("2---speech_embeddings---", speech_embeddings.size())

        print("2word_embeddings size:", word_embeddings.size())  


        # Concatenate the embeddings
        # combined_embeddings = torch.cat((speech_embeddings, word_embeddings), dim=-1)
        combined_embeddings = torch.cat((speech_embeddings, word_embeddings), dim=-1)

        # Print the sizes of input tensors
        print("combined_embeddings size:", combined_embeddings.size())  
        print("decoder_hidden size:", decoder_hidden[0].size()) 

        decoder_outputs = []
        for i in range(combined_embeddings.size(1)):
            decoder_hidden = self.decoder_rnn(combined_embeddings[:, i], decoder_hidden)
            decoder_output = self.fc_output(decoder_hidden[0])
            decoder_outputs.append(decoder_output)
        
        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        
        return decoder_outputs, None

# Instantiate the speech encoder
input_dim = 40  # Example input dimension of speech features (e.g., MFCCs)
speech_embedding_dim = 64  # Example dimension of speech embeddings
hidden_size_encoder = 128
speech_encoder = SpeechEncoder(input_dim, speech_embedding_dim, hidden_size_encoder)

# parameters for text encoder
vocab_size = 10000  # Example vocabulary size
embedding_dim = 100
hidden_size = 128
num_layers = 2
dropout = 0.1

# Instantiate the text encoder
text_encoder = TextEncoder(vocab_size, embedding_dim, hidden_size, num_layers, dropout)

# Instantiate the custom Tacotron 2 decoder
word_embedding_dim = 64
decoder_input_size = speech_embedding_dim + word_embedding_dim
print("---- decoder_input_size -----", decoder_input_size)
decoder_hidden_size = 128
output_size = 64

custom_decoder = CustomTacotron2Decoder(speech_embedding_dim, word_embedding_dim, decoder_input_size, decoder_hidden_size)

# Example inputs
text_input = torch.tensor([[1, 2, 3, 4, 5]])  # Example input sequence
speech_input = torch.randn(1, 10, input_dim)  # Example speech input with 10 frames and input_dim features

# Forward pass through speech encoder
speech_embedding = speech_encoder(speech_input)
print("------speech_embedding", speech_embedding.shape)

# Forward pass through text encoder
encoder_output, encoder_hidden = text_encoder(text_input)
print("encoder_output", encoder_output.shape)

# Forward pass through custom decoder
decoder_output, _ = custom_decoder(speech_embedding, encoder_output, None, None)

# Print shapes of decoder output
print("Decoder Output Shape:", decoder_output.shape)
