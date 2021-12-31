import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        print(f"num_layers: {num_layers}, hidden_size: {hidden_size}, vocab_size: {vocab_size}\n"
              f"embed_size: {embed_size}")
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self.fc.bias.data.fill_(0)
        nn.init.uniform_(self.fc.weight, -1, 1)
    
    def forward(self, features, captions):
        print(f"features.shape: {features.shape}\ncaptions.shape: {captions.shape}")
        
        batch_size = captions.shape[0]
        caption_length = captions.shape[1]
        
        # Pass the captions through the embedding layer
        embedded_captions = self.embed(captions)
        
        # Add the variable dimension for the feature vector
        features = features.view(features.shape[0], 1, -1)
        
        # Create the full input tensor of 
        # batch_size x (feature_vector appended with captions) x embed_size
        inputs_cat = torch.cat([features, embedded_captions], dim=1).view(batch_size, -1, 
                                                                         self.embed_size)
        outputs, _ = self.lstm(inputs_cat)
        outputs = self.fc(outputs)
        # Return the outputs, leaving out the <end> token
        outputs = outputs[:, :-1, :]
        return outputs
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        current_input = inputs
        (hidden, cell) = (None, None)
        word_outputs = []
        for step in range(max_len):
            # Pass the input through the LSTM
            if hidden is not None and cell is not None:
                output, (hidden, cell) = self.lstm(current_input, (hidden, cell))
            else:
                output, (hidden, cell) = self.lstm(current_input)
            
            # Determine the output word as the word with the highest softmax score
            word_scores = self.fc(output)
            softmax_word_scores = nn.functional.softmax(word_scores, dim=2)
            output_word = torch.argmax(softmax_word_scores)
            # Pass the output word through the embedding layer to create the model's next input
            current_input = self.embed(output_word)
            current_input = torch.reshape(current_input, (1, 1, -1))
            # Append the output word index to the list of outputs
            output_word = output_word.tolist()
            word_outputs.append(output_word)
        
        return word_outputs
            