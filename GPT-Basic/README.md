# GPT (Generative Pretrained Model) Overview

GPT (Generative Pre-trained Transformer) is a type of language model that utilizes the Transformer architecture to generate human-like text. It is trained on a large corpus of text data and can be fine-tuned for specific natural language processing tasks. 

The Transformer architecture, proposed in the paper "Attention Is All You Need," revolutionized natural language processing tasks by introducing a self-attention mechanism. It has been widely adopted for various sequence-to-sequence tasks, including machine translation and language generation.

### Building The GPT

Let's explore the key components of the Transformer architecture using some Python code snippets:

1. Self-Attention Mechanism:

   The self-attention mechanism allows the model to weigh the importance of different words in the input sequence when generating the output. It
   computes attention scores between each word and all other words in the sequence, capturing the relationships and dependencies. Here's a Python
   implementation of self-attention:

   ```
   import torch
   import torch.nn as nn
   
   class SelfAttention(nn.Module):
       def __init__(self, hidden_size):
           super(SelfAttention, self).__init__()
           self.hidden_size = hidden_size
           
           self.query = nn.Linear(hidden_size, hidden_size)
           self.key = nn.Linear(hidden_size, hidden_size)
           self.value = nn.Linear(hidden_size, hidden_size)
           
       def forward(self, inputs):
           Q = self.query(inputs)
           K = self.key(inputs)
           V = self.value(inputs)
           
           scores = torch.matmul(Q, K.transpose(-2, -1))
           attention_weights = torch.softmax(scores, dim=-1)
           
           output = torch.matmul(attention_weights, V)
           
           return output, attention_weights
   ```

2. Encoder:
   
   The encoder consists of a stack of identical layers, each containing a self-attention mechanism and a position-wise feed-forward neural
   network. The self-attention mechanism captures dependencies between words in the input sequence, while the feed-forward network applies non
   linear transformations to each position separately. Here's a Python implementation of the encoder layer:

   ```
   class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_dim, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attention = SelfAttention(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_size)
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, inputs):
        attention_output, _ = self.self_attention(inputs)
        attention_output = self.dropout(attention_output)
        residual_output = self.layer_norm(inputs + attention_output)
        
        feed_forward_output = self.feed_forward(residual_output)
        feed_forward_output = self.dropout(feed_forward_output)
        output = self.layer_norm(residual_output + feed_forward_output)
        
        return output
   ```


### Using already built GPTs
1. Importing Required Libraries:
   
   To use already built GPTs, you'll need to import the necessary libraries. The popular transformers library by Hugging Face provides pre-trained
   GPT models and utilities for text generation.
   ```
   from transformers import GPT2LMHeadModel, GPT2Tokenizer
   ```

3. Loading the Pre-trained Model and Tokenizer:

   GPT models are pre-trained on massive amounts of text data, making them proficient in generating coherent and contextually relevant text. We can
   load a pre-trained GPT model and its associated tokenizer.

   ```
   model_name = "gpt2"  # Specify the model name, e.g., "gpt2", "gpt2-medium", etc.
   model = GPT2LMHeadModel.from_pretrained(model_name)
   tokenizer = GPT2Tokenizer.from_pretrained(model_name)
   ```
4. Generating Text with GPTs:
   
   Using the pre-trained GPT model, you can generate text by providing a prompt or an initial input sequence. The model predicts the next token in
   the sequence based on the provided context.
   ```
   def generate_text(prompt, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text
   ```

   In the generate_text function, we encode the prompt using the tokenizer, generate text with the model's generate method, and decode the generated
   output into readable text.

5. Fine-tuning GPTs:

   GPT models can also be fine-tuned on specific tasks by training them on domain-specific data or by adding task-specific layers on top of the pre
   trained model. Fine-tuning allows the model to adapt to the specific nuances and requirements of the target task.

   Here's an example of fine-tuning GPT for a text classification task using the transformers library:

   ```
   from transformers import GPT2ForSequenceClassification, GPT2Tokenizer, AdamW

   # Load pre-trained GPT model and tokenizer
   model_name = "gpt2"
   model = GPT2ForSequenceClassification.from_pretrained(model_name)
   tokenizer = GPT2Tokenizer.from_pretrained(model_name)
   
   # Prepare training data
   train_dataset = ...
   train_dataloader = ...
   
   # Fine-tuning setup
   optimizer = AdamW(model.parameters(), lr=2e-5)
   epochs = 5
   
   # Fine-tuning loop
   for epoch in range(epochs):
       for batch in train_dataloader:
           inputs = tokenizer(batch['text'], truncation=True, padding=True, return_tensors="pt")
           labels = batch['labels']
   
           outputs = model(**inputs, labels=labels)
           loss = outputs.loss
   
           loss.backward()
           optimizer.step()
           optimizer.zero_grad()
   ```

