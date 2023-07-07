
# GPT（生成预训练变压器）概述

GPT（Generative Pre-trained Transformer）是一种语言模型，利用Transformer架构生成类似人类的文本。 它在大量文本数据的语料库上进行训练，可以针对特定的自然语言处理任务进行微调。

论文"Attention Is All You Need"中提出的Transformer架构通过引入自我关注机制彻底改变了自然语言处理任务。 它已被广泛应用于各种序列到序列的任务，包括机器翻译和语言生成。

### 建造政府资讯科技总监办公室

让我们使用一些Python代码片段来探索Transformer架构的关键组件:

1. 自我关注机制:

自我关注机制允许模型在生成输出时权衡输入序列中不同单词的重要性。 它
计算每个单词和序列中所有其他单词之间的注意力得分，捕获关系和依赖关系。 这是一条蟒蛇
实施自我关注:

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


2. 编码器:

编码器由一叠相同的层组成，每层都包含一个自我关注机制和一个位置前馈神经
网络。 自关注机制捕获输入序列中单词之间的依赖关系，而前馈网络应用非
分别对每个位置进行线性变换。 下面是编码器层的Python实现:

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

### 使用已构建的GPTs
1. 导入所需库:

要使用已经构建的GPTs，您需要导入必要的库。 流行的变形金刚图书馆拥抱脸提供预先训练
用于文本生成的GPT模型和实用程序。

 ```
 from transformers import GPT2LMHeadModel, GPT2Tokenizer
 ```


2. 加载预训练模型和标记器:

GPT模型在大量文本数据上进行预训练，使其能够熟练地生成连贯且与上下文相关的文本。 我们可以
加载预训练的GPT模型及其关联的标记器。

```
 model_name = "gpt2"  # Specify the model name, e.g., "gpt2", "gpt2-medium", etc.
 model = GPT2LMHeadModel.from_pretrained(model_name)
 tokenizer = GPT2Tokenizer.from_pretrained(model_name)
 ```

3. 使用GPTs生成文本:

使用预训练的GPT模型，您可以通过提供提示或初始输入序列来生成文本。 该模型预测下一个令牌
基于所提供的上下文的序列。

```
 def generate_text(prompt, max_length=100):
  inputs = tokenizer.encode(prompt, return_tensors="pt")
  outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)

  generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
  return generated_text
 ```

在generate_text函数中，我们使用标记器对提示进行编码，使用模型的generate方法生成文本，并对生成的
输出成可读文本。

4. 微调GPTs:

GPT模型还可以通过在特定于领域的数据上训练它们或在pre上添加特定于任务的层来对特定任务进行微调
训练好的模型。 微调使模型能够适应目标任务的特定细微差别和要求。

下面是使用transformers库对文本分类任务进行GPT微调的示例:

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
