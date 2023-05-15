import torch
import torch.nn as nn
from transformers import AutoTokenizer

class TransformerChatbot(nn.Module):
    def __init__(self, model_name_or_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = nn.Transformer(
            d_model=768,
            nhead=12,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=3072,
            dropout=0.1,
        )
        self.fc = nn.Linear(768, self.tokenizer.vocab_size)

    def forward(self, input_ids, decoder_input_ids):
        encoder_outputs = self.model.encoder(input_ids)
        decoder_outputs = self.model.decoder(
            decoder_input_ids, encoder_outputs=encoder_outputs
        )
        output_logits = self.fc(decoder_outputs)
        return output_logits

    def generate_text(self, prompt, max_length=50):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        if input_ids.shape[1] == 0:
            return "Please provide a prompt to continue the conversation."
        decoder_input_ids = input_ids.clone().detach()
        for i in range(max_length):
            output_logits = self.forward(input_ids, decoder_input_ids)
            next_token_logits = output_logits[:, -1, :]
            next_token_id = next_token_logits.argmax(-1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id.unsqueeze(-1)], dim=-1)
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
        generated_text = self.tokenizer.decode(decoder_input_ids.squeeze(), skip_special_tokens=True)
        return generated_text

