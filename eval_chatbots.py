import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from transformers import DistilBertModel, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments

# Knowledge Distillation: an approach for improving sentence encoder 

# Define the paths for teacher and student models
teacher_model_name = 'bert-base-uncased'
student_model_name = 'prajjwal1/bert-tiny'

# Load teacher and student models and tokenizers
tokenizer_teacher = BertTokenizer.from_pretrained(teacher_model_name)
model_teacher = BertModel.from_pretrained(teacher_model_name)

tokenizer_student = BertTokenizer.from_pretrained(student_model_name)
model_student = BertModel.from_pretrained(student_model_name)

# Define a sample sentence
sentence = "This is a sample sentence."

# Encoding input for both teacher and student
input_ids_teacher = tokenizer_teacher(sentence, return_tensors="pt")["input_ids"]
input_ids_student = tokenizer_student(sentence, return_tensors="pt")["input_ids"]

# Generate teacher's logits
with torch.no_grad():
    outputs_teacher = model_teacher(**input_ids_teacher)
    teacher_logits = outputs_teacher.last_hidden_state.mean(dim=1)

# Train student with knowledge distillation
training_args = TrainingArguments(
    output_dir='./distill_model',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir='./logs',
    logging_steps=100,
    do_train=True
)

# Define a custom distillation loss function
def distillation_loss(y_true, y_pred, teacher_scores, temperature=1):
    loss_fct = torch.nn.KLDivLoss(reduction='batchmean')
    return loss_fct(torch.nn.functional.log_softmax(y_pred / temperature, dim=-1),
                    torch.nn.functional.softmax(teacher_scores / temperature, dim=-1))

class CustomDistillationTrainer(Trainer):
    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        teacher_logits = inputs.pop("teacher_logits")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = distillation_loss(logits, teacher_logits, teacher_logits)
        return loss

# Train student model with distillation
training_args.distillation = True
training_args.do_train = True
training_args.per_device_train_batch_size = 8

trainer = CustomDistillationTrainer(
    model=model_student,
    args=training_args,
    data_collator=None,
    train_dataset=None,  # Add your training dataset here
)

# Train the model
trainer.train()

# Save the distilled model
model_student.save_pretrained('./distilled_model')
