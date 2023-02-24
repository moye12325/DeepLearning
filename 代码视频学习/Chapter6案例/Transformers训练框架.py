from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="path/to/save/folder/",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

train_dataset = dataset["train"]  # doctest: +SKIP
eval_dataset = dataset["eval"]  # doctest: +SKIP

from transformers import DefaultDataCollator

data_collator = DefaultDataCollator()

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)  # doctest: +SKIP

trainer.train()