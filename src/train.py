import os
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset, DatasetDict, concatenate_datasets
import numpy as np
import evaluate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the Weights & Biases API key from environment variables
api_key = os.environ.get("WANDB_API_KEY")

testing = os.environ.get("TESTING", "false").lower()  # Check if testing mode is enabled
run_name = os.environ.get("WANDB_RUN_NAME")  # Get the run name from environment variables
dataset_name = os.environ.get("DATASET_NAME")  # Get the dataset name from environment variables

# Raise an error if the API key is not found
if api_key is None:
    raise EnvironmentError("WANDB_API_KEY not found in environment variables.")

# Log in to Weights & Biases using the API key
wandb.login(key=api_key)

# Initialize a new W&B run for experiment tracking
run = wandb.init(project="bert-tiny-sentiment", name=run_name, job_type="train")

# Load the IMDB sentiment dataset (train and test splits)
dataset = load_dataset(dataset_name)
dataset = DatasetDict({
    "train": dataset["train"],
    "test": dataset["test"]
})

# Print the number of samples in the new train and test sets
print(f"Dataset size: {len(dataset['train'])} training samples, {len(dataset['test'])} test samples")

# for testing purposes, we can limit the training & test datasets
if testing == "true":
    dataset["train"] = dataset["train"].select(range(10))
    dataset["test"] = dataset["test"].select(range(1))
    print(f"FOR TESTING: Reduced dataset size: {len(dataset['train'])} training samples, {len(dataset['test'])} test samples")

# Load the BERT-tiny tokenizer
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

# Tokenization function to preprocess the text data
def tokenize_fn(example):
    # Tokenize the text with truncation and padding to max_length=256
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

# Apply the tokenization function to the dataset in batches
encoded = dataset.map(tokenize_fn, batched=True)

# Set the format of the dataset to PyTorch tensors for model training
encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Load the BERT-tiny model for sequence classification (binary classification)
model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)

# Load evaluation metrics: accuracy and F1 score
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

# Define a function to compute metrics during evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    f1_score = f1.compute(predictions=predictions, references=labels, average="weighted")
    return {
        "accuracy": acc["accuracy"],
        "f1": f1_score["f1"]
    }

# Set up training arguments for the Trainer
training_args = TrainingArguments(
    warmup_ratio=0.1,                        # Fraction of steps for learning rate warmup
    lr_scheduler_type="cosine",               # Use cosine learning rate scheduler
    learning_rate=2e-5,                       # Initial learning rate
    max_grad_norm=1.0,                        # Gradient clipping
    save_safetensors=True,                    # Save model in safetensors format
    output_dir="./bert-tiny-sentiment",       # Directory to save model checkpoints
    per_device_train_batch_size=8,            # Batch size for training
    per_device_eval_batch_size=8,             # Batch size for evaluation
    num_train_epochs=2,                       # Number of training epochs
    eval_strategy="steps",                    # Evaluate every N steps
    save_strategy="steps",                    # Save checkpoint every N steps
    save_steps=1000,                          # Save checkpoint every 1000 steps
    eval_steps=1000,                          # Evaluate every 1000 steps
    logging_dir="./logs",                     # Directory for logs
    logging_steps=50,                         # Log every 50 steps
    load_best_model_at_end=True,              # Load the best model at the end of training
    metric_for_best_model="f1",               # Use F1 score to select the best model
    greater_is_better=True,                   # Higher F1 is better
    report_to="wandb",                        # Report metrics to W&B
    run_name="bert-tiny-imdb-run",            # Name of the W&B run
    save_total_limit=2,                       # Keep only the 2 most recent checkpoints
)

# Initialize the Trainer with model, data, metrics, and callbacks
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["test"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Stop early if no improvement for 2 evals
)

# Start training the model
trainer.train()

# Set human-readable labels for the model config (for inference)
model.config.id2label = {0: "Negative", 1: "Positive"}
model.config.label2id = {"Negative": 0, "Positive": 1}

# Save the trained model and tokenizer to disk
model.save_pretrained("bert-tiny-imdb")
tokenizer.save_pretrained("bert-tiny-imdb")

# Log the trained model as a W&B artifact
model_artifact = wandb.Artifact(name="bert-tiny-sentiment-model", type="model")
model_artifact.add_dir("bert-tiny-imdb")  # Add the saved model directory to the artifact
run.log_artifact(model_artifact)          # Log the model artifact to W&B

# Save the train and test splits as CSV files for reproducibility
dataset["train"].to_csv("train_split.csv")
dataset["test"].to_csv("test_split.csv")

# Log the dataset splits as a W&B artifact
data_artifact = wandb.Artifact(name="imdb-75-25-split", type="dataset")
data_artifact.add_file("train_split.csv")  # Add train split CSV
data_artifact.add_file("test_split.csv")   # Add test split CSV
run.log_artifact(data_artifact)            # Log the dataset artifact to W&B

# Link the model artifact to a W&B registry path for versioning and sharing
run.link_artifact(
    artifact=model_artifact,
    target_path="wandb-registry-model/bert-tiny-sentiment-model"
)

# Finish the W&B run
run.finish()


