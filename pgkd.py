import json
from typing import Any

import datasets
import numpy as np
import torch
import wandb
from datasets import ClassLabel, concatenate_datasets
from pydantic import BaseModel, Field, ValidationError
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from providers import OpenAITeacher
from response_formats import GeneratedSamples

# init wandb
wandb.init(project="pgkd")


def compute_metrics_factory(num_labels: int, label_list=None):
    def _compute(pred):
        preds, labels = pred
        preds = np.argmax(preds, axis=1)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels,
            preds,
            average=None,
            labels=label_list or list(range(num_labels)),
            zero_division=0,
        )
        macro = precision_recall_fscore_support(labels, preds, average="macro")[:3]
        weighted = precision_recall_fscore_support(labels, preds, average="weighted")[
            :3
        ]
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_precision": macro[0],
            "macro_recall": macro[1],
            "macro_f1": macro[2],
            "weighted_precision": weighted[0],
            "weighted_recall": weighted[1],
            "weighted_f1": weighted[2],
        }

    return _compute


class PGKDConfig(BaseModel):
    tokenizer: Any
    model: Any
    num_labels: int
    initial_dataset: Any
    val_dataset: Any
    dataset_class_taxonomy: Any
    device: str = Field(default="cuda" if torch.cuda.is_available() else "cpu")
    num_kd_steps: int = Field(default=10, ge=1)
    patience_limit: int = Field(default=5, ge=1)
    batch_size: int = Field(default=32, ge=1)
    pgkd_number_samples: int = Field(default=16, ge=1)
    number_few_shot_samples: int = Field(default=16, ge=1)
    number_hard_negatives: int = Field(default=16, ge=1)
    num_correct_incorrect_samples: int = Field(default=16, ge=1)
    val_metrics: dict | None = None
    id2label: dict | None = None
    label2id: dict | None = None
    teacher: Any = Field(default_factory=OpenAITeacher)

    class Config:
        arbitrary_types_allowed = True


class PGKD:
    def __init__(self, config: PGKDConfig):
        # init wandb run
        wandb.run.name = "pgkd"

        self.device = config.device
        self.num_labels = config.num_labels
        self.num_kd_steps = config.num_kd_steps
        self.patience_limit = config.patience_limit
        self.batch_size = config.batch_size

        self.model = config.model.to(self.device)
        self.teacher = config.teacher
        self.tokenizer = config.tokenizer
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Prepare datasets
        self.train_dataset = config.initial_dataset
        self.val_dataset = config.val_dataset
        self.dataset_class_taxonomy = config.dataset_class_taxonomy

        # Initialize best model tracking
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.patience_counter = 0

        # initialise training arguments
        self.training_args = TrainingArguments(
            output_dir="pgkd_output",
            learning_rate=2e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=1,
            eval_strategy="no",
            save_strategy="epoch",
            save_total_limit=5,
            load_best_model_at_end=False,
            report_to="wandb",
        )

        self.id2label = config.id2label
        self.label2id = config.label2id
        label_names = [self.id2label[i] for i in range(len(self.id2label))]
        self.label_feature = ClassLabel(names=label_names)

        # initialise pgkd hyperparameters
        self.few_shot_samples = self.get_few_shot_samples(
            config.number_few_shot_samples
        )
        self.number_hard_negatives = config.number_hard_negatives
        self.num_correct_incorrect_samples = config.num_correct_incorrect_samples
        self.pgkd_number_samples = config.pgkd_number_samples
        if config.val_metrics is not None:
            self.val_metrics = config.val_metrics
        else:
            self.val_metrics = None
            self.get_val_metrics()
        self.compute_metrics = compute_metrics_factory(self.num_labels)

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["text"], padding=True, truncation=True, return_tensors="pt"
        )

    def get_val_metrics(self):
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )
        metrics = trainer.evaluate()

        # remove "eval_" prefix from keys, except for loss
        self.val_metrics = {
            k.replace("eval_", ""): v for k, v in metrics.items() if "loss" not in k
        }
        # add loss to val_metrics
        self.val_metrics["eval_loss"] = metrics["eval_loss"]

    def get_few_shot_samples(self, num_samples=16):
        # get random samples from the training dataset
        shuffled_dataset = self.train_dataset.shuffle(seed=42)
        samples = shuffled_dataset[:num_samples]
        # get text, label id and label name triplets
        few_shot_samples = []
        for i in range(num_samples):
            text = self.tokenizer.decode(
                samples["input_ids"][i], skip_special_tokens=True
            )
            label_id = samples["label"][i]
            # convert label id to label name
            label_name = self.id2label[label_id]
            few_shot_samples.append(
                {"text": text, "label_id": label_id, "label_name": label_name}
            )
        return few_shot_samples

    def train_epoch(self):
        """
        Training loop with memory tracking
        """
        import gc

        import psutil

        def log_memory(stage):
            process = psutil.Process()
            memory = process.memory_info().rss / 1024 / 1024
            print(f"Memory at {stage}: {memory:.2f} MB")

        log_memory("start")

        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

        # Convert to iterable dataset to avoid loading everything at once
        from torch.utils.data import IterableDataset

        class IterableWrapper(IterableDataset):
            def __init__(self, dataset):
                self.dataset = dataset

            def __iter__(self):
                for i in range(len(self.dataset)):
                    yield self.dataset[i]

        train_data = IterableWrapper(self.train_dataset)

        log_memory("after dataset creation")

        train_dataloader = DataLoader(
            train_data, batch_size=self.batch_size, collate_fn=self.data_collator
        )

        log_memory("after dataloader creation")

        total_loss = 0
        num_batches = 0
        for batch_idx, batch in enumerate(train_dataloader):
            log_memory(f"start of batch {batch_idx}")

            optimizer.zero_grad(set_to_none=True)

            outputs = self.model(**batch)
            loss = outputs.loss

            log_memory(f"after forward pass batch {batch_idx}")

            loss.backward()

            log_memory(f"after backward pass batch {batch_idx}")

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Aggressive cleanup
            del outputs
            del loss
            del batch
            gc.collect()

            log_memory(f"end of batch {batch_idx}")

            if batch_idx > 0 and batch_idx % 10 == 0:
                print(f"Processed {batch_idx} batches")

        avg_loss = total_loss / num_batches

        log_memory("before eval")

        # Evaluate
        self.model.eval()
        # get val_metrics
        self.get_val_metrics()
        # log val_metrics to wandb
        wandb.log(self.val_metrics)

        return avg_loss, self.val_metrics

    def get_hard_negatives_and_correct_incorrect_samples(self):
        hard_negatives = []
        incorrect_samples = []
        correct_samples = []
        self.model.eval()
        dataloader = DataLoader(
            self.train_dataset, self.batch_size, collate_fn=self.data_collator
        )
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}

                # get predictions and confidence
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                predicted = torch.argmax(probs, dim=1)
                confidence = torch.max(probs, dim=1).values

                # Find misclassified samples with high confidence
                misclassified = predicted != inputs["labels"]
                correctly_classified = predicted == inputs["labels"]

                conf_values = confidence[misclassified].cpu().numpy()

                if len(conf_values) > 0:
                    hard_negatives.extend(
                        [
                            {
                                "text": self.tokenizer.decode(
                                    inputs["input_ids"][i], skip_special_tokens=True
                                ),
                                "true_label": inputs["labels"][i].item(),
                                "predicted": predicted[i].item(),
                                "confidence": confidence[i].item(),
                            }
                            for i in range(len(inputs["labels"]))
                            if misclassified[i]
                        ]
                    )
                if len(misclassified) > 0:
                    incorrect_samples.extend(
                        [
                            {
                                "text": self.tokenizer.decode(
                                    inputs["input_ids"][i], skip_special_tokens=True
                                ),
                                "true_label": inputs["labels"][i].item(),
                                "predicted": predicted[i].item(),
                            }
                            for i in range(len(inputs["labels"]))
                            if misclassified[i]
                        ]
                    )
                if len(correctly_classified) > 0:
                    correct_samples.extend(
                        [
                            {
                                "text": self.tokenizer.decode(
                                    inputs["input_ids"][i], skip_special_tokens=True
                                ),
                                "true_label": inputs["labels"][i].item(),
                                "predicted": predicted[i].item(),
                            }
                            for i in range(len(inputs["labels"]))
                            if correctly_classified[i]
                        ]
                    )

        # Sort by confidence and get top_k
        hard_negatives.sort(key=lambda x: x["confidence"], reverse=True)
        return (
            hard_negatives[: self.number_hard_negatives],
            correct_samples[: self.num_correct_incorrect_samples],
            incorrect_samples[: self.num_correct_incorrect_samples],
        )

    def _generate_new_samples(self, hard_negatives, correct_samples, incorrect_samples):
        # Construct prompt for OpenAI
        prompt = (
            f"You are a Teacher model for a Student LM "
            f"to perform topic detection on the following "
            f"taxonomy:\n {self.dataset_class_taxonomy}\n"
            f"Here are a few labeled examples that show the "
            f"correct label for this task:\n"
            f"{json.dumps(self.few_shot_samples, indent=2)}\n"
            f"Given the current model performance, please "
            f"generate {self.pgkd_number_samples} training samples "
            "for the model to improve its performance. "
            f"The response should be a list of dictionaries in "
            "JSON format, the response needs to be "
            f"parsable so do not output anything else rather "
            f"than the response itself. The objective is to "
            f"maximize the model accuracy, generate new "
            f"samples knowing that the classification report "
            f"over validation set is:\n"
            f"{json.dumps(self.val_metrics, indent=2)}\n"
            f"Please consider a few samples that the model "
            f"was able to classify correctly:\n"
            f"{json.dumps(correct_samples, indent=2)}\n"
            f"And samples the model was not able to classify correctly:\n"
            f"{json.dumps(incorrect_samples, indent=2)}\n"
            f"The model has a high confidence in classifying "
            f"the following misclassified examples:\n"
            f"{json.dumps(hard_negatives, indent=2)}"
        )

        response = self.teacher.generate(prompt)
        try:
            samples = GeneratedSamples.model_validate_json(response)
        except ValidationError as e:
            # 💡 automatic repair loop
            repair_prompt = (
                "The previous response was not valid JSON "
                "according to this error:\n"
                f"{e}.\nPlease output ONLY a JSON object "
                "following the schema again."
            )
            response = self.teacher.generate(repair_prompt)
            samples = GeneratedSamples.model_validate_json(response)

        return samples

    def _add_new_samples_to_dataset(self, new_samples):
        if not new_samples.samples:
            return

        # Create new dataset
        dataset_dict = {
            "text": [sample.text for sample in new_samples.samples],
            "label": [sample.label for sample in new_samples.samples],
        }
        new_dataset = datasets.Dataset.from_dict(dataset_dict)

        # Tokenize without caching
        new_dataset = new_dataset.map(
            lambda x: self.tokenize_function(x, self.tokenizer),
            batched=True,
            remove_columns=["text"],
            load_from_cache_file=False,
        )

        # Cast features
        new_dataset = new_dataset.cast(self.train_dataset.features)

        # Concatenate
        self.train_dataset = concatenate_datasets([self.train_dataset, new_dataset])

    def train(self):
        datasets.disable_caching()
        for step in range(self.num_kd_steps):
            self.get_val_metrics()
            # Get hard negatives, correctly and incorrectly classififed samples
            hard_negatives, correct_samples, incorrect_samples = (
                self.get_hard_negatives_and_correct_incorrect_samples()
            )

            # Generate new samples using llm
            new_samples = self._generate_new_samples(
                hard_negatives, correct_samples, incorrect_samples
            )

            self._add_new_samples_to_dataset(new_samples)
            # Train for one epoch
            train_loss, val_metrics = self.train_epoch()
            # update val metrics
            self.val_metrics = val_metrics
            # Check for early stopping
            if val_metrics["eval_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["eval_loss"]
                self.best_model_state = self.model.state_dict().copy()
                # Not clear if this is actually reset in the paper
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience_limit:
                    print(f"Early stopping at step {step}")
                    # reinstate and return the best model
                    if self.best_model_state is not None:
                        self.model.load_state_dict(self.best_model_state)
                    return self.model

            print(
                f"Step {step}: Train Loss = {train_loss:.4f}, "
                f"Val Loss = {val_metrics['eval_loss']:.4f}"
            )

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return self.model
