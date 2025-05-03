import torch
from torch import nn
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import concatenate_datasets, ClassLabel

import datasets 
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import json

import wandb
from response_formats import GeneratedSamples

# init wandb
wandb.init(project="pgkd")

def compute_metrics(pred):
    """Compute per class precision, f1, recall and support.
    as well as accuracy, macro and weighted averages. 

    Args:
        pred (dict): model predictions

    Returns:
        dict: metrics
    """
    predictions, labels = pred
    # convert logits to predicted labels
    predictions = np.argmax(predictions, axis=1)

    # caculate precision, recall, f1 and support per class
    metrics = precision_recall_fscore_support(labels, predictions, average=None, zero_division=0, labels=list(range(41)))
    # caculate accuracy
    accuracy = accuracy_score(labels, predictions)
    # get macro and weighted averages of precision, recall, f1 and support
    macro = precision_recall_fscore_support(labels, predictions, average='macro')
    weighted = precision_recall_fscore_support(labels, predictions, average='weighted')

    # convert into a single dictionary
    metrics_output = {
        'accuracy': accuracy,
        'macro': {
            'precision': macro[0],
            'recall': macro[1],
            'f1': macro[2],
            'support': macro[3]
        },
        'weighted': {
            'precision': weighted[0],
            'recall': weighted[1],
            'f1': weighted[2],
            'support': weighted[3]
        }
    }
    # add per class metrics
    for i in list(range(41)):
        id = str(i)
        metrics_output[f"eval_{id}"] = {
            'precision': metrics[0][i],
            'recall': metrics[1][i],
            'f1': metrics[2][i],
            'support': metrics[3][i]
        }
    return metrics_output

    
    
class PGKD:
    def __init__(
        self,
        tokenizer,
        model, 
        num_labels,
        initial_dataset,
        val_dataset,
        dataset_class_taxonomy,
        openai_client,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_kd_steps=10,
        patience_limit=5,
        batch_size=32,
        pgkd_number_samples=16,
        number_few_shot_samples=16,
        number_hard_negatives=16,
        num_correct_incorrect_samples=16,
        val_metrics=None,
        id2label=None,
        label2id=None,
    ):
        # init wandb run 
        wandb.run.name = "pgkd"

        self.device = device
        self.num_labels = num_labels
        self.num_kd_steps = num_kd_steps
        self.patience_limit = patience_limit
        self.batch_size = batch_size
        
        self.client = openai_client
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Prepare datasets
        self.train_dataset = initial_dataset
        self.val_dataset = val_dataset
        self.dataset_class_taxonomy = dataset_class_taxonomy
        
        # Initialize best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0

        # initialise training arguments
        self.training_args = TrainingArguments(
                output_dir=f"pgkd_output",
                learning_rate=2e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=1,
                eval_strategy="no",
                save_strategy="epoch",
                save_total_limit = 5,
                load_best_model_at_end=False,
                report_to="wandb",
            )
        
        self.id2label = id2label
        self.label2id = label2id
        label_names = [id2label[i] for i in range(len(self.id2label))]
        self.label_feature = ClassLabel(names=label_names)

        # initialise pgkd hyperparameters
        self.few_shot_samples = self.get_few_shot_samples(number_few_shot_samples)
        self.number_hard_negatives = number_hard_negatives
        self.num_correct_incorrect_samples = num_correct_incorrect_samples
        self.pgkd_number_samples = pgkd_number_samples
        if val_metrics is not None:
            self.val_metrics = val_metrics
        else:
            self.val_metrics = None
            self.get_val_metrics()


    def tokenize_function(self, examples):
        return self.tokenizer(examples['text'], padding=True, trunctation=True, return_tensors='pt')
    
    def get_val_metrics(self):
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics)
        metrics = trainer.evaluate()

        # remove "eval_" prefix from keys, except for loss 
        self.val_metrics =  {k.replace("eval_", ""): v for k, v in metrics.items() if not 'loss' in k}
        # add loss to val_metrics
        self.val_metrics['eval_loss'] = metrics['eval_loss']


    def get_few_shot_samples(self, num_samples=16):
        # get random samples from the training dataset
        shuffled_dataset = self.train_dataset.shuffle(seed=42)
        samples = shuffled_dataset[:num_samples] 
        # get text, label id and label name triplets
        few_shot_samples = []
        for i in range(num_samples):
            text = self.tokenizer.decode(samples['input_ids'][i], skip_special_tokens=True)
            label_id = samples['label'][i]
            # convert label id to label name
            label_name = self.id2label[label_id]
            few_shot_samples.append({'text': text, 'label_id': label_id, 'label_name': label_name})
        return few_shot_samples

    @staticmethod
    def tokenize_function(examples, tokenizer):
        return tokenizer(examples['text'], padding='max_length')
    
    # def train_epoch(self):
    #     """
    #     Train the model for one epoch, returning the loss
    #     and the last log history
    #     """
    #     trainer = Trainer(
    #         model=self.model,
    #         args=self.training_args,
    #         train_dataset=self.train_dataset,
    #         eval_dataset=self.val_dataset,
    #         processing_class=self.tokenizer,
    #         data_collator=self.data_collator,
    #         # TODO TEMP
    #         compute_metrics=None)
    #     results = trainer.train()
    #     self.model = trainer.model
    #     return results.loss, trainer.state.log_history[-1]

    # def train_epoch(self):
    #     """
    #     Simplified training loop to debug memory issues
    #     """
    #     self.model.train()
    #     optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

    #     class SimpleDataset(Dataset):
    #         def __init__(self, data):
    #             self.data = data
    #         def __len__(self):
    #             return len(self.data)
    #         def __getitem__(self, idx):
    #             return {k: v[idx] for k, v in self.data.items()}
    
    #     train_data = SimpleDataset(self.train_dataset)
    #     val_data = SimpleDataset(self.val_dataset)
    #     train_dataloader = DataLoader(
    #         self.train_dataset, 
    #         batch_size=self.batch_size, 
    #         shuffle=True,
    #         collate_fn=self.data_collator
    #     )
        
    #     total_loss = 0
    #     for batch in train_dataloader:
    #         optimizer.zero_grad()
    #         outputs = self.model(**batch)
    #         loss = outputs.loss
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()
    #         del loss
    #         del outputs
            
    #     avg_loss = total_loss / len(train_dataloader)
        
    #     # Evaluate
    #     self.model.eval()
    #     eval_dataloader = DataLoader(
    #         self.val_dataset,
    #         batch_size=2,
    #         collate_fn=self.data_collator
    #     )
        
    #     val_loss = 0
    #     with torch.no_grad():
    #         for batch in eval_dataloader:
    #             outputs = self.model(**batch)
    #             val_loss += outputs.loss.item()
        
    #     avg_val_loss = val_loss / len(eval_dataloader)
    #     # Clear memory
    #     del train_dataloader
    #     del eval_dataloader
    #     del train_data
    #     del val_data
    #     metrics = {'eval_loss': avg_val_loss}
        
    #     return avg_loss, metrics

    def train_epoch(self):
        """
        Training loop with memory tracking
        """
        import psutil
        import gc
        
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
        val_data = IterableWrapper(self.val_dataset)
        
        log_memory("after dataset creation")
        
        train_dataloader = DataLoader(
            train_data, 
            batch_size=self.batch_size,
            collate_fn=self.data_collator
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
        dataloader = DataLoader(self.train_dataset,self.batch_size, collate_fn=self.data_collator)
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}

                # get predictions and confidence
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                predicted = torch.argmax(probs, dim=1)
                confidence = torch.max(probs, dim=1).values
                
                # Find misclassified samples with high confidence
                misclassified = (predicted != inputs['labels'])
                correctly_classified = (predicted == inputs['labels'])
            
                conf_values = confidence[misclassified].cpu().numpy()
                
                if len(conf_values) > 0:
                    hard_negatives.extend([
                        {   # store text with special tokens removed
                            'text': self.tokenizer.decode(inputs['input_ids'][i], skip_special_tokens=True),
                            'true_label': inputs['labels'][i].item(),
                            'predicted': predicted[i].item(),
                            'confidence': confidence[i].item()
                        }
                        for i in range(len(inputs['labels']))
                        if misclassified[i]
                    ])
                if len(misclassified) > 0:
                    incorrect_samples.extend([
                        {
                            'text': self.tokenizer.decode(inputs['input_ids'][i], skip_special_tokens=True),
                            'true_label': inputs['labels'][i].item(),
                            'predicted': predicted[i].item(),
                        }
                        for i in range(len(inputs['labels']))
                        if misclassified[i]
                    ])
                if len(correctly_classified) > 0:
                    correct_samples.extend([
                        {
                            'text': self.tokenizer.decode(inputs['input_ids'][i], skip_special_tokens=True),
                            'true_label': inputs['labels'][i].item(),
                            'predicted': predicted[i].item(),
                        }
                        for i in range(len(inputs['labels']))
                        if correctly_classified[i]
                    ])
                
        
        # Sort by confidence and get top_k
        hard_negatives.sort(key=lambda x: x['confidence'], reverse=True)
        return hard_negatives[:self.number_hard_negatives], correct_samples[:self.num_correct_incorrect_samples], incorrect_samples[:self.num_correct_incorrect_samples]

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
        f"The model has a high confidence in classifying the following misclassified examples:\n"
        f"{json.dumps(hard_negatives, indent=2)}"
        )

        print(prompt)
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format=GeneratedSamples
        )
        
        try:
            new_samples = completion.choices[0].message.parsed
            return new_samples
        except:
            logging.error("Failed to parse OpenAI response")
            return []

    def _add_new_samples_to_dataset(self, new_samples):
        if not new_samples.samples:
            return
        
        # Create new dataset
        dataset_dict = {
            'text': [sample.text for sample in new_samples.samples],
            'label': [sample.label for sample in new_samples.samples]
        }
        new_dataset = datasets.Dataset.from_dict(dataset_dict)
        
        # Tokenize without caching
        new_dataset = new_dataset.map(
            lambda x: self.tokenize_function(x, self.tokenizer),
            batched=True,
            remove_columns=['text'],
            load_from_cache_file=False
        )
        
        # Cast features
        new_dataset = new_dataset.cast(self.train_dataset.features)
        
        # Concatenate
        self.train_dataset = concatenate_datasets([
            self.train_dataset,
            new_dataset
        ])

    def train(self):
        datasets.disable_caching()
        for step in range(self.num_kd_steps):
            self.get_val_metrics()
            # Get hard negatives, correctly and incorrectly classififed samples
            hard_negatives, correct_samples, incorrect_samples = self.get_hard_negatives_and_correct_incorrect_samples()
            
            # Generate new samples using llm
            new_samples = self._generate_new_samples(hard_negatives, correct_samples, incorrect_samples)
            
            self._add_new_samples_to_dataset(new_samples)
            # Train for one epoch
            train_loss, val_metrics = self.train_epoch()
            # update val metrics
            self.val_metrics = val_metrics
            # Check for early stopping
            if val_metrics['eval_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['eval_loss']
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

            print(f"Step {step}: Train Loss = {train_loss:.4f}, Val Loss = {val_metrics['eval_loss']:.4f}")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self.model