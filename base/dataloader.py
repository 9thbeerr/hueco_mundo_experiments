import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class HFDataset(Dataset):
    def __init__(self, dataset_name, split, tokenizer, max_length=512):

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = load_dataset(dataset_name, split=split)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get a single example
        example = self.dataset[idx]
        text = example['text']
        
        # Tokenize the text
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Remove the batch dimension added by the tokenizer
        return {
            'input_ids': encodings['input_ids'][0],
            'attention_mask': encodings['attention_mask'][0],
        }

def create_dataloaders(dataset_name, tokenizer_name="gpt2", batch_size=16, max_length=512, num_workers=4):
    """
    Create train, validation, and test dataloaders for any Hugging Face dataset.
    
    Args:
        dataset_name (str): Name of the Hugging Face dataset
        tokenizer_name (str): Name of the tokenizer to use
        batch_size (int): Batch size for training
        max_length (int): Maximum sequence length
        num_workers (int): Number of workers for data loading
    
    Returns:
        dict: Dictionary containing train, validation, and test dataloaders
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Define standard splits
    splits = ['train', 'validation']
    
    dataloaders = {}
    
    # Create datasets and dataloaders for each split
    for split in splits:
        dataset = HFDataset(
            dataset_name, 
            split, 
            tokenizer, 
            max_length=max_length
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),  # Only shuffle training data
            num_workers=num_workers,
            pin_memory=True
        )
        
        dataloaders[split] = dataloader
    
    return dataloaders

# Example usage
if __name__ == "__main__":
    # Create dataloaders for TinyStories
    dataloaders = create_dataloaders(
        dataset_name="roneneldan/TinyStories",
        batch_size=32,
        max_length=256,
        num_workers=4
    )
    
    # Check sizes
    for split_name, dataloader in dataloaders.items():
        print(f"{split_name} dataloader has {len(dataloader)} batches")

    # Get a batch from the training dataloader
    # batch = next(iter(dataloaders['train']))
    # print(f"Batch shape: {batch['input_ids'].shape}")