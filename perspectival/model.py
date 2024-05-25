import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
from tqdm import tqdm

from .interfaces import Model
from .dataset import Item


class LazyTransformer(Model):
    """Class to use transformer models while only loading them when needed"""

    def __init__(self, model_name, **model_kwargs):
        self.name = model_name
        self.model_kwargs = model_kwargs

    def compute_option_log_likelihoods(
            self,
            items: List[Item],
            add_whitespace: bool=True,
            **kwargs,
        ) -> List[float]:
        # Set up
        model = AutoModelForCausalLM.from_pretrained(self.name, **self.model_kwargs)
        if self.name.startswith('apple/OpenELM'):
            tokenizer_name = 'NousResearch/Llama-2-7b-hf'
        else:
            tokenizer_name = self.name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()  # Set the model to evaluation mode

        def compute_logits(encoded_texts):
            with torch.no_grad():
                outputs = model(**encoded_texts, **kwargs)
                return outputs.logits

        # Initialize return list
        log_likelihoods = []

        print("Computing option log likelihoods ...")
        for item in tqdm(items):
            input_texts = [item.prompt + (" " if add_whitespace else "") + option for option in item.options]

            # Tokenize the combined texts
            encoded_inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt')

            # Get logits from the model
            logits = compute_logits(encoded_inputs)

            # Ignore the log probs at the beginning
            prompt_length = len(tokenizer(item.prompt, add_special_tokens=False)['input_ids']) #- 1

            # Calculate log probabilities from the logits for each token
            log_probs = F.log_softmax(logits, dim=-1)[:, prompt_length - 1:-1]
            # Need to offset by one since last position contains prediction for current token

            input_ids = encoded_inputs['input_ids'][:, prompt_length:]
            attention_mask = encoded_inputs['attention_mask'][:, prompt_length:]

            # Get log probabilities of actual tokens
            token_log_probs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)

            # Set irrelevant entries at the end (from padding) to zero
            masked_log_probs = token_log_probs * attention_mask

            item_log_likelihoods = torch.sum(masked_log_probs, dim=1).numpy()
            log_likelihoods.append(item_log_likelihoods)
        return log_likelihoods


class Transformer(Model):
    def __init__(self, model_name, **model_kwargs):
        """
        Initialize the Model with a transformer model and tokenizer from Hugging Face.
        Args:
        model_name (str): The name of the model to load (e.g., 'gpt2', 'distilgpt2').
        """
        self.name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if model_name.startswith('apple/OpenELM'):
            tokenizer_name = 'NousResearch/Llama-2-7b-hf'
        else:
            tokenizer_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_texts(self, texts, **kwargs):
        """
        Tokenize a list of texts using the loaded tokenizer.
        Args:
        texts (list of str): The texts to tokenize.
        Returns:
        torch.Tensor: The tokenized texts as a tensor.
        """
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', **kwargs)

    def get_logits(self, encoded_texts, **kwargs):
        """
        Compute the logits for the encoded texts using the loaded model.
        Args:
        encoded_texts (torch.Tensor): The tokenized and encoded texts.
        Returns:
        torch.Tensor: The logits from the model.
        """
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            outputs = self.model(**encoded_texts, **kwargs)
            return outputs.logits

    def compute_option_log_likelihoods(
            self,
            items: List[Item],
            add_whitespace: bool=True,
            **kwargs,
        ) -> List[float]:
        log_likelihoods = []

        print("Computing option log likelihoods ...")
        for item in tqdm(items):
            item_log_likelihoods = self.compute_option_log_likelihoods_single_item(item, **kwargs)
            log_likelihoods.append(item_log_likelihoods)
        return log_likelihoods

    def compute_option_log_likelihoods_single_item(
            self,
            item: Item,
            add_whitespace: bool=True,
        ) -> List[float]:
        input_texts = [item.prompt + (" " if add_whitespace else "") + option for option in item.options]

        # Tokenize the combined texts
        encoded_inputs = self.tokenize_texts(input_texts)

        # Get logits from the model
        logits = self.get_logits(encoded_inputs)

        # Ignore the log probs at the beginning
        prompt_length = len(self.tokenizer(item.prompt, add_special_tokens=False)['input_ids']) #- 1

        # Calculate log probabilities from the logits for each token
        log_probs = F.log_softmax(logits, dim=-1)[:, prompt_length - 1:-1]
        # Need to offset by one since last position contains prediction for current token

        input_ids = encoded_inputs['input_ids'][:, prompt_length:]
        attention_mask = encoded_inputs['attention_mask'][:, prompt_length:]

        # Get log probabilities of actual tokens
        token_log_probs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)

        # Set irrelevant entries at the end (from padding) to zero
        masked_log_probs = token_log_probs * attention_mask

        return torch.sum(masked_log_probs, dim=1).numpy()
