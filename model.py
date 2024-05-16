import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List


class Model:
    def __init__(self, model_name):
        """
        Initialize the Model with a transformer model and tokenizer from Hugging Face.
        Args:
        model_name (str): The name of the model to load (e.g., 'gpt2', 'distilgpt2').
        """
        self.name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
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
            item,
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


#TODO Write tests for the method to compute options probs
# (For single token options, do other way to compute and compare)
