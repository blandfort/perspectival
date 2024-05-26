import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
from tqdm import tqdm

from .interfaces import Model
from .dataset import Item


class Transformer(Model):
    """Class to use HuggingFace transformer models"""

    def __init__(self, model_name, lazy_loading: bool=True, **model_kwargs):
        self.name = model_name
        self.model_kwargs = model_kwargs
        self.lazy_loading = lazy_loading

        if not self.lazy_loading:
            self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.name, **self.model_kwargs)
        if self.name.startswith('apple/OpenELM'):
            tokenizer_name = 'NousResearch/Llama-2-7b-hf'
        else:
            tokenizer_name = self.name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()  # Set the model to evaluation mode

        return model, tokenizer

    def compute_option_log_likelihoods(
            self,
            items: List[Item],
            add_whitespace: bool=True,
            token_level: bool=False,
            **kwargs,
        ) -> List[float]:
        if self.lazy_loading:
            model, tokenizer = self._load_model()
        else:
            model, tokenizer = self.model, self.tokenizer

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
            log_probs = F.log_softmax(logits, dim=-1)

            if tokenizer.padding_side=='right':
                # Need to offset by one since last position contains prediction for current token
                log_probs = log_probs[:, prompt_length - 1:-1]

                input_ids = encoded_inputs['input_ids'][:, prompt_length:]
                attention_mask = encoded_inputs['attention_mask'][:, prompt_length:]

                # Get log probabilities of actual tokens
                token_log_probs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)

                # Set irrelevant entries at the end (from padding) to zero
                masked_log_probs = (token_log_probs * attention_mask).detach().numpy()

            else:
                # Offset by one since likelihoods refer to next position
                log_probs = log_probs[:, :-1]
                input_ids = encoded_inputs['input_ids'][:, 1:]
                attention_mask = encoded_inputs['attention_mask']
                offsets = np.argmax(attention_mask, axis=1) + prompt_length
                attention_mask = attention_mask[:, 1:]

                token_log_probs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)
                masked_log_probs = (token_log_probs * attention_mask).detach().numpy()

                input_ids = [ids[offset:] for ids, offset in zip(input_ids, offsets)]
                masked_log_probs = [lls[offset:] for lls, offset in zip(masked_log_probs, offsets)]

            if token_level:
                log_likelihoods.append({'tokens': input_ids, 'log_likelihoods': masked_log_probs})
            else:
                #item_log_likelihoods = torch.sum(masked_log_probs, dim=1).numpy()
                item_log_likelihoods = [np.sum(lls) for lls in masked_log_probs]
                log_likelihoods.append(item_log_likelihoods)
        return log_likelihoods
