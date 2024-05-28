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

    def __init__(self,
            model_name,
            lazy_loading: bool=True,
            model_kwargs: dict={},
            tokenizer_kwargs: dict={},
        ):
        self.name = model_name
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs
        self.lazy_loading = lazy_loading

        if not self.lazy_loading:
            self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.name, **self.model_kwargs)
        if self.name.startswith('apple/OpenELM'):
            tokenizer_name = 'NousResearch/Llama-2-7b-hf'
            #tokenizer_name = 'meta-llama/Llama-2-7b-hf'
        else:
            tokenizer_name = self.name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **self.tokenizer_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()  # Set the model to evaluation mode

        return model, tokenizer

    def compute_option_log_likelihoods(
            self,
            items: List[Item],
            #add_whitespace: bool=True,
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
            #input_texts = [item.prompt + (" " if add_whitespace else "") + option for option in item.options]
            input_texts = [item.prompt + (" " if (not item.prompt.endswith(' ') and not option.endswith(' ')) else "") + option for option in item.options]

            # Tokenize the combined texts
            encoded_inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True)
            #encoded_inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt')

            # Get logits from the model
            logits = compute_logits(encoded_inputs)

            # Ignore the log probs at the beginning
            prompt_length = len(tokenizer(item.prompt, add_special_tokens=False)['input_ids'])

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

                if token_level:
                    # Convert to token texts
                    input_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]

            else:
                prompt_length +=1  # BOS token

                # Offset by one since likelihoods refer to next position
                log_probs = log_probs[:, :-1]
                input_ids = encoded_inputs['input_ids'][:, 1:]
                attention_mask = encoded_inputs['attention_mask']

                # Note: We don't offset the attention mask to exclude
                # likelihoods of BOS tokens for padded sequences
                # (which can be quite low)
                #attention_mask = attention_mask[:, 1:]
                attention_mask = attention_mask[:, :-1]

                token_log_probs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)
                masked_log_probs = (token_log_probs * attention_mask).detach().numpy()

                # Note: Setting offsets to 0 leads to computing log likelihoods
                # over the whole sequence including the prompt (useful for debugging)
                offsets = np.argmax(attention_mask, axis=1) + prompt_length
                input_tokens = [tokenizer.convert_ids_to_tokens(ids[offset:]) for ids, offset in zip(input_ids, offsets)]
                masked_log_probs = [lls[offset:] for lls, offset in zip(masked_log_probs, offsets)]

                #TODO Apply padding to have same format as for right padding?
                # (Then the conversion to tokens can be done below)

            if token_level:
                log_likelihoods.append({'tokens': input_tokens, 'log_likelihoods': masked_log_probs})
            else:
                #item_log_likelihoods = torch.sum(masked_log_probs, dim=1).numpy()
                item_log_likelihoods = [np.sum(lls) for lls in masked_log_probs]
                log_likelihoods.append(item_log_likelihoods)
        return log_likelihoods


class SimpleTransformer(Transformer):
    """Class to use HuggingFace transformer models with a simple-to-verify
    inference method

    Note that this class is maximized for understandability but isn't efficient!"""

    def compute_option_log_likelihoods(
            self,
            items: List[Item],
            add_whitespace='conditional',
            #token_level: bool=False,
            **kwargs,
        ) -> List[float]:
        if self.lazy_loading:
            model, tokenizer = self._load_model()
        else:
            model, tokenizer = self.model, self.tokenizer

        print(f"Whitespace setting: {add_whitespace}")

        log_likelihoods = []

        print("Computing option log likelihoods ...")
        for item in tqdm(items):
            if add_whitespace=='conditional':
                input_texts = [item.prompt + (" " if (not item.prompt.endswith(' ') and not option.endswith(' ')) else "") + option for option in item.options]
            elif add_whitespace:
                input_texts = [item.prompt + " " + option for option in item.options]
            else:
                input_texts = [item.prompt + option for option in item.options]

            prompt_log_likelihood = np.sum(self.token_log_likelihood(transformer=model, tokenizer=tokenizer, text=item.prompt, add_special_tokens=True))

            option_log_likelihoods = []
            for text in input_texts:
                # First compute the log likelihood of the whole sequence
                log_likelihood = np.sum(self.token_log_likelihood(transformer=model, tokenizer=tokenizer, text=text, add_special_tokens=True))

                # Now remove the part from the prompt
                option_log_likelihood = log_likelihood - prompt_log_likelihood

                option_log_likelihoods.append(option_log_likelihood)
            log_likelihoods.append(option_log_likelihoods)

        return log_likelihoods

    @staticmethod
    def token_log_likelihood(transformer, tokenizer, text: str, add_special_tokens: bool=True) -> List[float]:
        """Simple but inefficient method to calculate the log likelihood of a given text."""
        # Tokenize the text
        encoded_text = tokenizer([text], padding=False, truncation=False, return_tensors='pt', add_special_tokens=add_special_tokens)

        token_indices = encoded_text.input_ids[0][:]

        # Verify that the tokenization did what we expected
        # (This part should probably be done nby tokenizer.batch_decode)
        #reconstructed_text = ''.join(tokenizer.convert_ids_to_tokens(token_indices)).replace('Ġ', ' ').replace('Ċ', '\n').replace('_', ' ')
        #assert reconstructed_text==text.replace('_', ' '), f"These should be the same but aren't:\n{text}\n{reconstructed_text}"

        # Get logits from the model at every position before EOS
        with torch.no_grad():
            logits = transformer(**encoded_text).logits[0]
        # Calculate log likelihoods from the logits
        log_ls = F.log_softmax(logits, dim=1)

        # Get log likelihoods of actual tokens
        token_log_ls = []
        for token_pos in range(1, len(token_indices)):
            # Note: To calculate the likelihood of the given sequence,
            # at a given position we need to check which likelihood where predicted in the last step
            token_log_ls.append(log_ls[token_pos-1, token_indices[token_pos]].item())

        return token_log_ls
