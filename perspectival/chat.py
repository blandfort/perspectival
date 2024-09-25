from typing import Optional, List

from .model import Transformer
from .dataset import Item


DEFAULT_CHAT_TEMPLATE = """{% for message in messages %}
{% if message['role'] == 'user' %}
{{ '<|user|>\n' + message['content'] + eos_token }}
{% elif message['role'] == 'system' %}
{{ '<|system|>\n' + message['content'] + eos_token }}
{% elif message['role'] == 'assistant' %}
{{ '<|assistant|>\n'  + message['content'] + eos_token }}
{% endif %}
{% if loop.last and add_generation_prompt %}
{{ '<|assistant|>' }}
{% endif %}
{% endfor %}"""


class ChatTransformer(Transformer):
    """Transformer model that is fine-tuned for conversations.

    This class applies chat templates to items, so special tags
    of instruct models are added automatically."""

    def __init__(
        self,
        model_name,
        *,
        lazy_loading: bool = True,
        model_kwargs: Optional[dict] = None,
        tokenizer_kwargs: Optional[dict] = None,
        name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        chat_template: Optional[str] = None
    ):
        self.system_prompt = system_prompt
        self.chat_template = chat_template

        super().__init__(
            model_name=model_name,
            lazy_loading=lazy_loading,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            name=name,
        )

    def _load_model(self):
        model, tokenizer = super()._load_model()

        if self.chat_template is not None:
            # Override chat template if it was given via __init__
            tokenizer.chat_template = self.chat_template

        # Ensure that a chat template is configured
        if tokenizer.chat_template is None:
            print("WARNING: No chat template provided, using default template!")
            tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

        # Permanently set the tokenizer since we need it at various places
        # (NOTE This means lazy loading only works for the main model here!)
        self.tokenizer = tokenizer

        return model, tokenizer

    def make_continuation_prompt(
        self,
        item: Item,
    ) -> str:
        chat = []
        if self.system_prompt is not None:
            chat.append({"role": "system", "content": self.system_prompt})
        chat.append({"role": "user", "content": item.get_continuation_prompt()})

        return self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

    def make_option_prompts(
        self,
        item: Item,
    ) -> List[str]:
        chat = []
        if self.system_prompt is not None:
            chat.append({"role": "system", "content": self.system_prompt})

        # We assume that item.prompt is the user prompt and options
        # describe answers by the assistant
        chat.append({"role": "user", "content": item.prompt})
        statements = []
        for option in item.options:
            assistant_turn = {"role": "assistant", "content": option}
            option_chat = chat + [assistant_turn]
            statements.append(
                self.tokenizer.apply_chat_template(
                    option_chat, tokenize=False, add_generation_prompt=False
                )
            )
        return statements
