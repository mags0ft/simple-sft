"""
Prompts and generation functions used to interact with the (remote) LLM.
"""

from random import choice

from llm_interface import process_many_out_of_order
from config_reader import config

SYSTEM_PROMPT_GENERATION_PROMPT = """
You're an expert at giving LLMs instructions via a system prompt. Come up \
with a creative, generic prompt for an AI chatbot. Prompt theme: %s. Format \
the prompt as a %s.
""".strip()

SYSTEM_PROMPT_GENERATION_ADDITION = """
Add one or several special requirements of differing complexity to your \
prompt which the assistant has to follow, like output formatting, speaking in \
a specific language/tone etc. Make them very specific.
""".strip()


FOLLOWUP_QUESTION_GENERATION_PROMPT = """
You are a user who is interacting with an AI chatbot. You are currently \
talking in the context of the topic "%s". This is your chat history so far:

```
%s
```

Generate a reasonable follow-up request/message from the viewpoint of the user.

Only answer with this message, nothing else.
""".strip()
