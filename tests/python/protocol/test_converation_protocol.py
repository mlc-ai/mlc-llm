import pytest

from mlc_llm.conversation_template import ConvTemplateRegistry
from mlc_llm.protocol.conversation_protocol import Conversation


def get_conv_templates():
    return ["llama-2", "mistral_default", "gorilla", "chatml", "phi-2"]


@pytest.mark.parametrize("conv_template_name", get_conv_templates())
def test_json(conv_template_name):
    template = ConvTemplateRegistry.get_conv_template(conv_template_name)
    j = template.to_json_dict()
    template_parsed = Conversation.from_json_dict(j)
    assert template == template_parsed


if __name__ == "__main__":
    test_json()
