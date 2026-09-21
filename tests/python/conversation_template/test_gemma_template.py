from mlc_llm.conversation_template import ConvTemplateRegistry
from mlc_llm.interface.gen_config import CONV_TEMPLATES
from mlc_llm.model.gemma4.gemma4_config import Gemma4Config
from mlc_llm.model.gemma4.gemma4_model import gemma4_artifact_tasks
from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders


def _gemma4_prompt(system_message: str = "") -> tuple[Conversation, str]:
    conversation = ConvTemplateRegistry.get_conv_template("gemma4_instruction").model_copy(
        deep=True
    )
    conversation.system_message = system_message
    conversation.messages.extend(
        [
            ("user", "Transcribe this audio."),
            ("assistant", None),
        ]
    )
    return conversation, conversation.as_prompt()[0]


def test_gemma4_prompt_without_system_turn():
    conversation, prompt = _gemma4_prompt()

    assert conversation.system_prefix_token_ids == [2]
    assert prompt == "<|turn>user\nTranscribe this audio.<turn|>\n<|turn>model\n"


def test_gemma4_prompt_with_system_turn():
    _, prompt = _gemma4_prompt("Answer briefly.")

    assert prompt == (
        "<|turn>system\nAnswer briefly.<turn|>\n"
        "<|turn>user\nTranscribe this audio.<turn|>\n"
        "<|turn>model\n"
    )


def test_empty_system_message_rendering_is_template_specific():
    conversation = Conversation(
        system_template=f"<system>{MessagePlaceholders.SYSTEM.value}</system>",
        system_message="",
        render_empty_system_message=False,
        roles={"user": "<user>", "assistant": "<assistant>"},
        seps=["</turn>"],
    )
    conversation.messages.extend([("user", "hello"), ("assistant", None)])

    assert conversation.as_prompt() == ["<user>hello</turn><assistant>"]

    conversation.system_message = "rules"
    assert conversation.as_prompt() == ["<system>rules</system><user>hello</turn><assistant>"]

    existing = ConvTemplateRegistry.get_conv_template("olmo2").model_copy(deep=True)
    existing.messages.extend([("user", "hello"), ("assistant", None)])
    assert existing.render_empty_system_message is True
    assert existing.as_prompt() == ["<|system|>\n\n<|user|>\nhello<|endoftext|>\n<|assistant|>\n"]


def test_gemma4_generated_config_and_audio_prompt_contract():
    conversation, _ = _gemma4_prompt()
    config_json = conversation.to_json_dict()
    audio_prompt = gemma4_artifact_tasks(Gemma4Config.from_dict({}))["chat.completions"]["inputs"][
        "audio"
    ]["prompt"]

    assert "gemma4_instruction" in CONV_TEMPLATES
    assert config_json["name"] == "gemma4_instruction"
    assert config_json["render_empty_system_message"] is False
    assert Conversation.from_json_dict(config_json).render_empty_system_message is False
    assert config_json["roles"] == {
        "user": "<|turn>user",
        "assistant": "<|turn>model",
    }
    assert config_json["seps"] == ["<turn|>\n"]
    assert config_json["stop_token_ids"] == [1, 106, 50]
    assert audio_prompt == {
        "prefix_token_ids": [256_000],
        "placeholder_token_id": 258_881,
        "suffix_token_ids": [258_883],
    }
