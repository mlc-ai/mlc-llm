extern crate mlc_llm;

use mlc_llm::chat_module::{ChatMessage, ChatModule};

fn main() {
    // Single prompt example
    let cm = ChatModule::new("/path/to/Llama2-13B-q8f16_1", "rocm", None).unwrap();
    let output = cm.generate("what is the meaning of life?", None).unwrap();
    println!("resp: {:?}", output);
    println!("stats: {:?}", cm.stats(false));

    // Multiple prompts example
    let message1 = ChatMessage {
        role: "user".to_owned(),
        content: "suppose we already have projects llama, alpaca and vicuna, what do you think would be a great name for the next project?".to_string(),
    };
    let message2 = ChatMessage {
        role: "assistant".to_owned(),
        content: "based on the previous projects, a possible name for the next project could be \"cervidae\" which is the scientific name for deer family. this name reflects the collaboration and teamwork involved in the development of the project, and also nods to the previous projects that have been developed by the team.".to_string(),
    };
    let message3 = ChatMessage {
        role: "user".to_owned(),
        content: "I like cervidae, but the name is too long!".to_string(),
    };
    let message4 = ChatMessage {
        role: "assistant".to_owned(),
        content: "In that case, a shorter and catchier name for the next project could be \"DeerRun\" which plays on the idea of the project being fast and efficient, just like a deer running through the woods. This name is memorable and easy to pronounce, making it a good choice for a project name.".to_string(),
    };
    let message5 = ChatMessage {
        role: "user".to_owned(),
        content: "Summarize our conversations.".to_string(),
    };

    let messages = vec![message1, message2, message3, message4, message5];

    let output = cm.generate(messages, None).unwrap();
    println!("resp: {:?}", output);
    println!("stats: {:?}", cm.stats(false));
}
