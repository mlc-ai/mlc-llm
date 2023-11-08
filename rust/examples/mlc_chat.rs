extern crate mlc_llm;

use mlc_llm::chat_module::ChatModule;

fn main() {
    let cm = ChatModule::new("/path/to/Llama2-13B-q8f16_1", "rocm", None).unwrap();
    let output = cm.generate("what is the meaning of life?", None).unwrap();
    println!("resp: {:?}", output);
    println!("stats: {:?}", cm.stats(false));
}
