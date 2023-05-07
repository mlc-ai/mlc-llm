// A simple C wrapper of tokenzier library
use serde_json::Value;
use std::{collections::HashMap, str::FromStr};
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::tokenizer::Tokenizer;

pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
    encode_ids: Vec<u32>,
    decode_str: String,
}

pub type Vocab = HashMap<String, u32>;
pub type Merges = Vec<(String, String)>;

impl TokenizerWrapper {
    pub fn from_str(json: &str) -> TokenizerWrapper {
        TokenizerWrapper {
            tokenizer: Tokenizer::from_str(json).unwrap().into(),
            encode_ids: Vec::new(),
            decode_str: String::new(),
        }
    }

    pub fn byte_level_bpe_from_str(
        vocab: &str,
        merges: &str,
        added_tokens: &str,
    ) -> TokenizerWrapper {
        let vocab_json: Value = serde_json::from_str(vocab).unwrap();
        let added_tokens_json: Value = serde_json::from_str(added_tokens).unwrap();
        let mut vocab = HashMap::new();
        match vocab_json {
            Value::Object(m) => {
                for (token, id) in m {
                    if let Value::Number(id) = id {
                        let id = id.as_u64().unwrap() as u32;
                        vocab.insert(token, id);
                    }
                }
            }
            _ => panic!("Invalid vocab.json file."),
        };
        match added_tokens_json {
            Value::Object(m) => {
                for (token, id) in m {
                    if let Value::Number(id) = id {
                        let id = id.as_u64().unwrap() as u32;
                        vocab.insert(token, id);
                    }
                }
            }
            _ => panic!("Invalid added_tokens.json file."),
        }

        let merges = merges
            .lines()
            .filter(|line| !line.starts_with("#version"))
            .map(|line| {
                let parts = line.split(' ').collect::<Vec<_>>();
                if parts.len() != 2 {
                    panic!("Invalid merges.txt file.")
                }
                return (parts[0].to_string(), parts[1].to_string()); // Add the `return` keyword here
            })
            .collect::<Vec<(String, String)>>();
        let byte_level = ByteLevel::new(
            /*add_prefix_space=*/ false, /*trim_offsets=*/ false,
            /*use_regex=*/ false,
        );
        let mut tokenizer = Tokenizer::new(BPE::new(vocab, merges));
        tokenizer
            .with_pre_tokenizer(byte_level)
            .with_decoder(byte_level);
        TokenizerWrapper {
            tokenizer: tokenizer,
            encode_ids: Vec::new(),
            decode_str: String::new(),
        }
    }

    pub fn encode(&mut self, text: &str, add_special_tokens: bool) {
        self.encode_ids = Vec::from(
            self.tokenizer
                .encode(text, add_special_tokens)
                .unwrap()
                .get_ids(),
        );
    }

    pub fn decode(&mut self, ids: Vec<u32>, skip_special_tokens: bool) {
        self.decode_str = self.tokenizer.decode(ids, skip_special_tokens).unwrap();
    }
}

#[no_mangle]
extern "C" fn tokenizers_new_from_str(input_cstr: *const u8, len: usize) -> *mut TokenizerWrapper {
    unsafe {
        let json = std::str::from_utf8(std::slice::from_raw_parts(input_cstr, len)).unwrap();
        return Box::into_raw(Box::new(TokenizerWrapper::from_str(json)));
    }
}

#[no_mangle]
extern "C" fn byte_level_bpe_tokenizers_new_from_str(
    input_vocab_str: *const u8,
    len_vocab: usize,
    input_merges_str: *const u8,
    len_merges: usize,
    input_added_tokens_str: *const u8,
    len_added_tokens: usize,
) -> *mut TokenizerWrapper {
    unsafe {
        let vocab =
            std::str::from_utf8(std::slice::from_raw_parts(input_vocab_str, len_vocab)).unwrap();
        let merges =
            std::str::from_utf8(std::slice::from_raw_parts(input_merges_str, len_merges)).unwrap();
        let added_tokens = std::str::from_utf8(std::slice::from_raw_parts(
            input_added_tokens_str,
            len_added_tokens,
        ))
        .unwrap();
        return Box::into_raw(Box::new(TokenizerWrapper::byte_level_bpe_from_str(
            vocab,
            merges,
            added_tokens,
        )));
    }
}

#[no_mangle]
extern "C" fn tokenizers_encode(
    handle: *mut TokenizerWrapper,
    input_cstr: *const u8,
    len: usize,
    add_special_tokens: i32,
) {
    unsafe {
        let input_data = std::str::from_utf8(std::slice::from_raw_parts(input_cstr, len)).unwrap();
        (*handle).encode(input_data, add_special_tokens != 0);
    }
}

#[no_mangle]
extern "C" fn tokenizers_get_encode_ids(
    handle: *mut TokenizerWrapper,
    out_data: *mut *mut u32,
    out_len: *mut usize,
) {
    unsafe {
        *out_data = (*handle).encode_ids.as_mut_ptr();
        *out_len = (*handle).encode_ids.len()
    }
}

#[no_mangle]
extern "C" fn tokenizers_decode(
    handle: *mut TokenizerWrapper,
    input_ids: *const u32,
    len: usize,
    skip_special_tokens: i32,
) {
    unsafe {
        let input_data = Vec::from(std::slice::from_raw_parts(input_ids, len));
        (*handle).decode(input_data, skip_special_tokens != 0);
    }
}

#[no_mangle]
extern "C" fn tokenizers_get_decode_str(
    handle: *mut TokenizerWrapper,
    out_cstr: *mut *mut u8,
    out_len: *mut usize,
) {
    unsafe {
        *out_cstr = (*handle).decode_str.as_mut_ptr();
        *out_len = (*handle).decode_str.len();
    }
}

#[no_mangle]
extern "C" fn tokenizers_free(wrapper: *mut TokenizerWrapper) {
    unsafe {
        drop(Box::from_raw(wrapper));
    }
}
