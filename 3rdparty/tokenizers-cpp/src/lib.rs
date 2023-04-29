// A simple C wrapper of tokenzier library
use std::str::FromStr;
use tokenizers::tokenizer::Tokenizer;

pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
    encode_ids: Vec<u32>,
    decode_str: String,
}

impl TokenizerWrapper {
    pub fn from_str(json: &str) -> TokenizerWrapper {
        TokenizerWrapper {
            tokenizer: Tokenizer::from_str(json).unwrap().into(),
            encode_ids: Vec::new(),
            decode_str: String::new(),
        }
    }

    pub fn encode(&mut self, text: &str, add_special_tokens: bool)  {
        self.encode_ids = Vec::from(
            self.tokenizer.encode(text, add_special_tokens).unwrap().get_ids()
        );
    }

    pub fn decode(&mut self, ids: Vec<u32>, skip_special_tokens: bool) {
        self.decode_str = self.tokenizer.decode(ids, skip_special_tokens).unwrap();
    }

}

#[no_mangle]
extern "C" fn tokenizers_new_from_str(
    input_cstr: *const u8,
    len: usize
) -> *mut TokenizerWrapper {
    unsafe {
        let json = std::str::from_utf8(
            std::slice::from_raw_parts(input_cstr, len)
        ).unwrap();
        return Box::into_raw(
            Box::new(TokenizerWrapper::from_str(json))
        );
    }
}

#[no_mangle]
extern "C" fn tokenizers_encode(
    handle: *mut TokenizerWrapper,
    input_cstr: *const u8,
    len: usize,
    add_special_tokens: i32
) {
    unsafe {
        let input_data = std::str::from_utf8(
            std::slice::from_raw_parts(input_cstr, len)
        ).unwrap();
        (*handle).encode(input_data, add_special_tokens != 0);
    }
}

#[no_mangle]
extern "C" fn tokenizers_get_encode_ids(
    handle: *mut TokenizerWrapper,
    out_data: *mut *mut u32,
    out_len: *mut usize) {
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
    skip_special_tokens: i32
) {
    unsafe {
        let input_data = Vec::from(
            std::slice::from_raw_parts(input_ids, len)
        );
        (*handle).decode(input_data, skip_special_tokens != 0);
    }
}

#[no_mangle]
extern "C" fn tokenizers_get_decode_str(
    handle: *mut TokenizerWrapper,
    out_cstr: *mut *mut u8,
    out_len: *mut usize){
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
