use std::fs;
use std::path::{Path, PathBuf};
use std::result;
use tracing::info;
use tvm_rt::{function::Function, Module};

use super::config::*;

#[derive(Debug)]
pub enum ChatModuleError {
    /// Global function in a TVM Module is not found
    GlobalFuncNotFound,
    /// TVM Runtime error
    TvmRuntime(tvm_rt::Error),
}

impl From<tvm_rt::Error> for ChatModuleError {
    fn from(e: tvm_rt::Error) -> Self {
        Self::TvmRuntime(e)
    }
}

pub type Result<T> = result::Result<T, ChatModuleError>;

/// The ChatModule for MLC LLM.
///
/// # Examples
///
/// ```
/// use mlc_llm::chat_module::ChatModule;
///
/// // Create a ChatModule instance
/// let cm = ChatModule::new("Llama-2-7b-chat-hf-q4f16_1", "cuda", None, None).unwrap();
///
/// // Generate a response for a given prompt
/// let output = cm.generate("What is the meaning of life?", None).unwrap();
///
/// // Print prefill and decode performance statistics
/// println!("Statistics: {:?}\n", cm.stats(false).unwrap());
///
/// let output = cm.generate("What is Rust?", None).unwrap();
/// ```
pub struct ChatModule {
    chat_module: Module,
    chat_config: ChatConfig,
}

#[derive(Debug, Copy, Clone)]
pub enum PlaceInPrompt {
    All = 0,
    Begin = 1,
    Middle = 2,
    End = 3,
}

impl PlaceInPrompt {
    pub fn to_value(&self) -> i32 {
        *self as i32
    }
}

/// Parse the input device identifier into device name and id.
///
/// # Parameters
/// * `device` - The device identifier to parse. It can be in the format "device_name" (e.g., "cuda")
/// or "device_name:device_id" (e.g., "cuda:1").
///
/// # Returns
/// * `device_name` - The name of the device.
/// * `device_id` - The id of the device, or 0 if not specified in the input.
fn parse_device_str(device: &str) -> (&str, i32) {
    let device_err_msg = format!(
        "Invalid device name: {}. Please enter the device in the form \
        'device_name:device_id' or 'device_name', where 'device_name' needs to be \
        one of 'cuda', 'metal', 'vulkan', 'rocm', 'opencl', 'auto'.",
        device
    );
    let device_args: Vec<&str> = device.split(':').collect();
    match device_args.len() {
        1 => (device_args[0], 0),
        2 => (device_args[0], device_args[1].parse::<i32>().unwrap()),
        _ => panic!("{}", device_err_msg),
    }
}

/// Use user-provided argument `model` to search for a valid model path.
/// We define "valid" as having an `mlc-chat-config.json` right under the folder.
///
/// # Parameters
/// * `model`: User's input; may be a compiled model's name, or a full path.
///
/// # Returns
/// * `model_path`: A "valid" path to model folder with `mlc-chat-config.json` existing under it.
/// * `chat_file`: The path to the `mlc-chat-config.json` file.
///
/// # Panics
/// * If a valid model_path cannot be found.
pub fn get_model_path(model: &str) -> (PathBuf, PathBuf) {
    // Note that the order of this list corresponds to our search priority
    let candidate_paths = vec![
        PathBuf::from(model),                              // full path, or just the name
        PathBuf::from(format!("{}/params", model)), // Default directory after mlc_llm.build_model()
        PathBuf::from(format!("dist/prebuilt/{}", model)), // Using prebuilt workflow
        PathBuf::from(format!("dist/{}/params", model)), // Default directory after mlc_llm.build_model() in the current path
        PathBuf::from(format!("dist/prebuilt/mlc-chat-{}", model)), // Also prebuilt workflow, but missed prefix
    ];

    // Look for the first folder that has `mlc-chat-config.json` under it
    for candidate in &candidate_paths {
        let chat_file = candidate.join("mlc-chat-config.json");
        if chat_file.is_file() {
            info!(
                "Using model folder: {:?}",
                candidate.canonicalize().unwrap()
            );
            info!(
                "Using mlc chat config: {:?}",
                chat_file.canonicalize().unwrap()
            );
            return (candidate.clone(), chat_file);
        }
    }

    let mut found_folder = false;
    let mut valid_dir_str = String::new();
    for candidate in &candidate_paths {
        if candidate.is_dir() {
            valid_dir_str += &format!("- {:?}\n", candidate.canonicalize().unwrap());
            found_folder = true;
        }
    }

    if found_folder {
        // Error 1: there is a folder, but not an mlc-llm model folder (E1)
        let err_msg = format!(
            "The model folder provided does not seem to refer to a valid mlc-llm model folder.\n\
            Specifically, we cannot find `mlc-chat-config.json`, a required file. You should \
            provide a path that contains the file.\n\
            According to your input `model`, we looked at folder(s):\n\
            {}\n\
            MLC-Chat consumes models that are processed by the MLC-LLM build process.\n\
            ",
            valid_dir_str,
        );
        panic!("{}", err_msg);
    } else {
        // Error 2: cannot find a folder (E0)
        let all_paths_str = candidate_paths
            .iter()
            .map(|path| format!("- {}\n", path.display()))
            .collect::<String>();
        let err_msg = format!(
            "Cannot find the model folder. We searched over the following possible paths:\n\
            {}\n\
            You can try to pass in `model=/path/to/your-model-path`, and confirm \
            that it contains `mlc-chat-config.json`, among other essential files.\n\
            ",
            all_paths_str,
        );
        panic!("{}", err_msg);
    }
}

/// Read in the config file in model path, then potentially override with user input.
///
/// # Parameters:
/// * `config_file_path`: &Path
///   `chat_file` returned by a function like `get_model_path()`.
fn get_chat_config(
    config_file_path: &Path,
) -> result::Result<ChatConfig, Box<dyn std::error::Error>> {
    // Read the base configuration from the file
    let file_contents = fs::read_to_string(config_file_path)?;
    let final_chat_config = ChatConfig::from_json(&file_contents)?;
    Ok(final_chat_config)
}

fn get_lib_module_path(
    model: &str,
    model_path: &Path,
    chat_config: &ChatConfig,
    model_lib_path: Option<&str>,
    device_name: &str,
    config_file_path: &Path,
) -> PathBuf {
    // 1. Use user's model_lib_path if provided
    if let Some(lib_path) = model_lib_path {
        let path = Path::new(lib_path);
        if path.is_file() {
            info!("Using library model: {:?}", path);
            return path.to_path_buf();
        } else {
            panic!(
                "The `model_lib_path` you passed in is not a file: {:?}.",
                lib_path
            );
        }
    }

    // 2. Generate all possible file names according to OS
    let mut candidate_paths = Vec::new();
    if let Some(model_lib) = &chat_config.model_lib {
        let candidate_lib_names: Vec<String> = if cfg!(target_os = "linux") {
            vec![format!("{}-{}.so", model_lib, device_name)]
        } else if cfg!(target_os = "macos") {
            vec![
                format!("{}-{}.dylib", model_lib, device_name),
                format!("{}-{}.so", model_lib, device_name),
            ]
        } else if cfg!(target_os = "windows") {
            vec![format!("{}-{}.dll", model_lib, device_name)]
        } else {
            vec![
                format!("{}-{}.dylib", model_lib, device_name),
                format!("{}-{}.so", model_lib, device_name),
                format!("{}-{}.dll", model_lib, device_name),
            ]
        };

        // 3. Generate possible model library paths
        let pardir_model_path = model_path.parent().unwrap();
        for lib_name in &candidate_lib_names {
            let paths: Vec<String> = vec![
                lib_name.clone(),
                format!("dist/prebuilt/lib/{}", lib_name),
                format!("dist/{}/{}", model, lib_name),
                model_path.join(lib_name).to_string_lossy().into_owned(),
                pardir_model_path
                    .join(lib_name)
                    .to_string_lossy()
                    .into_owned(),
            ];

            candidate_paths.extend(paths);
        }

        // 4. Search for model library
        for candidate in &candidate_paths {
            let candidate_path = Path::new(candidate);
            if candidate_path.is_file() {
                info!("Using library model: {:?}", candidate_path);
                return candidate_path.to_path_buf();
            }
        }

        // 5. Error
        let mut err_msg = format!(
            "Cannot find the model library that corresponds to `{:?}`.\n\
             `{:?}` is either provided in the `chat_config` \
             you passed in, or specified in {:?}.\n\
             We searched over the following possible paths: \n",
            model_lib, model_lib, config_file_path
        );
        for candidate in &candidate_paths {
            err_msg += &format!("- {}\n", candidate);
        }
        err_msg += &format!(
            "If you would like to directly specify the model library path, you may \
             consider passing in the `ChatModule.model_lib_path` parameter."
        );

        panic!("{}", err_msg);
    } else {
        panic!("Cannot find the model library, you need to either pass it in, or specify in the chat_config file.");
    }
}

impl ChatModule {
    pub fn new(model: &str, device: &str, model_lib_path: Option<&str>) -> Result<Self> {
        let device_err_msg = format!(
            "Invalid device name: {}. Please enter the device in the form \
            'device_name:device_id' or 'device_name', where 'device_name' needs to be \
            one of 'cuda', 'metal', 'vulkan', 'rocm', 'opencl', 'auto'.",
            device
        );

        let (device_name, device_id) = parse_device_str(device);

        // 1. Get device name and id
        let device_type = match device_name {
            "cude" => 2,
            "opencl" => 4,
            "vulkan" => 7,
            "metal" => 8,
            "rocm" => 10,
            _ => panic!("{}", device_err_msg),
        };

        static GLOBAL_FUNC_NAME: &str = "mlc.llm_chat_create";
        let f = Function::get(GLOBAL_FUNC_NAME).ok_or(ChatModuleError::GlobalFuncNotFound)?;
        let m: Module = f
            .invoke(vec![device_type.into(), device_id.into()])
            .unwrap()
            .try_into()
            .expect("call should succeed");

        // 2. Look up the model path
        let (model_path, config_file_path) = get_model_path(model);

        // 3. Instantiate chat_config
        let chat_config = get_chat_config(&config_file_path).unwrap();

        // 4. Look up the model library
        let model_lib_path = get_lib_module_path(
            model,
            &model_path,
            &chat_config,
            model_lib_path,
            device_name,
            &config_file_path,
        );

        let chat_mod = Self {
            chat_module: m,
            chat_config: chat_config,
        };
        let model_lib_str = model_lib_path.as_path().display().to_string();
        let model_path_str = model_path.as_path().display().to_string();
        chat_mod
            .reload(&model_lib_str, &model_path_str, "")
            .unwrap();
        Ok(chat_mod)
    }

    /// Reload the chat module from the given library and model path.
    fn reload(&self, lib: &str, model_path: &str, app_config_json: &str) -> Result<()> {
        let f = self.chat_module.get_function("reload", false)?;
        f.invoke(vec![lib.into(), model_path.into(), app_config_json.into()])?;
        Ok(())
    }

    /// Reset the chat session, clear all chat history, and potentially
    /// override the original `mlc-chat-config.json`.
    pub fn reset_chat(&self) -> Result<()> {
        // TODO: add optional user-specified ChatConfig
        let f = self.chat_module.get_function("reset_chat", false)?;
        f.invoke(vec![])?;
        Ok(())
    }

    /// Get the runtime stats of the encoding step, decoding step (and embedding step if exists)
    /// of the chat module in text form.
    pub fn stats(&self, verbose: bool) -> Result<String> {
        if verbose {
            let f = self
                .chat_module
                .get_function("verbose_runtime_stats_text", false)?;
            let res: String = f.invoke(vec![])?.try_into().expect("call should succeed");
            return Ok(res);
        }
        let f = self.chat_module.get_function("runtime_stats_text", false)?;
        let res: String = f.invoke(vec![])?.try_into().expect("call should succeed");
        return Ok(res);
    }

    /// Check if the stop condition is met for the current round.
    fn stopped(&self) -> Result<bool> {
        let f = self.chat_module.get_function("stopped", false)?;
        let res: bool = f.invoke(vec![])?.try_into().expect("call should succeed");
        Ok(res)
    }

    /// Get the output message in the current round.
    fn get_message(&self) -> Result<String> {
        let f = self.chat_module.get_function("get_message", false)?;
        let res: String = f.invoke(vec![])?.try_into().expect("call should succeed");
        Ok(res)
    }

    /// Decode the next token, the decoding result is stored in a buffer and
    /// can be retrieved by [get_message].
    fn decode(&self, generation_config: Option<&GenerationConfig>) -> Result<()> {
        let generation_config_str = match generation_config {
            Some(config) => serde_json::to_string(config).unwrap(),
            None => {
                let config = GenerationConfig::from_chat_config(&self.chat_config);
                serde_json::to_string(&config).unwrap()
            }
        };
        let f = self.chat_module.get_function("decode", false)?;
        f.invoke(vec![generation_config_str.into()])?;
        Ok(())
    }

    /// A high-level method that returns the full response from the chat module given a user
    /// prompt. User can optionally specify which callback method to use upon receiving the
    /// response.
    pub fn generate(
        &self,
        prompt: &str,
        generation_config: Option<&GenerationConfig>,
    ) -> Result<Vec<String>> {
        // TODO: add progress_callback
        let mut new_msgs: Vec<String> = vec![];
        let mut num_return_sequences: usize = 1;

        if let Some(gc) = generation_config {
            if let Some(n) = gc.n {
                num_return_sequences = n;
            }
        }

        for _ in 0..num_return_sequences {
            self.reset_chat().unwrap();
            self.prefill(prompt, true, PlaceInPrompt::All, generation_config)
                .unwrap();

            while !self.stopped().unwrap() {
                self.decode(generation_config)?;
            }
            let new_msg = self.get_message().unwrap();
            new_msgs.push(new_msg);
        }

        Ok(new_msgs)
    }

    /// Run prefill stage for a given input and optionally decode the first output token.
    /// User can decide where to place the input in the prompt.
    fn prefill(
        &self,
        input: &str,
        decode_next_token: bool,
        place_in_promt: PlaceInPrompt,
        generation_config: Option<&GenerationConfig>,
    ) -> Result<()> {
        let generation_config_str = match generation_config {
            Some(config) => serde_json::to_string(config).unwrap(),
            None => {
                let config = GenerationConfig::from_chat_config(&self.chat_config);
                serde_json::to_string(&config).unwrap()
            }
        };

        let f = self.chat_module.get_function("prefill", false)?;
        f.invoke(vec![
            input.into(),
            (&decode_next_token).into(),
            place_in_promt.to_value().into(),
            generation_config_str.into(),
        ])?;
        Ok(())
    }
}

