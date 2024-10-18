#include <tvm/runtime/registry.h>
#include <tvm/runtime/module.h>
#include <thread>
#include <vector>
#include <cstdlib> // for exit()
#include "json_ffi/json_ffi_engine.h"
#include "support/result.h"


using namespace tvm::runtime;
using namespace mlc::llm::json_ffi;

auto start = std::chrono::high_resolution_clock::now();
auto end = std::chrono::high_resolution_clock::now();
int counter = 0;
int glob_max_tokens = 0;
int iterations = 1;

void exitProgram() {
    std::this_thread::sleep_for(std::chrono::seconds(3)); // Sleep for 3 seconds
    std::cout << "Exiting program after 3 seconds..." << std::endl;
    exit(0); // Exit the program
}

// Define the callback function that processes the responses
void RequestStreamCallback(const std::string& response) {
    mlc::llm::Result<ChatCompletionStreamResponse> stream_response_result = 
        ChatCompletionStreamResponse::FromJSON(
            response.substr(1, response.size() - 2)
        );

    if (stream_response_result.IsOk()) {
        auto unwrp_res = stream_response_result.Unwrap();

        if (unwrp_res.choices.size() > 1) {
            std::cerr << response << "(!!!More choices!!!)\n";
        } else {
            std::string chunk_text = ".";
            if (iterations == 1) {
                chunk_text = unwrp_res.choices[0].delta.content.Text();
            }
            std::cout << chunk_text;
            counter++;
            if (counter >= (glob_max_tokens - 20)) {
                end = std::chrono::high_resolution_clock::now();
            }
        }
    } else {
        std::string chunk_text = stream_response_result.UnwrapErr();
        std::cerr << "Error parsing response." + chunk_text + "\n";
        std::cerr << response << "\n";
    }
}


void benchmark_llm(
    const std::string& model_path, 
    const std::string& modellib_path, 
    const std::string& mode, 
    const int device_type,
    const int timeout,
    const std::string& input_text) {

    int device_id = 0;
    tvm::runtime::PackedFunc request_stream_callback = tvm::runtime::PackedFunc(
        [](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* ret) {
            std::string response = args[0];
            RequestStreamCallback(response);
        }
    );


    const PackedFunc* create_engine_func = tvm::runtime::Registry::Get("mlc.json_ffi.CreateJSONFFIEngine");

    if (create_engine_func == nullptr) {
        throw std::runtime_error("Cannot find mlc.json_ffi.CreateJSONFFIEngine in the registry.");
    }

    // Call the function and get the module (which holds JSONFFIEngineImpl)
    Module engine_mod = (*create_engine_func)();

    // Cast the module to JSONFFIEngineImpl
    auto* engine = dynamic_cast<mlc::llm::json_ffi::JSONFFIEngineImpl*>(engine_mod.operator->());
    if (!engine) {
        throw std::runtime_error("Failed to cast to JSONFFIEngineImpl.");
    }

    engine->InitBackgroundEngine(device_type, device_id, request_stream_callback);

    std::thread background_stream_back_loop([&engine]() {
        engine->RunBackgroundStreamBackLoop();
    });

    std::thread background_loop([&engine]() {
        engine->RunBackgroundLoop();
    });


    // Now call the Reload function
    std::string engine_json = "{\"model\":\"" + model_path + "\", \"model_lib\":\"" + modellib_path + "\", \"mode\": \"" + mode + "\"}";
    std::cerr << engine_json << std::endl;
    engine->Reload(engine_json);
    std::cerr << "\engine->Reload\n";
    
    
    // Prepare input
    std::string request_json = "{\"messages\":[{\"role\":\"user\",\"content\":\"" + input_text + "\"}], \"max_tokens\": " + std::to_string(glob_max_tokens) + "}";
    std::string request_id = "benchmark_request";

    for(int i = 1; i <= iterations; i++) {
        counter = 0;
        // Measure inference time
        start = std::chrono::high_resolution_clock::now();
        engine->ChatCompletion(request_json, request_id);

        // std::cerr << "\nRunning in background. Sleeping main thread " + timeout + "s... Wait for text response (3s - 1m).\n\n";
        std::this_thread::sleep_for(std::chrono::seconds(timeout));
        // std::cerr << "\nWakeup...\n";

        // std::cerr << "\nAborting...\n";
        engine->Abort(request_id);
        std::this_thread::sleep_for(std::chrono::seconds(3));

        std::cerr << i << " Max tokens:" << glob_max_tokens << "; Counter:" << counter << "\n\n";

        std::chrono::duration<double> elapsed = end - start;
        std::cerr << i << " Inference time: " << elapsed.count() << " seconds" << std::endl;

        std::cerr << i << " End-to-end decoded avg token/s: " << std::to_string(counter / elapsed.count()) << "\n";
    }


    engine->ExitBackgroundLoop();
    std::this_thread::sleep_for(std::chrono::seconds(3));

    background_stream_back_loop.join();
    background_loop.join();

    std::cerr << "engine->Unload\n";
    std::thread(exitProgram).detach();
    engine->Unload();
    return;
}

int main(int argc, char* argv[]) {
    if (argc != 9) {
        std::cerr << "Usage: " << argv[0] << " <1:model_path:str> <2:model_lib_path:str> <3:mode:str> <4:device_type:int> <5:timeout:int> <6:max_tokens:int> <7:input_text:str> <8:iterations>" << std::endl
        << "Device types: kDLCPU = 1; kDLOpenCL = 4; kDLVulkan = 7;\n" << "Be carefull with number of iterations.\n 1 iteration gives you text outputs.";
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string modellib_path = argv[2];
    std::string mode = argv[3];
    int device_type = std::stoi(argv[4]);
    int timeout = std::stoi(argv[5]);
    glob_max_tokens = std::stoi(argv[6]);
    std::string input_text = argv[7];
    iterations = std::stoi(argv[8]);

    benchmark_llm(model_path, modellib_path, mode, device_type, timeout, input_text);  

    return 0;
}