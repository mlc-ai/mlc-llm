fn main() {
    println!("cargo:rustc-link-lib=dylib=mlc_llm_module");
    println!("cargo:rustc-link-search=native={}/build", env!("MLC_HOME"));
}
