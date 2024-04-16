fn main() {
    let mlc_home = env!("MLC_HOME");

    println!("cargo:rustc-link-lib=dylib=mlc_llm_module");
    println!("cargo:rustc-link-search=native={}/build", mlc_home);
}
