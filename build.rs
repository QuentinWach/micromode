use std::path::Path;

fn main() {
    #[cfg(target_os = "macos")]
    {
        if std::env::var_os("CARGO_FEATURE_ARPACK_BACKEND").is_none() {
            return;
        }
        for candidate in [
            "/opt/homebrew/opt/gcc/lib/gcc/current",
            "/usr/local/opt/gcc/lib/gcc/current",
        ] {
            if Path::new(candidate).join("libgcc_s.1.1.dylib").exists() {
                println!("cargo:rustc-link-search=native={candidate}");
                println!("cargo:rustc-link-arg=-Wl,-rpath,{candidate}");
                break;
            }
        }
    }
}
