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
            "/opt/homebrew/opt/lapack/lib",
            "/usr/local/opt/lapack/lib",
            "/opt/homebrew/opt/openblas/lib",
            "/usr/local/opt/openblas/lib",
            "/opt/homebrew/opt/arpack/lib",
            "/usr/local/opt/arpack/lib",
            "/opt/homebrew/opt/suite-sparse/lib",
            "/usr/local/opt/suite-sparse/lib",
        ] {
            if Path::new(candidate).exists() {
                println!("cargo:rustc-link-search=native={candidate}");
                println!("cargo:rustc-link-arg=-Wl,-rpath,{candidate}");
            }
        }
    }
}
