//! Compile a program abstract-syntax-tree to bytecode.

mod code_generator;
pub use code_generator::*;
mod parser;
pub use parser::*;
//mod precedence;
