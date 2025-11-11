//#![cfg_attr(test, feature(test))]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! # Ratón 🐁
//! A tiny, highly modular, embeddable, dynamically typed scripting language with a bytecode VM, intended for use in games.

pub mod ast;
pub mod bytecode;
mod common;
pub mod compiler;
pub mod runtime;
pub use common::*;
pub mod prelude;

#[cfg(all(test, not(miri)))]
mod benches;
