//! Abstract syntax tree type definitions.

mod expression;
pub use expression::*;
mod statement;
pub use statement::*;

/// Variable or function name.
pub type Identifier = String;

/// A function with a name, parameters, and a body.
///
/// fn identifier(params) { body }
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct Function {
    /// The name that can be used to call the function.
    pub identifier: Identifier,
    /// The names of the function parameters.
    pub arguments: Vec<Identifier>,
    /// The statementse that will be executed when the function is called.
    pub body: BlockExpression,
}

/// A channel declaration in the program
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Channel {
    /// The name that can be used to reference the channel.
    pub identifier: Identifier,
    /// The identifiers of the required input channels
    pub inputs: Vec<Identifier>,
    /// The computation block ending in a required expression
    pub compute: ComputeBlock,
    /// An option default value in the event of a failed computation
    pub default: Option<Expression>,
    /// A conditional check of when to calculate the output
    pub when: Option<Expression>,
}

/// A abstract syntax tree representing a program.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct Program {
    /// Functions defined in the program.
    pub functions: Vec<Function>,
    /// Channels defined in the program
    pub channels: Vec<Channel>,
}
