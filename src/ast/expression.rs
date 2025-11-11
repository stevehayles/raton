use super::{Identifier, Statement};
use crate::{BinaryOperator, UnaryOperator, Value};

/// An expresssion that may be evaluated to produce a value.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Expression {
    /// `null`
    /// `true`
    /// `42`
    /// `3.14`
    /// `"hello"`
    Literal(Value),
    /// `argname`
    /// `varname`
    Variable(Identifier),
    /// See [`UnaryExpression`].
    Unary(UnaryExpression),
    /// See [`BinaryExpression`].
    Binary(BinaryExpression),
    /// See [`CallExpression`].
    Call(CallExpression),
    /// See [`MethodCallExpression`].
    #[cfg(feature = "method_call_expression")]
    MethodCall(MethodCallExpression),
    #[cfg(feature = "if_expression")]
    /// See [`IfExpression`].
    If(IfExpression),
    /// See [`BlockExpression`].
    Block(BlockExpression),
}

/// `-operand`
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct UnaryExpression {
    /// Operator.
    pub operator: UnaryOperator,
    /// Operand.
    pub operand: Box<Expression>,
}

/// `lhs * rhs`
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct BinaryExpression {
    /// Left-hand side operand.
    pub left: Box<Expression>,
    /// Operator.
    pub operator: BinaryOperator,
    /// Right-hand side operand.
    pub right: Box<Expression>,
}

/// `identifier(arg1, arg2)`
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct CallExpression {
    /// Name of the function to call.
    pub identifier: Identifier,
    /// Expressions to evaluate to produce function arguments.
    pub arguments: Vec<Expression>,
}

/// `receiver.identifier(arg1, arg2)`
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg(feature = "method_call_expression")]
#[non_exhaustive]
pub struct MethodCallExpression {
    /// Reciever.
    pub receiver: Box<Expression>,
    /// Name of the method to call.
    pub identifier: Identifier,
    /// Expressions to evaluate to produce function arguments.
    pub arguments: Vec<Expression>,
}

/// `if cond { then_branch }`
/// `if cond { then_branch } else { else_branch }`
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg(feature = "if_expression")]
#[non_exhaustive]
pub struct IfExpression {
    /// Check if this condition is true.
    pub condition: Box<Expression>,
    /// Evaluate if the condition was true.
    pub then_branch: BlockExpression,
    /// Evaluate if the condition was false.
    ///
    /// If absent, a false condition means the [`IfExpression`]
    /// implicitly evaluates to [`Value::Null`].
    pub else_branch: Option<BlockExpression>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
/// A sequence of statements, possibly followed by an expression,
/// in curly brackets.
///
/// `{ stmt1; stmt2; }`
/// `{ stmt1; stmt2; value }`
pub struct BlockExpression {
    /// Statements to execute sequentially.
    pub statements: Vec<Statement>,
    /// Expression to evaluate to produce a value, the value of the block.
    pub value: Option<Box<Expression>>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
/// A sequence of statements, followed by a required expression,
/// in curly brackets.
///
/// `{ stmt1; stmt2; value }`
pub struct ComputeBlock {
    /// Statements to execute sequentially
    pub statements: Vec<Statement>,
    /// Expression that produces the final value (required)
    pub value: Box<Expression>,
}
