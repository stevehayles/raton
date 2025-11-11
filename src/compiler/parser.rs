use crate::{ast::*, BinaryOperator, UnaryOperator, Value};
#[allow(unused_imports)]
use nom::{
    branch::alt,
    bytes::complete::{tag, take_until, take_while1},
    character::complete::{alpha1, alphanumeric1, anychar, char, digit1, multispace0},
    combinator::{cut, eof, map, not, opt, peek, recognize, value, verify},
    error::{ContextError, ErrorKind, FromExternalError, ParseError as NomParseError},
    multi::{many0, separated_list0},
    sequence::{delimited, pair, preceded, terminated},
    IResult, Parser as NomParser,
};
use nom_language::{
    error::{convert_error, VerboseError},
    precedence::{binary_op, precedence, unary_op, Assoc, Operation},
};
use std::{
    error::Error,
    fmt::{self, Debug, Display},
};

/// Parses source code into an abstract syntax tree.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct Parser {
    max_depth: u8,
}

impl Default for Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl Parser {
    /// Create a configurable parser.
    pub fn new() -> Self {
        Self { max_depth: 50 }
    }

    /// Return a [`ParseError`] if nesting depth of expressions/statements exceeds this.
    ///
    /// Note: parsing uses the call stack, so stack overflow is a concern.
    ///
    /// Default: 50
    pub fn with_max_depth(mut self, max: u8) -> Self {
        self.max_depth = max;
        self
    }

    /// Parse a program abstract-syntax-tree from source code.
    pub fn parse<'a>(&self, src: &'a str) -> Result<Program, ParseError<'a>> {
        depth_limiter::reset(self.max_depth as u32);

        type E<'a> = nom_language::error::VerboseError<&'a str>;
        let ret = parse_program::<E>(src);

        ret.map_err(|e| ParseError {
            src,
            inner: match e {
                nom::Err::Incomplete(_) => E::from_error_kind(src, ErrorKind::Complete),
                nom::Err::Error(e) | nom::Err::Failure(e) => e,
            },
        })
    }
}

/// A parse error at the given location in the source code.
#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct ParseError<'a> {
    src: &'a str,
    inner: VerboseError<&'a str>,
}

impl<'a> Error for ParseError<'a> {}

impl<'a> Display for ParseError<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&convert_error(self.src, self.inner.clone()))
    }
}

mod depth_limiter {
    use nom::error::{ContextError, ErrorKind, FromExternalError, ParseError as NomParseError};
    use std::sync::atomic::{AtomicU32, Ordering};

    thread_local! {
        static DEPTH: AtomicU32 = const { AtomicU32::new(0) };
    }
    #[must_use]
    pub struct DepthGuard {}

    pub fn reset(max: u32) {
        DEPTH.with(|depth| {
            depth.store(max, Ordering::Relaxed);
        });
    }

    pub fn dive<
        'a,
        E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
    >(
        i: &'a str,
    ) -> Result<DepthGuard, nom::Err<E>> {
        DEPTH.with(|depth| {
            let depth = depth.fetch_sub(1, Ordering::Relaxed);
            if depth == 0 {
                return Err(nom::Err::Failure(E::from_error_kind(
                    i,
                    ErrorKind::TooLarge,
                )));
            }
            Ok(DepthGuard {})
        })
    }

    impl Drop for DepthGuard {
        fn drop(&mut self) {
            DEPTH.with(|depth| {
                depth.fetch_add(1, Ordering::Relaxed);
            })
        }
    }
}

/// Parse a single comment.
#[cfg(any(feature = "single_line_comment", feature = "multi_line_comment"))]
fn comment<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, &'a str, E> {
    preceded(
        char('/'),
        alt((
            #[cfg(feature = "single_line_comment")]
            preceded(
                char('/'),
                terminated(take_until("\n"), alt((tag("\n"), eof))),
            ),
            #[cfg(feature = "multi_line_comment")]
            preceded(char('*'), cut(terminated(take_until("*/"), tag("*/")))),
        )),
    )
    .parse(i)
}

/// Parse several comments.
#[cfg(any(feature = "single_line_comment", feature = "multi_line_comment"))]
fn comments<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, &'a str, E> {
    recognize(many0(terminated(comment, multispace0))).parse(i)
}

/// In-between token parser (spaces and comments).
fn blank<'a, E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>>(
    i: &'a str,
) -> IResult<&'a str, (), E> {
    #[cfg(any(feature = "single_line_comment", feature = "multi_line_comment"))]
    return value((), preceded(multispace0, comments)).parse(i);
    #[cfg(not(any(feature = "single_line_comment", feature = "multi_line_comment")))]
    return value((), multispace0).parse(i);
}

// Whitespace helper
fn ws<
    'a,
    F,
    O,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    inner: F,
) -> impl NomParser<&'a str, Output = O, Error = E>
where
    F: NomParser<&'a str, Output = O, Error = E>,
{
    delimited(blank, inner, blank)
}

// Keyword that must not be the prefix of an ident.
fn keyword<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    k: &'a str,
) -> impl NomParser<&'a str, Output = &'a str, Error = E> {
    terminated(
        tag(k),
        not(verify(peek(anychar), |&c: &char| {
            c.is_ascii_alphanumeric() || c == '_'
        })),
    )
}

fn parse_identifier<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, String, E> {
    let (i, _) = blank(i)?;
    let (i, ident) = recognize((
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    ))
    .parse(i)?;
    let (i, _) = blank(i)?;

    let err = || Err(nom::Err::Error(E::from_error_kind(i, ErrorKind::TooLarge)));

    // Check for keywords
    match ident {
        "let" | "if" | "else" | "while" | "break" | "continue" | "return" | "fn" | "null"
        | "channel" | "inputs" | "compute" | "default" | "when" => err(),
        #[cfg(feature = "bool_type")]
        "true" | "false" => err(),
        _ => Ok((i, ident.to_owned())),
    }
}

fn parse_null<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, Value, E> {
    value(Value::Null, ws(keyword("null"))).parse(i)
}

#[cfg(feature = "bool_type")]
fn parse_bool<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, Value, E> {
    alt((
        value(Value::Bool(true), ws(keyword("true"))),
        value(Value::Bool(false), ws(keyword("false"))),
    ))
    .parse(i)
}

#[cfg(feature = "f32_type")]
fn parse_f32<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, Value, E> {
    let (i, _) = blank(i)?;
    let (i, num_str) = recognize((opt(char('-')), digit1, char('.'), digit1)).parse(i)?;
    let (i, _) = blank(i)?;

    match num_str.parse::<f32>() {
        Ok(f) => Ok((i, Value::F32(f))),
        Err(_) => Err(nom::Err::Error(E::from_error_kind(i, ErrorKind::Verify))),
    }
}

#[cfg(feature = "i32_type")]
fn parse_i32<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, Value, E> {
    let (i, _) = blank(i)?;
    let (i, num_str) = recognize(pair(opt(char('-')), digit1)).parse(i)?;
    let (i, _) = blank(i)?;

    match num_str.parse::<i32>() {
        Ok(n) => Ok((i, Value::I32(n))),
        Err(_) => Err(nom::Err::Error(E::from_error_kind(i, ErrorKind::Verify))),
    }
}

#[cfg(feature = "string_type")]
fn parse_string<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, Value, E> {
    let (i, _) = blank.parse(i)?;
    let (i, _) = char('"').parse(i)?;
    let (i, content) = take_while1(|c| c != '"').parse(i)?;
    let (i, _) = char('"').parse(i)?;
    let (i, _) = blank.parse(i)?;

    Ok((i, Value::String(content.to_string())))
}

fn parse_literal<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, Value, E> {
    alt((
        parse_null,
        #[cfg(feature = "bool_type")]
        parse_bool,
        #[cfg(feature = "f32_type")]
        parse_f32,
        #[cfg(feature = "i32_type")]
        parse_i32,
        #[cfg(feature = "string_type")]
        parse_string,
    ))
    .parse(i)
}

// Postfix operators for precedence parser
#[derive(Clone)]
enum PostfixOp {
    Call(Vec<Expression>),
    #[cfg(feature = "method_call_expression")]
    MethodCall(Identifier, Vec<Expression>),
}

#[cfg(feature = "method_call_expression")]
fn method_call<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, PostfixOp, E> {
    map(
        preceded(
            char('.'),
            (
                cut(parse_identifier),
                cut(delimited(
                    ws(char('(')),
                    separated_list0(ws(char(',')), expression),
                    ws(cut(char(')'))),
                )),
            ),
        ),
        |(i, a)| PostfixOp::MethodCall(i, a),
    )
    .parse(i)
}

fn function_call<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, PostfixOp, E> {
    map(
        delimited(
            ws(char('(')),
            separated_list0(ws(char(',')), expression),
            ws(cut(char(')'))),
        ),
        PostfixOp::Call,
    )
    .parse(i)
}

fn parse_block<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, BlockExpression, E> {
    let (mut i, _) = ws(char('{')).parse(i)?;

    let mut statements = Vec::new();
    let mut value = None;

    loop {
        let (i2, soe) = opt(parse_stmt_or_expr).parse(i)?;
        i = i2;

        if let Some(soe) = soe {
            if value.is_some() {
                return Err(nom::Err::Failure(E::from_error_kind(i, ErrorKind::Verify)));
            }

            match soe {
                StmtOrExpr::Expr(expr) => {
                    value = Some(expr);
                }
                StmtOrExpr::Stmt(stmt) => {
                    statements.push(stmt);
                }
            }
        } else {
            break;
        }
    }

    let (i, _) = ws(cut(char('}'))).parse(i)?;

    Ok((
        i,
        BlockExpression {
            statements,
            value: value.map(Box::new),
        },
    ))
}

// If expression parser
#[cfg(feature = "if_expression")]
fn parse_if<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, Expression, E> {
    let (i, _) = ws(keyword("if")).parse(i)?;
    let (i, cond) = cut(expression).parse(i)?;
    let (i, then_branch) = cut(parse_block).parse(i)?;
    let (i, else_branch) = opt(preceded(ws(keyword("else")), cut(parse_block))).parse(i)?;
    Ok((
        i,
        Expression::If(IfExpression {
            condition: Box::new(cond),
            then_branch,
            else_branch,
        }),
    ))
}

// Primary expression (atom)
fn primary_expr<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, Expression, E> {
    alt((
        map(parse_literal, Expression::Literal),
        #[cfg(feature = "if_expression")]
        parse_if,
        map(parse_block, Expression::Block),
        map(parse_identifier, Expression::Variable),
        delimited(ws(char('(')), expression, ws(cut(char(')')))),
    ))
    .parse(i)
}

// Main expression parser using precedence
fn expression<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, Expression, E> {
    let _guard = depth_limiter::dive(i)?;

    precedence(
        // Prefix operators
        alt((
            #[cfg(feature = "bool_type")]
            unary_op(2, value(UnaryOperator::Not, ws(tag("!")))),
            unary_op(2, value(UnaryOperator::Negate, ws(tag("-")))),
        )),
        // Postfix operators (function calls)
        unary_op(1, alt((
            #[cfg(feature = "method_call_expression")]
            method_call,
            function_call,
        ))),
        // Binary operators with precedence levels
        alt((
            // Level 3: Multiplicative
            binary_op(
                3,
                Assoc::Left,
                alt((
                    value(BinaryOperator::Multiply, ws(tag("*"))),
                    value(BinaryOperator::Divide, ws(tag("/"))),
                    value(BinaryOperator::Modulo, ws(tag("%"))),
                )),
            ),
            // Level 4: Additive
            binary_op(
                4,
                Assoc::Left,
                alt((
                    value(BinaryOperator::Add, ws(tag("+"))),
                    value(BinaryOperator::Subtract, ws(tag("-"))),
                )),
            ),
            // Level 5: Comparison
            #[cfg(feature = "bool_type")]
            binary_op(
                5,
                Assoc::Left,
                alt((
                    value(BinaryOperator::LessThanOrEqual, ws(tag("<="))),
                    value(BinaryOperator::GreaterThanOrEqual, ws(tag(">="))),
                    value(BinaryOperator::LessThan, ws(tag("<"))),
                    value(BinaryOperator::GreaterThan, ws(tag(">"))),
                )),
            ),
            // Level 6: Equality
            #[cfg(feature = "bool_type")]
            binary_op(
                6,
                Assoc::Left,
                alt((
                    value(BinaryOperator::Equal, ws(tag("=="))),
                    value(BinaryOperator::NotEqual, ws(tag("!="))),
                )),
            ),
            // Level 7: Logical AND
            #[cfg(feature = "bool_type")]
            binary_op(7, Assoc::Left, value(BinaryOperator::And, ws(tag("&&")))),
            // Level 8: Logical OR
            #[cfg(feature = "bool_type")]
            binary_op(8, Assoc::Left, value(BinaryOperator::Or, ws(tag("||")))),
        )),
        primary_expr,
        |op: Operation<UnaryOperator, PostfixOp, BinaryOperator, Expression>| -> Result<Expression, ()> {
            use Operation::*;
            match op {
                Prefix(operator, operand) => Ok(Expression::Unary(UnaryExpression { operator, operand: Box::new(operand) })),
                Postfix(Expression::Variable(name), PostfixOp::Call(arguments)) => Ok(Expression::Call(CallExpression{identifier: name, arguments})),
                #[cfg(feature = "method_call_expression")]
                Postfix(receiver, PostfixOp::MethodCall(name, arguments)) => Ok(Expression::MethodCall(MethodCallExpression{receiver: Box::new(receiver), identifier: name, arguments})),
                Postfix(_, PostfixOp::Call(_)) => Err(()),
                Binary(left, operator, right) => Ok(Expression::Binary(BinaryExpression{operator, left: Box::new(left), right: Box::new(right)})),
            }
        },
    )(i)
}

fn parse_let<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, Statement, E> {
    let (i, _) = ws(keyword("let")).parse(i)?;
    let (i, identifier) = cut(parse_identifier).parse(i)?;
    let (i, _) = ws(cut(char('='))).parse(i)?;
    let (i, expression) = cut(expression).parse(i)?;
    let (i, _) = ws(cut(char(';'))).parse(i)?;
    Ok((
        i,
        Statement::Let(LetStatement {
            identifier,
            expression,
        }),
    ))
}

fn parse_assign<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, Statement, E> {
    let (i, identifier) = parse_identifier.parse(i)?;
    let (i, _) = ws(char('=')).parse(i)?;
    let (i, expression) = cut(expression).parse(i)?;
    let (i, _) = ws(cut(char(';'))).parse(i)?;
    Ok((
        i,
        Statement::Assign(AssignStatement {
            identifier,
            expression,
        }),
    ))
}

#[cfg(feature = "while_loop")]
fn parse_while<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, Statement, E> {
    let (i, _) = ws(keyword("while")).parse(i)?;
    let (i, condition) = cut(expression).parse(i)?;
    let (i, _) = ws(cut(char('{'))).parse(i)?;

    // While can be nested without expressions.
    let _guard = depth_limiter::dive(i)?;

    let (i, body) = many0(parse_stmt).parse(i)?;
    let (i, _) = ws(cut(char('}'))).parse(i)?;
    Ok((i, Statement::While(WhileLoop { condition, body })))
}

#[cfg(feature = "while_loop")]
fn parse_break<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, Statement, E> {
    let (i, _) = ws(keyword("break")).parse(i)?;
    let (i, _) = ws(cut(char(';'))).parse(i)?;
    Ok((i, Statement::Break))
}

#[cfg(feature = "while_loop")]
fn parse_continue<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, Statement, E> {
    let (i, _) = ws(keyword("continue")).parse(i)?;
    let (i, _) = ws(cut(char(';'))).parse(i)?;
    Ok((i, Statement::Continue))
}

fn parse_return<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, Statement, E> {
    let (i, _) = ws(keyword("return")).parse(i)?;
    let (i, value) = opt(expression).parse(i)?;
    let (i, _) = ws(cut(char(';'))).parse(i)?;
    Ok((i, Statement::Return(ReturnStatement { value })))
}

#[cfg(feature = "while_loop")]
fn parse_expr_stmt<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, Statement, E> {
    let (i, expr) = expression.parse(i)?;
    let (i, _) = ws(char(';')).parse(i)?;
    Ok((i, Statement::Expression(expr)))
}

#[cfg(feature = "while_loop")]
fn parse_stmt<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, Statement, E> {
    alt((
        parse_let,
        #[cfg(feature = "while_loop")]
        parse_while,
        #[cfg(feature = "while_loop")]
        parse_break,
        #[cfg(feature = "while_loop")]
        parse_continue,
        parse_return,
        parse_assign,
        parse_expr_stmt,
    ))
    .parse(i)
}

enum StmtOrExpr {
    Stmt(Statement),
    Expr(Expression),
}

fn parse_stmt_or_expr<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, StmtOrExpr, E> {
    let (i, expr) = opt(expression).parse(i)?;

    if let Some(expr) = expr {
        let (i, semi) = opt(ws(char(';'))).parse(i)?;
        return Ok((
            i,
            if semi.is_some() {
                StmtOrExpr::Stmt(Statement::Expression(expr))
            } else {
                StmtOrExpr::Expr(expr)
            },
        ));
    }

    let (i, stmt) = alt((
        parse_let,
        #[cfg(feature = "while_loop")]
        parse_while,
        #[cfg(feature = "while_loop")]
        parse_break,
        #[cfg(feature = "while_loop")]
        parse_continue,
        parse_return,
        parse_assign,
    ))
    .parse(i)?;

    Ok((i, StmtOrExpr::Stmt(stmt)))
}

fn parse_function<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, Function, E> {
    let (i, _) = ws(keyword("fn")).parse(i)?;
    let (i, identifier) = cut(parse_identifier).parse(i)?;
    let (i, _) = ws(cut(char('('))).parse(i)?;
    let (i, arguments) = separated_list0(ws(char(',')), parse_identifier).parse(i)?;
    let (i, _) = ws(cut(char(')'))).parse(i)?;
    let (i, body) = parse_block(i)?;
    Ok((
        i,
        Function {
            identifier,
            arguments,
            body,
        },
    ))
}

/// Channel parsing
fn parse_identifier_list<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, Vec<String>, E> {
    delimited(
        ws(char('[')),
        separated_list0(ws(char(',')), parse_identifier),
        ws(cut(char(']'))),
    )
    .parse(i)
}

fn parse_channel_inputs<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, Vec<String>, E> {
    let (i, _) = ws(keyword("inputs")).parse(i)?;
    let (i, _) = ws(cut(char(':'))).parse(i)?;
    let (i, inputs) = cut(parse_identifier_list).parse(i)?;
    Ok((i, inputs))
}

fn parse_compute_block<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, ComputeBlock, E> {
    let (mut i, _) = ws(char('{')).parse(i)?;

    let mut statements = Vec::new();

    // Parse statements until we hit a final expression
    loop {
        // Try to parse a statement
        let (i2, stmt) = opt(alt((
            parse_let,
            #[cfg(feature = "while_loop")]
            parse_while,
            parse_assign,
            // Don't allow return/break/continue in compute blocks
        )))
        .parse(i)?;

        if let Some(stmt) = stmt {
            statements.push(stmt);
            i = i2;
        } else {
            break;
        }
    }

    // Now we MUST have a final expression (no semicolon)
    let (i, value) = cut(expression).parse(i)?;
    let (i, _) = ws(cut(char('}'))).parse(i)?;

    Ok((
        i,
        ComputeBlock {
            statements,
            value: Box::new(value),
        },
    ))
}

fn parse_channel_compute<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, ComputeBlock, E> {
    let (i, _) = ws(keyword("compute")).parse(i)?;
    let (i, _) = ws(cut(char(':'))).parse(i)?;

    // Allow either a compute block or wrap a simple expression
    let (i, compute) = cut(alt((
        parse_compute_block,
        map(expression, |expr| ComputeBlock {
            statements: vec![],
            value: Box::new(expr),
        }),
    )))
    .parse(i)?;

    Ok((i, compute))
}

fn parse_channel_default<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, Expression, E> {
    let (i, _) = ws(keyword("default")).parse(i)?;
    let (i, _) = ws(cut(char(':'))).parse(i)?;
    let (i, expr) = cut(expression).parse(i)?;
    Ok((i, expr))
}

fn parse_channel_when<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, Expression, E> {
    let (i, _) = ws(keyword("when")).parse(i)?;
    let (i, _) = ws(cut(char(':'))).parse(i)?;
    let (i, expr) = cut(expression).parse(i)?;
    Ok((i, expr))
}

fn parse_channel<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> IResult<&'a str, Channel, E> {
    let (i, _) = ws(keyword("channel")).parse(i)?;
    let (i, identifier) = cut(parse_identifier).parse(i)?;
    let (mut i, _) = ws(cut(char('{'))).parse(i)?;

    // Parse channel body - order doesn't matter
    let mut inputs = None;
    let mut compute = None;
    let mut default = None;
    let mut when = None;

    loop {
        // Try to parse each field
        let (i2, field) = opt(alt((
            map(parse_channel_inputs, |v| ("inputs", v, None, None)),
            map(parse_channel_compute, |v| {
                ("compute", vec![], Some(v), None)
            }),
            map(parse_channel_default, |v| {
                ("default", vec![], None, Some(v))
            }),
            map(parse_channel_when, |v| ("when", vec![], None, Some(v))),
        )))
        .parse(i)?;

        if let Some((field_name, inp, comp, def_or_when)) = field {
            match field_name {
                "inputs" => {
                    if inputs.is_some() {
                        return Err(nom::Err::Failure(E::from_error_kind(i2, ErrorKind::Verify)));
                    }
                    inputs = Some(inp);
                }
                "compute" => {
                    if compute.is_some() {
                        return Err(nom::Err::Failure(E::from_error_kind(i2, ErrorKind::Verify)));
                    }
                    compute = comp;
                }
                "default" => {
                    if default.is_some() {
                        return Err(nom::Err::Failure(E::from_error_kind(i2, ErrorKind::Verify)));
                    }
                    default = def_or_when;
                }
                "when" => {
                    if when.is_some() {
                        return Err(nom::Err::Failure(E::from_error_kind(i2, ErrorKind::Verify)));
                    }
                    when = def_or_when;
                }
                _ => unreachable!(),
            }
            i = i2;
        } else {
            break;
        }
    }

    let (i, _) = ws(cut(char('}'))).parse(i)?;

    // Compute is required, inputs defaults to empty
    let compute =
        compute.ok_or_else(|| nom::Err::Failure(E::from_error_kind(i, ErrorKind::Verify)))?;
    let inputs = inputs.unwrap_or_default();

    Ok((
        i,
        Channel {
            identifier,
            inputs,
            compute,
            default,
            when,
        },
    ))
}

fn parse_program<
    'a,
    E: NomParseError<&'a str> + ContextError<&'a str> + FromExternalError<&'a str, ()>,
>(
    i: &'a str,
) -> Result<Program, nom::Err<E>> {
    let (i, items) = many0(alt((
        map(parse_function, |f| ("fn", Some(f), None)),
        map(parse_channel, |c| ("channel", None, Some(c))),
    )))
    .parse(i)?;

    let mut functions = Vec::new();
    let mut channels = Vec::new();

    for (_, func, chan) in items {
        if let Some(f) = func {
            functions.push(f);
        }
        if let Some(c) = chan {
            channels.push(c);
        }
    }

    let (i, _) = blank(i)?;
    let (_, _) = eof(i)?;
    Ok(Program {
        functions,
        channels,
    })
}
