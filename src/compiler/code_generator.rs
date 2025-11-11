use crate::{ast::*, bytecode::*, Value};
use std::collections::BTreeMap;
use thiserror::Error;

/// Turns an abstract syntax tree into bytecode.
pub struct CodeGenerator {
    public_functions: BTreeMap<Identifier, PublicFunction>,
    channels: BTreeMap<Identifier, PublicChannel>,
    instructions: Vec<Instruction>,
    variable_stack: Vec<Vec<String>>,
    variable_count: u16,
    /// (start, breaks, continues)
    #[cfg(feature = "while_loop")]
    loop_stack: Vec<(u32, Vec<u32>, Vec<u32>)>,
    max_instructions: u32,
    max_depth: u32,
    max_local_variables: u16,
}

impl Default for CodeGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// An error produced when compiling a program into bytecode.
#[derive(Debug, Error)]
#[allow(missing_docs)]
pub enum CompileError {
    #[error("undefined variable ({name})")]
    UndefinedVariable { name: Identifier },
    #[error("undefined variable ({name})")]
    Shadowing { name: Identifier },
    #[error("max instructions exceeded")]
    MaxInstructionsExceeded,
    #[error("max local variables exceeded")]
    MaxLocalVariablesExceeded,
    #[error("max depth exceeded")]
    MaxDepthExceeded,
    #[error("arguments mismatch")]
    ArgumentsMismatch,
    #[error("internal compiler error")]
    Internal,
}

impl CodeGenerator {
    /// Create a configurable code generator.
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            variable_stack: Vec::new(),
            variable_count: 0,
            #[cfg(feature = "while_loop")]
            loop_stack: Vec::new(),
            public_functions: Default::default(),
            channels: Default::default(),
            max_instructions: 1_000_000,
            max_depth: 50,
            max_local_variables: 100,
        }
    }

    /// Return a [`CompileError`] if there would be more than `max` instructions.
    ///
    /// Default: 1 million
    pub fn with_max_instructions(mut self, max: u32) -> Self {
        self.max_instructions = max;
        self
    }

    /// Return a [`CompileError`] if there would be more than this many variables in a function,
    /// which includes the function's arguments.
    ///
    /// Default: 100
    pub fn with_max_local_variables(mut self, max: u16) -> Self {
        self.max_local_variables = max;
        self
    }

    /// Return a [`CompileError`] if nesting depth of expressions/statements exceeds this.
    ///
    /// Default: 50
    pub fn with_max_depth(mut self, max: u32) -> Self {
        self.max_depth = max;
        self
    }

    fn emit(&mut self, inst: Instruction) -> Result<u32, CompileError> {
        let addr = self.instructions.len() as u32;
        if addr >= self.max_instructions {
            return Err(CompileError::MaxInstructionsExceeded);
        }
        self.instructions.push(inst);
        Ok(addr)
    }

    #[allow(unused)]
    fn current_addr(&self) -> u32 {
        self.instructions.len() as u32
    }

    #[allow(unused)]
    fn patch_jump(&mut self, addr: u32, target: u32) -> Result<(), CompileError> {
        match &mut self.instructions[addr as usize] {
            Instruction::Jump(t) => {
                *t = target;
                Ok(())
            }
            #[cfg(feature = "bool_type")]
            Instruction::JumpIfFalse(t) => {
                *t = target;
                Ok(())
            }
            _ => {
                debug_assert!(false);
                Err(CompileError::Internal)
            }
        }
    }

    fn variables_so_far(&self) -> usize {
        self.variable_stack
            .iter()
            .flat_map(|vars| vars.iter())
            .count()
    }

    fn variable_index(&mut self, name: &Identifier) -> Result<u16, CompileError> {
        let reverse_index = self
            .variable_stack
            .iter()
            .rev()
            .flat_map(|vars: &Vec<String>| vars.iter().rev())
            .position(|v| v == name)
            .ok_or(CompileError::UndefinedVariable { name: name.clone() })?;
        let len = self.variables_so_far();

        self.variable_count = self.variable_count.max(len as u16);

        Ok((len - 1 - reverse_index) as u16)
    }

    fn generate_block(&mut self, block: &BlockExpression) -> Result<(), CompileError> {
        if self.variable_stack.len() >= self.max_depth as usize {
            return Err(CompileError::MaxDepthExceeded);
        }
        self.variable_stack.push(Vec::new());
        for stmt in &block.statements {
            self.generate_stmt(stmt)?;
        }
        if let Some(value) = &block.value {
            self.generate_expr(value)?;
        } else {
            self.emit(Instruction::LoadConstant(Value::Null))?;
        }
        self.variable_stack.pop().ok_or(CompileError::Internal)?;
        Ok(())
    }

    fn generate_expr(&mut self, expr: &Expression) -> Result<(), CompileError> {
        match expr {
            Expression::Literal(val) => {
                self.emit(Instruction::LoadConstant(val.clone()))?;
            }
            Expression::Variable(name) => {
                let index = self.variable_index(name)?;
                self.emit(Instruction::LoadVariable(index))?;
            }
            Expression::Unary(UnaryExpression { operand, operator }) => {
                self.generate_expr(operand)?;
                self.emit(Instruction::UnaryOperator(operator.clone()))?;
            }
            #[cfg(feature = "bool_type")]
            Expression::Binary(BinaryExpression {
                operator: crate::BinaryOperator::And,
                left,
                right,
            }) => {
                self.generate_expr(left)?;
                let jump_addr = self.emit(Instruction::JumpIfFalse(0))?;
                self.emit(Instruction::Pop)?;
                self.generate_expr(right)?;
                let end_addr = self.current_addr();
                self.patch_jump(jump_addr, end_addr)?;
            }
            #[cfg(feature = "bool_type")]
            Expression::Binary(BinaryExpression {
                operator: crate::BinaryOperator::Or,
                left,
                right,
            }) => {
                self.generate_expr(left)?;
                let jump_addr = self.emit(Instruction::JumpIfFalse(0))?;
                let jump_end = self.emit(Instruction::Jump(0))?;
                self.current_addr();
                let else_addr = self.emit(Instruction::Pop)?;
                self.patch_jump(jump_addr, else_addr)?;
                self.generate_expr(right)?;
                let end_addr = self.current_addr();
                self.patch_jump(jump_end, end_addr)?;
            }
            Expression::Binary(BinaryExpression {
                left,
                right,
                operator,
            }) => {
                self.generate_expr(left)?;
                self.generate_expr(right)?;
                self.emit(Instruction::BinaryOperator(operator.clone()))?;
            }
            #[cfg(feature = "method_call_expression")]
            Expression::MethodCall(MethodCallExpression {
                receiver,
                identifier,
                arguments,
            }) => {
                if arguments.len() + 1 > self.max_local_variables as usize {
                    // TODO: more specialized error.
                    return Err(CompileError::MaxLocalVariablesExceeded);
                }

                self.generate_expr(receiver)?;
                let receiver_location = if let Expression::Variable(name) = &**receiver {
                    let location = self.variable_index(name)?;
                    ReceiverLocation::Variable(location)
                } else {
                    ReceiverLocation::Temporary
                };
                for argument in arguments {
                    self.generate_expr(argument)?;
                }
                self.emit(Instruction::CallByName(
                    receiver_location,
                    identifier.clone(),
                    // TODO: Overflow handling.
                    arguments.len() as u16 + 1,
                ))?;
            }
            Expression::Call(CallExpression {
                identifier,
                arguments,
            }) => {
                if arguments.len() > self.max_local_variables as usize {
                    // TODO: more specialized error.
                    return Err(CompileError::MaxLocalVariablesExceeded);
                }
                for argument in arguments {
                    self.generate_expr(argument)?;
                }
                self.emit(Instruction::CallByName(
                    ReceiverLocation::None,
                    identifier.clone(),
                    arguments.len() as u16,
                ))?;
            }
            #[cfg(feature = "if_expression")]
            Expression::If(IfExpression {
                condition: cond,
                then_branch,
                else_branch,
            }) => {
                self.generate_expr(cond)?;
                let jump_else = self.emit(Instruction::JumpIfFalse(0))?;
                self.emit(Instruction::Pop)?;

                self.generate_block(then_branch)?;

                let jump_end = self.emit(Instruction::Jump(0))?;
                let else_addr = self.current_addr();
                self.patch_jump(jump_else, else_addr)?;
                self.emit(Instruction::Pop)?;

                if let Some(else_stmts) = else_branch {
                    self.generate_block(else_stmts)?;
                } else {
                    self.emit(Instruction::LoadConstant(Value::Null))?;
                }

                let end_addr = self.current_addr();
                self.patch_jump(jump_end, end_addr)?;
            }
            Expression::Block(block) => {
                self.generate_block(block)?;
            }
        }
        Ok(())
    }

    fn generate_stmt(&mut self, stmt: &Statement) -> Result<(), CompileError> {
        match stmt {
            Statement::Let(LetStatement {
                identifier,
                expression,
            }) => {
                if self.variable_index(identifier).is_ok() {
                    return Err(CompileError::Shadowing {
                        name: identifier.clone(),
                    });
                } else if self.variables_so_far() >= self.max_local_variables as usize {
                    return Err(CompileError::MaxLocalVariablesExceeded);
                }
                if let Some(last) = self.variable_stack.last_mut() {
                    last.push(identifier.clone());

                    let index = self
                        .variable_index(identifier)
                        .map_err(|_| CompileError::Internal)?;

                    self.generate_expr(expression)?;
                    self.emit(Instruction::StoreVariable(index))?;
                }
            }
            Statement::Assign(AssignStatement {
                identifier,
                expression,
            }) => {
                let index = self.variable_index(identifier)?;
                self.generate_expr(expression)?;
                self.emit(Instruction::StoreVariable(index))?;
            }
            Statement::Expression(expr) => {
                self.generate_expr(expr)?;
                self.emit(Instruction::Pop)?;
            }
            #[cfg(feature = "while_loop")]
            Statement::While(WhileLoop { condition, body }) => {
                if self.variable_stack.len() >= self.max_depth as usize {
                    return Err(CompileError::MaxDepthExceeded);
                }

                let loop_start = self.current_addr();
                self.generate_expr(condition)?;
                let jump_end = self.emit(Instruction::JumpIfFalse(0))?;
                self.emit(Instruction::Pop)?;

                self.loop_stack.push((loop_start, Vec::new(), Vec::new()));
                self.variable_stack.push(Vec::new());

                for stmt in body {
                    self.generate_stmt(stmt)?;
                }

                self.variable_stack.pop().ok_or(CompileError::Internal)?;

                self.emit(Instruction::Jump(loop_start))?;
                let end_addr = self.current_addr();
                self.patch_jump(jump_end, end_addr)?;
                self.emit(Instruction::Pop)?;
                let break_target_addr = self.current_addr();

                let (_, breaks, continues) = self.loop_stack.pop().unwrap();
                for break_addr in breaks {
                    self.patch_jump(break_addr, break_target_addr)?;
                }
                for continue_addr in continues {
                    self.patch_jump(continue_addr, loop_start)?;
                }
            }
            #[cfg(feature = "while_loop")]
            Statement::Break => {
                let addr = self.emit(Instruction::Jump(0))?;
                if let Some((_, breaks, _)) = self.loop_stack.last_mut() {
                    breaks.push(addr);
                }
            }
            #[cfg(feature = "while_loop")]
            Statement::Continue => {
                let addr = self.emit(Instruction::Jump(0))?;
                if let Some((_, _, continues)) = self.loop_stack.last_mut() {
                    continues.push(addr);
                }
            }
            Statement::Return(ReturnStatement { value }) => {
                if let Some(expr) = value {
                    self.generate_expr(expr)?;
                } else {
                    self.emit(Instruction::LoadConstant(Value::Null))?;
                }
                self.emit(Instruction::Return)?;
            }
        }
        Ok(())
    }

    fn generate_function(&mut self, func: &Function) -> Result<(), CompileError> {
        if self.public_functions.contains_key(&func.identifier) {
            return Err(CompileError::Shadowing {
                name: func.identifier.clone(),
            });
        }
        if func.arguments.len() > self.max_local_variables as usize {
            return Err(CompileError::MaxLocalVariablesExceeded);
        }
        self.variable_count = 0;
        let ip = self.emit(Instruction::AllocVariables(0))?;
        self.public_functions.insert(
            func.identifier.clone(),
            PublicFunction {
                address: ip,
                arguments: func.arguments.len() as u16,
            },
        );

        self.variable_stack.clear();
        self.variable_stack.push(func.arguments.clone());
        self.generate_block(&func.body)?;
        if !matches!(self.instructions.last(), Some(Instruction::Return)) {
            self.emit(Instruction::Return)?;
        }

        self.instructions[ip as usize] = Instruction::AllocVariables(self.variable_count);

        Ok(())
    }

    fn generate_compute_block(&mut self, block: &ComputeBlock) -> Result<(), CompileError> {
        if self.variable_stack.len() >= self.max_depth as usize {
            return Err(CompileError::MaxDepthExceeded);
        }
        self.variable_stack.push(Vec::new());

        for stmt in &block.statements {
            self.generate_stmt(stmt)?;
        }

        // ComputeBlock always has a value (required)
        self.generate_expr(&block.value)?;

        self.variable_stack.pop().ok_or(CompileError::Internal)?;
        Ok(())
    }

    fn generate_channel(&mut self, channel: &Channel) -> Result<(), CompileError> {
        if self.public_functions.contains_key(&channel.identifier) {
            return Err(CompileError::Shadowing {
                name: channel.identifier.clone(),
            });
        }
        if channel.inputs.len() > self.max_local_variables as usize {
            return Err(CompileError::MaxLocalVariablesExceeded);
        }

        // Generate compute block
        self.variable_count = 0;
        let compute_ip = self.emit(Instruction::AllocVariables(0))?;

        self.variable_stack.clear();
        self.variable_stack.push(channel.inputs.clone());
        self.generate_compute_block(&channel.compute)?;
        if !matches!(self.instructions.last(), Some(Instruction::Return)) {
            self.emit(Instruction::Return)?;
        }
        self.instructions[compute_ip as usize] = Instruction::AllocVariables(self.variable_count);

        // Generate default if present
        let default_address = if let Some(ref expr) = channel.default {
            self.variable_count = 0;
            let default_ip = self.emit(Instruction::AllocVariables(0))?;

            self.variable_stack.clear();
            self.variable_stack.push(channel.inputs.clone());
            self.generate_expr(expr)?;
            self.emit(Instruction::Return)?;

            self.instructions[default_ip as usize] =
                Instruction::AllocVariables(self.variable_count);
            Some(default_ip)
        } else {
            None
        };

        // Generate when if present
        let when_address = if let Some(ref expr) = channel.when {
            self.variable_count = 0;
            let when_ip = self.emit(Instruction::AllocVariables(0))?;

            self.variable_stack.clear();
            self.variable_stack.push(channel.inputs.clone());
            self.generate_expr(expr)?;
            self.emit(Instruction::Return)?;

            self.instructions[when_ip as usize] = Instruction::AllocVariables(self.variable_count);
            Some(when_ip)
        } else {
            None
        };

        // Insert the channel into the map (channels field should be added to CodeGenerator)
        self.channels.insert(
            channel.identifier.clone(),
            PublicChannel {
                inputs: channel.inputs.len() as u16,
                compute_address: compute_ip,
                default_address,
                when_address,
            },
        );

        Ok(())
    }

    /// Generate bytecode from a program abstract-syntax-tree.
    pub fn generate_program(mut self, program: &Program) -> Result<ProgramBytecode, CompileError> {
        for function in &program.functions {
            self.generate_function(function)?;
        }
        for channel in &program.channels {
            self.generate_channel(channel)?;
        }
        for instruction in &mut self.instructions {
            if let Instruction::CallByName(ReceiverLocation::None, name, args) = instruction {
                if let Some(func) = self.public_functions.get(&*name) {
                    if func.arguments != *args {
                        return Err(CompileError::ArgumentsMismatch);
                    }
                    *instruction = Instruction::CallByAddress(func.address, *args);
                }
            }
        }

        Ok(ProgramBytecode {
            public_functions: self.public_functions,
            channels: self.channels,
            instructions: self.instructions,
        })
    }
}
