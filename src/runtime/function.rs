// For macro.
#![allow(unused_braces)]

use super::{Extern, RuntimeError, RuntimeValue, Type};
use crate::Value;
use std::any::TypeId;

/// Type-erased function, callable by a script.
pub(crate) struct ErasedFunction<'data, 'func> {
    // TODO: SmallBox?
    #[allow(clippy::type_complexity)]
    inner: Box<
        dyn FnMut(&mut [RuntimeValue<'data>]) -> Result<RuntimeValue<'data>, RuntimeError> + 'func,
    >,
}

impl<'data, 'func> ErasedFunction<'data, 'func> {
    /// Erase the type of a function.
    pub(crate) fn new<A, R, F: Function<'data, A, R> + 'func>(mut inner: F) -> Self {
        Self {
            inner: Box::new(move |arguments| inner.call(arguments)),
        }
    }

    /// Call the type-erased function.
    pub(crate) fn call(
        &mut self,
        arguments: &mut [RuntimeValue<'data>],
    ) -> Result<RuntimeValue<'data>, RuntimeError> {
        (self.inner)(arguments)
    }
}

/// A Rust function callable by a script.
pub trait Function<'a, A, R>: Send + Sync {
    /// The number of arguments.
    const ARGS: usize;

    /// Get the [`TypeId`] of the receiver, if this is a method.
    fn receiver_type_id_extern_type() -> Option<(TypeId, Option<Type>)>;

    /// Call the function.
    fn call(
        &mut self,
        arguments: &mut [RuntimeValue<'a>],
    ) -> Result<RuntimeValue<'a>, RuntimeError>;
}

/// Types that can be converted to a [`RuntimeValue`], allowing them to
/// be passed as arguments from the host to a script function.
pub trait ToRuntimeValue<'a>: Sized {
    /// Perform the conversion.
    fn to_value(self) -> RuntimeValue<'a>;
}

/// Types that can be converted from a [`RuntimeValue`], allowing them to
/// be passed as arguments from a script function to the host.
pub trait FromRuntimeValue<'a>: Sized {
    /// The specific type this is looking for, if any. This is used to
    /// improve errors.
    const TYPE: Option<Type> = None;

    /// If this is eventually used in a [`Receiver`], the relevant type.
    ///
    /// Must compute this before creating a non-`'static` reference, which
    /// wouldn't support `TypeId` without `unsafe` code.
    ///
    /// Only `None` for `RuntimeValue<'_>`.
    fn type_id() -> Option<TypeId>;

    /// Perform the conversion, returning [`None`] if the [`RuntimeValue`]
    /// is of an incompatible type.
    fn from_value(value: &mut RuntimeValue<'a>) -> Option<Self>;
}

impl<'a> ToRuntimeValue<'a> for RuntimeValue<'a> {
    fn to_value(self) -> RuntimeValue<'a> {
        self
    }
}

impl<'a> FromRuntimeValue<'a> for RuntimeValue<'a> {
    fn type_id() -> Option<TypeId> {
        None
    }

    fn from_value(value: &mut RuntimeValue<'a>) -> Option<Self> {
        Some(std::mem::take(value))
    }
}

macro_rules! both_ways {
    ($v:ident, $t:path, $typeid:tt) => {
        impl<'a> ToRuntimeValue<'a> for $t {
            fn to_value(self) -> RuntimeValue<'a> {
                RuntimeValue::$v(self)
            }
        }

        impl<'a, 'b> FromRuntimeValue<'a> for $t {
            fn type_id() -> Option<TypeId> {
                $typeid
            }

            fn from_value(value: &mut RuntimeValue<'a>) -> Option<Self> {
                if let RuntimeValue::$v(v) = std::mem::take(value) {
                    Some(v)
                } else {
                    None
                }
            }
        }
    };
}

both_ways!(Value, Value, { Some(TypeId::of::<Value>()) });
both_ways!(Extern, Extern<'a>, None);

/// A shared [`std::rc::Rc`]-reference to a host value, which may be cheaply copied in a script.
///
/// Since [`std::rc::Rc`] is heap-allocated, requiring no value to reference it's easier to
/// create this type of extern value in a host function called by the script.
#[cfg(feature = "extern_value_type")]
pub struct ExternValue<T>(pub std::rc::Rc<T>);

/// An immutable reference to a host value, which may be freely copied in a script.
pub struct ExternRef<'a, T>(pub &'a T);

/// A mutable reference to a host value, which is taken when used in a script.
pub struct ExternMut<'a, T>(pub &'a mut T);

/// Wrap the first function argument in this to indicate it is a receiver, such that
/// two functions with the same name can be disambiguated by which type they are
/// called on.
#[cfg(feature = "method_call_expression")]
pub struct Receiver<'a, 'b, T>
where
    'a: 'b,
{
    borrow: &'b mut RuntimeValue<'a>,
    _spooky: std::marker::PhantomData<&'b T>,
}

#[cfg(feature = "method_call_expression")]
impl std::ops::Deref for Receiver<'_, '_, Value> {
    type Target = Value;

    fn deref(&self) -> &Self::Target {
        match &*self.borrow {
            RuntimeValue::Value(v) => v,
            // Wrong receiver chosen.
            _ => unreachable!(),
        }
    }
}

#[cfg(feature = "method_call_expression")]
impl std::ops::DerefMut for Receiver<'_, '_, Value> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self.borrow {
            RuntimeValue::Value(v) => v,
            // Wrong receiver chosen.
            _ => unreachable!(),
        }
    }
}

#[cfg(all(feature = "method_call_expression", feature = "extern_value_type"))]
impl<T: 'static> std::ops::Deref for Receiver<'_, '_, ExternValue<T>> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match &*self.borrow {
            RuntimeValue::Extern(Extern::Value(v)) => (**v).downcast_ref().unwrap(),
            // Wrong receiver chosen.
            _ => unreachable!(),
        }
    }
}

#[cfg(feature = "method_call_expression")]
impl<'b, 'a: 'b, T: 'static> std::ops::Deref for Receiver<'a, 'b, ExternRef<'_, T>> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match &*self.borrow {
            RuntimeValue::Extern(Extern::Ref(v)) => (**v).downcast_ref().unwrap(),
            // Wrong receiver chosen.
            _ => unreachable!(),
        }
    }
}

#[cfg(feature = "method_call_expression")]
impl<'b, 'a: 'b, T: 'static> std::ops::Deref for Receiver<'a, 'b, ExternMut<'_, T>> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match &*self.borrow {
            RuntimeValue::Extern(Extern::Mut(v)) => (**v).downcast_ref().unwrap(),
            // Wrong receiver chosen.
            _ => unreachable!(),
        }
    }
}

#[cfg(feature = "method_call_expression")]
impl<'b, 'a: 'b, T: 'static> std::ops::DerefMut for Receiver<'a, 'b, ExternMut<'_, T>> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self.borrow {
            RuntimeValue::Extern(Extern::Mut(v)) => (**v).downcast_mut().unwrap(),
            // Wrong receiver chosen.
            _ => unreachable!(),
        }
    }
}

#[cfg(feature = "extern_value_type")]
impl<'a, T: 'static> ToRuntimeValue<'a> for ExternValue<T> {
    fn to_value(self) -> RuntimeValue<'a> {
        RuntimeValue::Extern(Extern::Value(self.0))
    }
}

#[cfg(feature = "extern_value_type")]
impl<'a, T: 'static> FromRuntimeValue<'a> for ExternValue<T> {
    const TYPE: Option<Type> = Some(Type::ExternValue);

    fn type_id() -> Option<TypeId> {
        Some(TypeId::of::<T>())
    }

    fn from_value(value: &mut RuntimeValue<'a>) -> Option<Self> {
        if let RuntimeValue::Extern(Extern::Value(value)) = std::mem::take(value) {
            value.downcast().ok().map(ExternValue)
        } else {
            None
        }
    }
}

impl<'a, T: 'static> ToRuntimeValue<'a> for ExternRef<'a, T> {
    fn to_value(self) -> RuntimeValue<'a> {
        RuntimeValue::Extern(Extern::Ref(self.0))
    }
}

impl<'a, T: 'static> FromRuntimeValue<'a> for ExternRef<'a, T> {
    const TYPE: Option<Type> = Some(Type::ExternRef);

    fn type_id() -> Option<TypeId> {
        Some(TypeId::of::<T>())
    }

    fn from_value(value: &mut RuntimeValue<'a>) -> Option<Self> {
        if let RuntimeValue::Extern(Extern::Ref(value)) = std::mem::take(value) {
            value.downcast_ref().map(ExternRef)
        } else {
            None
        }
    }
}

impl<'a, T: 'static> ToRuntimeValue<'a> for ExternMut<'a, T> {
    fn to_value(self) -> RuntimeValue<'a> {
        RuntimeValue::Extern(Extern::Mut(self.0))
    }
}

impl<'a, T: 'static> FromRuntimeValue<'a> for ExternMut<'a, T> {
    const TYPE: Option<Type> = Some(Type::ExternMut);

    fn type_id() -> Option<TypeId> {
        Some(TypeId::of::<T>())
    }

    fn from_value(value: &mut RuntimeValue<'a>) -> Option<Self> {
        if let RuntimeValue::Extern(Extern::Mut(value)) = std::mem::take(value) {
            value.downcast_mut().map(ExternMut)
        } else {
            None
        }
    }
}

#[allow(unused)]
macro_rules! impl_convert_argument {
    ($v:ident, $t:ident) => {
        impl<'a> ToRuntimeValue<'a> for $t {
            fn to_value(self) -> RuntimeValue<'a> {
                RuntimeValue::Value(Value::$v(self))
            }
        }

        impl<'a, 'b> FromRuntimeValue<'a> for $t {
            const TYPE: Option<Type> = Some(Type::$v);

            fn type_id() -> Option<TypeId> {
                Some(TypeId::of::<Value>())
            }

            fn from_value(value: &mut RuntimeValue<'a>) -> Option<Self> {
                if let RuntimeValue::Value(Value::$v(v)) = std::mem::take(value) {
                    Some(v)
                } else {
                    None
                }
            }
        }
    };
}

#[cfg(feature = "bool_type")]
impl_convert_argument!(Bool, bool);
#[cfg(feature = "i32_type")]
impl_convert_argument!(I32, i32);
#[cfg(feature = "f32_type")]
impl_convert_argument!(F32, f32);
#[cfg(feature = "string_type")]
impl_convert_argument!(String, String);

/*
fn assert_arg<'a, T>()
where
    for<'b> Receiver<'a, 'b, T>: FromRuntimeValue<'a>
{}
//fn assert_arg<'a, A: for<'b> FromRuntimeValue<'a>>() {}
fn _test(){
    assert_arg::<Receiver<ExternValue<()>>>();//
}
*/

macro_rules! impl_function {
    ($($a: ident),*) => {
        impl<'a, $($a,)* R, FUNC: FnMut($($a),*) -> Result<R, RuntimeError> + Send + Sync> Function<'a, ((), ($($a,)*)), R> for FUNC
            where $($a: FromRuntimeValue<'a> + 'a,)*
                R: ToRuntimeValue<'a> {
            const ARGS: usize = 0 $(
                + {
                    let _ = std::mem::size_of::<$a>();
                    1
                }
            )*;

            fn receiver_type_id_extern_type() -> Option<(TypeId, Option<Type>)> {
                None
            }

            fn call(
                    &mut self,
                    mut _arguments: &mut [RuntimeValue<'a>],
                ) -> Result<RuntimeValue<'a>, RuntimeError> {
                if _arguments.len() != Self::ARGS {
                    return Err(RuntimeError::WrongNumberOfArguments{expected: Self::ARGS as u16, actual: _arguments.len() as u16});
                }
                let mut _i = 0;
                (self)($({
                    let (first, rest) = _arguments.split_at_mut(1);
                    _arguments = rest;
                    let arg = &mut first[0];
                    let type_of = arg.type_of();
                    <$a>::from_value(arg).ok_or(RuntimeError::InvalidArgument{
                        expected: $a::TYPE,
                        actual: type_of
                    })?
                }),*).map(move |v| v.to_value())
            }
        }
    };
}

// it took 4 days to figure out the lifetimes (╯°□°)╯︵ ┻━┻
#[cfg(feature = "method_call_expression")]
macro_rules! impl_method {
    ($($a: ident),*) => {
        impl<'a, RECEIVER, $($a,)* R: 'a, FUNC: for<'b> FnMut(Receiver<'a, 'b, RECEIVER>, $($a),*) -> Result<R, RuntimeError> + Send + Sync> Function<'a, (RECEIVER, (), ($($a,)*)), R> for FUNC
            where RECEIVER: FromRuntimeValue<'a> + 'a,
                $($a: FromRuntimeValue<'a>,)*
                R: ToRuntimeValue<'a> {
            const ARGS: usize = 1 $(
                + {
                    let _ = std::mem::size_of::<$a>();
                    1
                }
            )*;

            fn receiver_type_id_extern_type() -> Option<(TypeId, Option<Type>)> {
                <RECEIVER>::type_id().map(|type_id| (type_id, RECEIVER::TYPE.filter(|t| {
                    match t {
                        Type::ExternRef | Type::ExternMut => true,
                        #[cfg(feature = "extern_value_type")]
                        Type::ExternValue => true,
                        _ => false,
                    }
                })))
            }

            fn call(
                    &mut self,
                    mut _arguments: &mut [RuntimeValue<'a>],
                ) -> Result<RuntimeValue<'a>, RuntimeError> {
                if _arguments.len() != Self::ARGS {
                    return Err(RuntimeError::WrongNumberOfArguments{expected: Self::ARGS as u16, actual: _arguments.len() as u16});
                }
                let mut _i = 0;
                (self)(
                    {
                        let (first, rest) = _arguments.split_at_mut(1);
                        _arguments = rest;
                        let arg = &mut first[0];
                        Receiver{
                            borrow: arg,
                            _spooky: std::marker::PhantomData,
                        }
                    },
                $({
                    let (first, rest) = _arguments.split_at_mut(1);
                    _arguments = rest;
                    let arg = &mut first[0];
                    let type_of = arg.type_of();
                    <$a>::from_value(arg).ok_or(RuntimeError::InvalidArgument{
                        expected: $a::TYPE,
                        actual: type_of
                    })?
                }),*).map(move |v| v.to_value())
            }
        }
    };
}

macro_rules! impl_function_and_method {
    ($($a: ident),*) => {
        impl_function!($($a),*);
        #[cfg(feature = "method_call_expression")]
        impl_method!($($a),*);
    }
}

impl_function_and_method!();
impl_function_and_method!(A);
impl_function_and_method!(A, B);
impl_function_and_method!(A, B, C);
impl_function_and_method!(A, B, C, D);
impl_function_and_method!(A, B, C, D, E);
impl_function_and_method!(A, B, C, D, E, F);
impl_function_and_method!(A, B, C, D, E, F, G);
impl_function_and_method!(A, B, C, D, E, F, G, H);
impl_function_and_method!(A, B, C, D, E, F, G, H, I);
impl_function_and_method!(A, B, C, D, E, F, G, H, I, J);
