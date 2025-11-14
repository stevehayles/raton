use super::Type;
use crate::Value;
use std::{
    any::{Any, TypeId},
    fmt::Debug,
};

/// Values that may exist at runtime.
#[derive(Debug)]
pub enum RuntimeValue<'a> {
    /// See [`Value`].
    Value(Value),
    /// See [`Extern`].
    Extern(Extern<'a>),
}

impl RuntimeValue<'_> {
    /// Gets the [`Type`] of this value.
    pub fn type_of(&self) -> Type {
        match self {
            Self::Value(v) => v.type_of(),
            Self::Extern(e) => e.type_of(),
        }
    }

    /// Clones the value or, failing that, takes it (setting the original to [`Value::Null`]).
    pub fn clone_or_take(&mut self) -> Self {
        match self {
            Self::Value(value) => Self::Value(value.clone()),
            #[cfg(feature = "extern_value_type")]
            Self::Extern(Extern::Value(v)) => Self::Extern(Extern::Value(std::rc::Rc::clone(&*v))),
            Self::Extern(Extern::Ref(v)) => Self::Extern(Extern::Ref(*v)),
            v => std::mem::take(v),
        }
    }

    #[cfg(feature = "bool_type")]
    pub(crate) fn as_bool(&self) -> Result<bool, super::RuntimeError> {
        use super::Type;

        match self {
            Self::Value(Value::Bool(b)) => Ok(*b),
            _ => Err(super::RuntimeError::TypeMismatch {
                expected: Type::Bool,
                actual: self.type_of(),
            }),
        }
    }

    pub(crate) fn receiver_type_id_extern_type(&self) -> (TypeId, Option<Type>) {
        match self {
            Self::Value(_) => (TypeId::of::<Value>(), None),
            Self::Extern(e) => match e {
                #[cfg(feature = "extern_value_type")]
                Extern::Value(v) => ((**v).type_id(), Some(Type::ExternValue)),
                Extern::Ref(r) => ((**r).type_id(), Some(Type::ExternRef)),
                Extern::Mut(m) => ((**m).type_id(), Some(Type::ExternMut)),
            },
        }
    }
}

impl Default for RuntimeValue<'_> {
    fn default() -> Self {
        Self::Value(Default::default())
    }
}

/// An opaque Rust value from the host, outside the script.
#[non_exhaustive]
pub enum Extern<'a> {
    #[cfg(feature = "extern_value_type")]
    /// A value that can be passed to or returned from host functions.
    Value(std::rc::Rc<dyn Any>),
    /// A value that can be passed to host functions, but only returned in special cases.
    Ref(&'a dyn Any),
    /// A mutable value that can be passed to host functions, but only returned in special cases.
    Mut(&'a mut dyn Any),
}

impl Extern<'_> {
    /// Get the [`Type`] of this value.
    pub fn type_of(&self) -> Type {
        match self {
            #[cfg(feature = "extern_value_type")]
            Self::Value(_) => Type::ExternValue,
            Self::Ref(_) => Type::ExternRef,
            Self::Mut(_) => Type::ExternMut,
        }
    }
}

impl Debug for Extern<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "extern_value_type")]
            Self::Value(p) => write!(
                f,
                "ExternValue@{:x}",
                std::rc::Rc::as_ptr(p) as *const () as usize
            ),
            Self::Ref(p) => write!(f, "ExternRef@{:x}", *p as *const _ as *const () as usize),
            Self::Mut(p) => write!(f, "ExternMut@{:x}", *p as *const _ as *const () as usize),
        }
    }
}

impl<'b> PartialEq<RuntimeValue<'b>> for RuntimeValue<'_> {
    fn eq(&self, other: &RuntimeValue<'b>) -> bool {
        match (self, other) {
            (Self::Value(v), RuntimeValue::Value(v2)) => v == v2,
            (Self::Extern(e), RuntimeValue::Extern(e2)) => e == e2,
            _ => false,
        }
    }
}

impl<'b> PartialEq<Extern<'b>> for Extern<'_> {
    fn eq(&self, other: &Extern<'b>) -> bool {
        match (self, other) {
            #[cfg(feature = "extern_value_type")]
            (Self::Value(v), Extern::Value(v2)) => std::rc::Rc::ptr_eq(v, v2),
            (Self::Ref(v), Extern::Ref(v2)) => std::ptr::eq(&**v, &**v2),
            (Self::Mut(v), Extern::Mut(v2)) => std::ptr::eq(&**v, &**v2),
            _ => false,
        }
    }
}
