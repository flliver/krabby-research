# Physical units

This directory defines the physical quantities used by the firmware as distinct
C++ types. The goal is to prevent accidental unit mixing at compile time and
through static analysis, eliminating unit-conversion bugs from typed firmware
code.

For example, `Volts`, `Amps`, and `Watts` are not interchangeable even when
they use the same numeric representation. Code cannot pass an `Amps` value to
an API expecting `Volts`, or accidentally add the two.

Conversions between compatible units are explicit and named:

```cpp
const Amps current = MilliAmps(500.0f).toAmps();
```

## Raw-value boundaries

Hardware and third-party libraries define their own numeric and unit
conventions. Adapters at those boundaries must convert incoming values into
firmware unit types immediately and expose raw values only when calling back
into the external API.

The underlying representation is private. Calling `.value()` deliberately
leaves the unit-safe boundary and should be limited to integration points such
as:

- hardware drivers;
- telemetry and serial protocols;
- persistent storage;
- display formatting.

Operations involving a dimensionless scalar must be explicitly named so that
the raw scalar's role is visible at the call site. The result of such an
operation remains unit-typed.

Internal APIs should accept and return unit types instead of raw numeric values.
The type system can prevent unit mixing only while those types are preserved.
