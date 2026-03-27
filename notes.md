# MiniTorch Concepts

---

## Module 0 — Fundamentals

### Operators

| Function | What it does | Gotcha |
|---|---|---|
| `sigmoid(x)` | Squashes input to (0, 1) | Crosses 0.5 at x=0, not x=0.5. Two-branch form avoids overflow. |
| `relu(x)` | Returns x if x > 0, else 0 | Not differentiable at x=0 — MiniTorch convention is derivative = 0 there |
| `relu_back(x, d)` | Backprop through relu | Passes gradient `d` through if x > 0, kills it otherwise |
| `inv_back(x, d)` | Backprop through 1/x | Chain rule: d * (-1/x²) |
| `log_back(x, d)` | Backprop through log | Chain rule: d * (1/x) |

### Why two-branch sigmoid?
Standard form `1 / (1 + e^{-x})` overflows when x is very negative (e^{-x} → ∞).
Alternate form `e^x / (1 + e^x)` is used for x < 0 — both are mathematically equal but numerically stable.

### Higher-Order Functions

| Function | Shape | Built with |
|---|---|---|
| `map(fn, ls)` | Apply fn to every element | — |
| `zipWith(fn, ls1, ls2)` | Apply fn pairwise | — |
| `reduce(fn, ls, start)` | Fold list into single value | — |
| `negList(ls)` | Negate every element | `map(neg, ls)` |
| `addLists(ls1, ls2)` | Elementwise add | `zipWith(add, ls1, ls2)` |
| `sum(ls)` | Sum all elements | `reduce(add, ls, 0.0)` |
| `prod(ls)` | Product of all elements | `reduce(mul, ls, 1.0)` |

### Property Testing (Hypothesis)
- Use `@given` to test properties over many random inputs, not just one hardcoded case
- `small_floats` is a custom strategy — avoids NaN/inf edge cases
- `assert_close` instead of `==` for floats — accounts for floating point error

### Gotchas
- `sigmoid(0.0) == 0.5`, not `sigmoid(0.5)`
- Shadowing builtins (`max`, `sum`, `map`) is intentional here but dangerous in general
- `_back` functions are chain rule applications — they always take `(x, d)` where `d` is the upstream gradient
- Never use `==` to compare floats after arithmetic — use `assert_close`
- `sigmoid` saturates to exactly 0.0 or 1.0 at extreme inputs due to float
  precision — bounds checks should be `>=` / `<=`, not strict `>` / `<`

---

## Youtube Notes
### Model
- models: parameterized functions
### Parameters
- modern parameters sets are both large and complex
- massive increaase in size and also complexity

#### Specifying parameters
- to handle parameters, we need DATA STRUCTURES
- 2 goals
1. independent of implementation (train model in one language, and deploy in another)
2. compostiional (parameters should fit the part that they belong to)

### Module Tree
- To handle the massive scale and complexity of parameters, PyTorch and MiniTorch use a declarative data structure known as a Module Tree
- benefits
1. can extract all params without knowing about themodules
2. can use mix and match modules:  meaning you can mix and match different modules (like parts from different research papers) without dealing with one giant binary blob of parameters

#### Structures of Modules
- every module inherits from the Module base class and can contain 3 members
1. explicit parameters
2. arbitrary user data
3. other sub-modules

