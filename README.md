# Regression
 Regression Analysis Utility

## Requirement
.NET 8.0  
[Algebra](https://github.com/tk-yoshimura/Algebra)  
[DoubleDouble](https://github.com/tk-yoshimura/DoubleDouble)

## Install
[Download DLL](https://github.com/tk-yoshimura/Regression/releases)  
[Download Nuget](https://www.nuget.org/packages/tyoshimura.regression/)  

## Usage

```csharp
static ddouble f(ddouble x, ddouble y) {
    return 0.5 + 0.25 * x + 0.125 * y + 0.0625 * x * y + 2 * x * x + 4 * y * y;
}

Vector x = new ddouble[] { 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4 };
Vector y = new ddouble[] { 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4 };

Vector z = Vector.Func(x, y, f);

RobustRegressor fitter = new([x, y, x * y, x * x, y * y], z);

Vector param = fitter.ExecuteFitting();
```

## Licence
[MIT](https://github.com/tk-yoshimura/Regression/blob/main/LICENSE)

## Author

[T.Yoshimura](https://github.com/tk-yoshimura)
