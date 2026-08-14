# Tools

Development-only utilities belong here, including reference fixture generation,
intermediate-output comparison, compiled-function inspection, and benchmark result
processing. These tools are not part of the serving runtime.

Infurnace consumes GGUF directly through tinygrad, so the tools directory does not
define a separate production weight format.
