---
layout: default
title: Code Style
parent: Documentation
---

## Buddy Compiler Code Styles

### Python Code Formatting

Buddy Compiler adheres to the PEP 8 style guide while following the LLVM's formatting approach.
See the [PEP8](https://peps.python.org/pep-0008/) and [LLVM documentation](https://llvm.org/docs/CodingStandards.html#python-version-and-source-code-formatting) for details on code style and formatting.
Here is a list of some key points:

- Formatting: Use `black` and `darker` to format the code.

```
$ pip install black=='23.*' darker # install black 23.x and darker
$ darker test.py                   # format uncommitted changes
$ darker -r HEAD^ test.py          # also format changes from last commit
$ black test.py                    # format entire file
```

- Indentation: Use 4 spaces per indentation level. Do not use tabs.

- Maximum Line Length: Limit all lines to a maximum of 79 characters for code, and 72 for comments and docstrings.

- Imports: Imports should usually be on separate lines and grouped in the following order:

    - Standard library imports.
    - Related third-party imports.
    - Local application/library specific imports.
    - Place a blank line between each group of imports.

- Whitespace: Use whitespace around binary operators; for instance: a = b + c. Avoid extraneous whitespace.

- Comments: Add docstrings in Google style for all public functions and classes.

- Naming Conventions:

    - Use `CamelCase` for class names.
    - Use `snake_case` for functions, methods, and variable names.
    - Use `UPPERCASE_WITH_UNDERSCORES` for constants.

- Whitespace in Expressions and Statements: Avoid extraneous whitespace in the following situations:

    - Immediately inside parentheses, brackets, or braces.
    - Immediately before a comma, semicolon, or colon.
    - Immediately before the open parenthesis that starts the argument list of a function call.
    - Immediately before the open parenthesis that starts an indexing or slicing.

- Module Names:

    - Modules should have short, all-lowercase names.
    - Underscores can be used in the module name if it improves readability.
    - Python packages (directories) should also have short, all-lowercase names, but it's preferable not to use underscores.

    For example:

    `my_module.py` is preferred over `MyModule.py` or `mymodule.py`.
    For a package (directory), `mypackage` is preferred over `my_package`.

- Other:
    - Use two blank lines to separate top-level function and class definitions.
    - Use a single blank line to separate method definitions inside a class.
