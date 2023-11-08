---
layout: default
title: Contributor Guide
nav_order: 3
---

# Contributor Guide

If you wish to contribute a new feature or a bug fix, please follow the workflow explained in this document.

## Table of Content

- [Prerequisites](#prerequisites)
- [Pull Requests](#pull-requests)
- [Submitting a Pull Request](#submitting-a-pull-request-)
- [Submitting an Issue/Feature Request](#submitting-an-issuefeature-request-)
- [Contributor Guidelines](#contributor-guidelines-)

## Prerequisites

- Experience with `git` command line basics.
- Familiarity with build toolset and development environment of your choice.

## Pull Requests

- **DO** all your work in your fork of the project repository. 
- **DO** base your work against the `main` branch.
- **DO** submit all major changes to code via pull requests (PRs) rather than through
  a direct commit. Contributors with commit access may submit trivial patches or changes to the project
  infrastructure configuration via direct commits (CAUTION!)
- **DO NOT** mix independent, unrelated changes in one PR.
  Separate unrelated fixes into separate PRs, especially if they are in different components of the project.
- **DO** give PRs short-but-descriptive names (e.g. "[subsection_tag] Add test for algorithm XXX", 
  not "Fix #1234").
- **DO** name your PR in a meaningful manner and start with a relevant tag for the targeted 
  subsection of the project (Ex. [DIP] Develop support for Fast Fourier Transform, [DIP] replace 
  C-style variadic function with std::vector, etc.). 
- **DO** provide a description summarising the proposed changes in your PR. This is a good place to 
  cite related issues and other PRs in context of your work as well. 
- **DO** [refer](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/autolinked-references-and-urls) to any 
  relevant issues, and include the [keywords](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue) 
  that automatically close issues when the PR is merged.
- **DO** [mention](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax#mentioning-people-and-teams) 
  any users that should know about and/or review the change.
- **DO** ensure any new features or changes to existing behaviours are covered with test cases.
- **DO** address PR feedback in an additional commit(s) rather than amending the existing commits.
  This makes it easier for reviewers to track changes.
- **DO** assume that the [Squash and Merge] will be used to merge your commit unless you
  request otherwise in the PR.

### Merging Pull Requests (for maintainers with write access)

- **DO** use [Squash and Merge] by default for individual contributions unless requested
  by the PR author. Do so, even if the PR contains only one commit. It creates a simpler
  history than [Create a Merge Commit]. Reasons that PR authors may request the true
  merge recording a merge commit may include (but are not limited to):
  - The change is easier to understand as a series of focused commits.
    Each commit in the series must be buildable so as not to break git bisect.

## Submitting a Pull Request : 

1. Follow [Forking Projects](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) guide to get personal copy of 
   [buddy-mlir](https://github.com/buddy-compiler/buddy-mlir) repository from where you will be able to submit new contributions as 
   [pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).

2. Clone your fork in your local system.

3. Create a new branch based on `main` and start working on your code inside this branch. Relevant git commands are : 

   ```shell
   git branch foo
   git checkout foo
   ```

4. Follow the [Getting started](#getting-started-with-git-workflow) workflow to set up the project on your system.

5. Make changes in your branch and push them to the upstream copy of your fork. Relevant git commands are : 

   ```shell
   git add .
   git commit -m "Add feature foo1"
   git push origin
   ```

6. You should update your local pull request branch with upstream main branch and your local llvm submodule with upstream specified llvm submodule before
   submitting a Pull request. Relevant git commands are : 

   ```shell
   cd buddy-mlir
   git pull https://github.com/buddy-compiler/buddy-mlir.git main
   git submodule update --init --recursive
   ```
    And then build llvm and buddy-mlir to make sure your commits are based on the latest buddy-mlir.

    Note : You should build your changes against updated `buddy-mlir` and `llvm` submodule before opening a PR (Pull Request).

7. Follow [this](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)
   for opening a pull request using GitHub GUI.

## Submitting an Issue/Feature Request : 

We welcome relevant and descriptive issues/feature requests for this project. Deafault templates are provided for creating both of them. Although it is recommended to use provided templates while creating an issue/feature request, it is not a compulsion and you can choose not to use them. Please ensure that your request/issue is self-explanable and provides sufficient technical info in latter case.



## Contributor Guidelines : 

- Name files in a meaningful way.
- Put copyright and license information in every relevant file.
- All non-public definitions should reside in scope enclosed by `namespace detail {...}`.
- Use the provided `clang-format` file for applying appropriate formatting.
- It is advisable to apply `clang-format` in a separate commit after making changes to the code.
- Indent with 2 spaces while using a new line.
- Remove any trailing whitespace at the end of each line.
- Leave one single blank line at the end of each file.
- Group variable declaration(s) and loops separately in files containing code.
- Leave space after commas in .mlir files.
- Leave space before and after '->', ':' and '=' in .mlir files.
