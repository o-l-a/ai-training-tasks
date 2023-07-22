# Tic-Tac-Toe Strategy using Minimax and Alpha-Beta Pruning

## Table of Contents
- [Introduction](#introduction)
- [Objective](#objective)
- [Example](#example)

## Introduction

This project aims to analyze an intelligent Tic-Tac-Toe playing strategy using the Minimax and Alpha-Beta Pruning algorithms.

## Objective

The primary objectives of this project are as follows:

- Implement the Minimax algorithm to evaluate game states.
- Implement Alpha-Beta Pruning to optimize the Minimax search process.
- Develop a Tic-Tac-Toe playing strategy that provides challenging gameplay.

### Game Values

During the implementation of the algorithm, the following values were adopted:

- "Max" player's victory: 1
- "Min" player's victory: -1
- Draw: 0

### Symbols

- "o" - circle (player's move)
- "x" - cross (opponent's move)
- "N" - empty cell on the game board

## Example
Each generated state is presented in the order adhering to the recursive process, annotated with a unique identifier (root denoted as 0) and its corresponding value, v.
```commandline
------------------------------------------------------------ 
MIN
[['o' 'x' 'o']
 ['x' 'o' 'x']
 ['x' 'o' 'o']] 

(#6, v=1)
------------------------------------------------------------ 
MAX
[['o' 'x' 'o']
 ['x' 'o' 'x']
 ['x' 'o' 'N']] 

(#4, v=1)
└──────────────(#6, v=1)
------------------------------------------------------------ 
MIN
[['o' 'x' 'o']
 ['x' 'o' 'o']
 ['x' 'o' 'x']] 

(#7, v=0)
------------------------------------------------------------ 
MAX
[['o' 'x' 'o']
 ['x' 'o' 'N']
 ['x' 'o' 'x']] 

(#5, v=0)
└──────────────(#7, v=0)
------------------------------------------------------------ 
MIN
[['o' 'x' 'o']
 ['x' 'o' 'N']
 ['x' 'o' 'N']] 

(#1, v=0)
└──────────────(#4, v=1)
               └──────────────(#6, v=1)
└──────────────(#5, v=0)
               └──────────────(#7, v=0)
------------------------------------------------------------ 
MIN
[['o' 'x' 'o']
 ['x' 'x' 'o']
 ['x' 'o' 'o']] 

(#10, v=1)
------------------------------------------------------------ 
MAX
[['o' 'x' 'o']
 ['x' 'x' 'o']
 ['x' 'o' 'N']] 

(#8, v=1)
└──────────────(#10, v=1)
------------------------------------------------------------ 
MIN
[['o' 'x' 'o']
 ['x' 'o' 'o']
 ['x' 'o' 'x']] 

(#11, v=0)
------------------------------------------------------------ 
MAX
[['o' 'x' 'o']
 ['x' 'N' 'o']
 ['x' 'o' 'x']] 

(#9, v=0)
└──────────────(#11, v=0)
------------------------------------------------------------ 
MIN
[['o' 'x' 'o']
 ['x' 'N' 'o']
 ['x' 'o' 'N']] 

(#2, v=0)
└──────────────(#8, v=1)
               └──────────────(#10, v=1)
└──────────────(#9, v=0)
               └──────────────(#11, v=0)
------------------------------------------------------------ 
MIN
[['o' 'x' 'o']
 ['x' 'x' 'o']
 ['x' 'o' 'o']] 

(#14, v=1)
------------------------------------------------------------ 
MAX
[['o' 'x' 'o']
 ['x' 'x' 'N']
 ['x' 'o' 'o']] 

(#12, v=1)
└──────────────(#14, v=1)
------------------------------------------------------------ 
MIN
[['o' 'x' 'o']
 ['x' 'o' 'x']
 ['x' 'o' 'o']] 

(#15, v=1)
------------------------------------------------------------ 
MAX
[['o' 'x' 'o']
 ['x' 'N' 'x']
 ['x' 'o' 'o']] 

(#13, v=1)
└──────────────(#15, v=1)
------------------------------------------------------------ 
MIN
[['o' 'x' 'o']
 ['x' 'N' 'N']
 ['x' 'o' 'o']] 

(#3, v=1)
└──────────────(#12, v=1)
               └──────────────(#14, v=1)
└──────────────(#13, v=1)
               └──────────────(#15, v=1)
------------------------------------------------------------ 
MAX
[['o' 'x' 'o']
 ['x' 'N' 'N']
 ['x' 'o' 'N']] 

(#0, v=1)
└──────────────(#1, v=0)
               └──────────────(#4, v=1)
                              └──────────────(#6, v=1)
               └──────────────(#5, v=0)
                              └──────────────(#7, v=0)
└──────────────(#2, v=0)
               └──────────────(#8, v=1)
                              └──────────────(#10, v=1)
               └──────────────(#9, v=0)
                              └──────────────(#11, v=0)
└──────────────(#3, v=1)
               └──────────────(#12, v=1)
                              └──────────────(#14, v=1)
               └──────────────(#13, v=1)
                              └──────────────(#15, v=1)
```
