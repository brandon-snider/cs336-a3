#import "@preview/ilm:1.4.1": *
#import "@preview/tablem:0.2.0": *

#let three-line-table = tablem.with(
  render: (columns: auto, ..args) => {
    table(
      columns: columns,
      stroke: none,
      align: center + horizon,
      table.hline(y: 0),
      table.hline(y: 1, stroke: .5pt),
      ..args,
      table.hline(),
    )
  }
)

#set text(lang: "en")

#show: ilm.with(
  title: [CS 336: Assignment 3],
  author: "Brandon Snider",
  date: datetime(year: 2025, month: 05, day: 06),
  figure-index: (enabled: true),
  table-index: (enabled: true),
  listing-index: (enabled: true),
  bibliography: bibliography("refs.bib")
)

#set enum(numbering: "a)")
#set heading(numbering: none)
#show link: underline

#set table(
    inset: 6pt, // default is 5pt
    stroke: (0.5pt + stroke-color),
)

== 2 Scaling Laws Review

== Problem (`chinchilla_isoflops`):  5 points

+ See `cs336_scaling/chinchilla_isoflops.py`

  #figure(image("out/chinchilla/params.png"), caption: "Chinchilla-optimal compute budget vs. parameter count")

  Empirical $angle.l C_i, N_"opt"(C_i) angle.r$ points obtained:

  #figure(tablem[
  | Compute (C) | Optimal Params (N) | Human-readable | Scientific |
  | ----------- | ------------------ | -------------- | ---------- |
  | 6e+18       | 762,093,419        | 762M           | 7.62e+08   |
  | 1e+19       | 806,647,749        | 806M           | 8.07e+08   |
  | 3e+19       | 1,536,852,354      | 1.54B          | 1.54e+09   |
  | 6e+19       | 1,952,041,776      | 1.95B          | 1.95e+09   |
  | 1e+20       | 3,253,402,960      | 3.25B          | 3.25e+09   |
  | 3e+20       | 5,903,836,027      | 5.90B          | 5.90e+09   |
  | 6e+20       | 6,971,055,968      | 6.97B          | 6.97e+09   |
  | 1e+21       | 6,859,328,563      | 6.86B          | 6.86e+09   |
  | 3e+21       | 12,148,905,329     | 12.15B         | 1.21e+10   |
  ], caption: "Empirically optimal parameter count for various compute budgets")

  Predicted optimal model sizes:

  $10^23 = "1e23" "FLOPs" -> 50,022,254,912 space (50B "or" "5e10") "parameters"$ \
  $10^24 = "1e24" "FLOPs" -> 126,757,785,319 space (126B "or" "1.27e11") "parameters"$

+ #figure(image("out/chinchilla/data.png"), caption: "Predicted optimal dataset size for various compute budgets")

  Empirical $angle.l C_i, D_"opt"(C_i) angle.r$ points obtained:

  #figure(tablem[
    | Compute (C) | Optimal Tokens (D) | Human-readable | Scientific |
    | ----------- | ------------------ | -------------- | ---------- |
    | 6e+18       | 1,312,175,089      | 1.31B          | 1.31e+09   |
    | 1e+19       | 2,066,164,157      | 2.07B          | 2.07e+09   |
    | 3e+19       | 3,253,402,961      | 3.25B          | 3.25e+09   |
    | 6e+19       | 5,122,841,182      | 5.12B          | 5.12e+09   |
    | 1e+20       | 5,122,841,182      | 5.12B          | 5.12e+09   |
    | 3e+20       | 8,469,069,901      | 8.47B          | 8.47e+09   |
    | 6e+20       | 14,345,028,996     | 14.35B         | 1.43e+10   |
    | 1e+21       | 24,297,810,658     | 24.30B         | 2.43e+10   |
    | 3e+21       | 41,155,971,378     | 41.16B         | 4.12e+10   |
    ], caption: "Empirically optimal total token count for various compute budgets"
  )

  Predicted optimal token dataset sizes:

  $10^23 = "1e23" "FLOPs" -> 333,185,033,259 space (333B "or" "3.33e11") "tokens"$ \
  $10^24 = "1e24" "FLOPs" -> 1,314,843,630,676 space (1.31T "or" "1.31e12") "tokens"$

