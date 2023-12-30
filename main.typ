#import "style.typ": style, formatHeader

#show: doc => style(
  title: "Explorations of In-Context Reinforcement Learning",
  author: "Ethan Brooks",
  department: "Computer Science and Engineering",
  date: datetime(year: 2024, month: 1, day: 11),
  email: link("ethanbro@umich.edu"),
  orcid: "0000-0003-2557-4994",
  committee: (
    "Professor Satinder Singh, Chair",
    "Professor Richard L. Lewis, Co-Chair",
    "Professor Rada Mihalcea",
    "Professor Honglak Lee",
  ),
  doc: doc,
)

= Dedication

#lorem(80)

#include "acknowledgements.typ"

#outline(
  title: [#formatHeader(body: "Table of Contents")],
  depth: 2,
  indent: auto,
)
#outline(
  title: [#formatHeader(body: "List of Figures")],
  target: figure.where(kind: image),
)

// TODO: abbreviations

#set page(numbering: "1")
#counter(page).update(1)

#include "abstract.typ"
#include "introduction.typ"
#include "policy-iteration.typ"
#include "adpp.typ"
#include "bellman-update-networks.typ"
#include "conclusion.typ"

#bibliography("main.bib", style: "american-society-of-civil-engineers")