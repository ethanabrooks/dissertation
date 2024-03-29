#import "style.typ": style, formatHeader

#show: doc => style(
  title: "Explorations of In-Context Reinforcement Learning",
  author: "Ethan Brooks",
  department: "Computer Science and Engineering",
  date: datetime(year: 2024, month: 1, day: 11),
  email: "ethanbro@umich.edu",
  orcid: "0000-0003-2557-4994",
  committee: (
    "Professor Satinder Singh, Chair",
    "Professor Richard L. Lewis, Co-Chair",
    "Professor Honglak Lee",
    "Professor Rada Mihalcea",
    "Professor Thad Polk",
  ),
  doc: doc,
)

#counter(page).update(2)
#include "dedication.typ"
#include "acknowledgements.typ"

#outline(
  title: [#formatHeader(body: "Table of Contents")],
  depth: 2,
  indent: 0.25in,
)
= List of Figures
#outline(
  title: none, // [#formatHeader(body: "List of Figures")],
  target: figure.where(kind: image),
)

#include "acronyms.typ"
#include "abstract.typ"

#set page(numbering: "1")
#counter(page).update(1)

#counter(heading).update(0)
#set heading(numbering: "1.1")
#include "introduction.typ"
#include "policy-iteration.typ"
#include "adpp.typ"
#include "bellman-update-networks.typ"
#include "conclusion.typ"

#bibliography("main.bib", style: "american-society-of-civil-engineers")