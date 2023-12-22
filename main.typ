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

// = Dedication

// #lorem(80)

// = Acknowledgements

// #lorem(80)

#outline(title: [#formatHeader(body: "Table of Contents")], indent: auto)
#outline(
  title: [#formatHeader(body: "List of Figures")],
  target: figure.where(kind: image),
)

#set heading(numbering: "1.1", supplement: "Chapter")
#set page(numbering: "1")
#counter(page).update(1)
#set par(first-line-indent: .5in)

#include "abstract.typ"
#include "introduction.typ"
#include "policy-iteration.typ"
#include "adpp.typ"

= Bellman Update Networks
= Conclusion

#bibliography("main.bib", style: "association-for-computing-machinery")