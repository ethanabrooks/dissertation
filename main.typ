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

= Acknowledgements

#lorem(80)

#outline(
  title: [#formatHeader(body: "Table of Contents")],
  target: heading.where(numbering: none),
)
\
#text(12pt, weight: "bold")[Chapter]
#outline(title: none, target: heading.where(numbering: "1.1"))

#outline(
  title: [#formatHeader(body: "List of Figures")],
  target: figure.where(kind: image),
)

#set heading(numbering: "1.1")
#set page(numbering: "1")
#counter(page).update(1)

#include "introduction.typ"
= In-Context Policy Iteration
= Algorithm Distillation + Policy Iteration
= Bellman Update Networks
= Conclusion

#bibliography("main.bib", style: "association-for-computing-machinery")