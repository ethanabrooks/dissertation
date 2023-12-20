#import "style.typ": style
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

= Table of Contents

#outline(title: none)

= List of Figures

#outline(title: none, target: figure.where(kind: image))